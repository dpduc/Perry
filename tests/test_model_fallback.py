"""Chat on the free model first, and on the paid one when the free one can't answer."""

from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import pytest
from fakes import ScriptedChatModel, call, fake_provider, last_message
from langchain_core.messages import AIMessage

from perry import models, usage
from perry.supervisor import run_turn


def daily_quota_used_up() -> Exception:
    return models.RateLimitError(
        "429 RESOURCE_EXHAUSTED: Quota exceeded for metric generate_content_free_tier_requests, "
        "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    )


def per_minute_limit() -> Exception:
    return models.RateLimitError(
        "429 RESOURCE_EXHAUSTED: quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
    )


def paid(text: str, input_tokens: int = 0, output_tokens: int = 0) -> AIMessage:
    """A reply from the paid model, named the way OpenAI names its snapshots."""
    return AIMessage(
        text,
        response_metadata={"model_name": "gpt-5.4-mini-2026-03-17"},
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


@pytest.fixture(autouse=True)
def fresh_quota():
    models.free_quota.reset()
    _threads.clear()
    yield
    models.free_quota.reset()


@pytest.fixture
def two_models(owner):
    """Turns answered by a free and a paid scripted model, free first."""
    free, fallback = ScriptedChatModel(), ScriptedChatModel()
    provider = fake_provider(free)
    provider = provider.__class__(
        **{**provider.__dict__, "chat_model": lambda: models.free_first(free, fallback)}
    )
    with models.use(provider):
        yield free, fallback


_threads: dict[str, str] = {}


def ask(text: str) -> str:
    """One Turn, all in one Thread per test (a test's first message opens it)."""
    from perry import threads, users

    thread = _threads.get("current") or threads.create(users.owner(), text)
    _threads["current"] = thread
    return run_turn(users.owner(), thread, text)


def test_the_free_model_answers_while_it_can(two_models):
    free, fallback = two_models
    free.steps = ["Hello from the free model."]

    assert ask("Hi") == "Hello from the free model."
    assert fallback.calls == []


def test_once_the_free_daily_quota_is_used_up_the_paid_model_answers_for_the_rest_of_the_day(
    two_models,
):
    free, fallback = two_models
    free.steps = [daily_quota_used_up()]
    fallback.steps = [paid("Hello from the paid model."), paid("Still the paid model.")]

    assert ask("Hi") == "Hello from the paid model."
    # The free model isn't asked again today: it has no replies left to give.
    assert ask("Again?") == "Still the paid model."
    assert len(free.calls) == 1
    assert models.free_quota.exhausted()


def test_a_per_minute_limit_falls_back_for_that_call_only(two_models):
    free, fallback = two_models
    free.steps = [per_minute_limit(), "The free model again."]
    fallback.steps = [paid("The paid model, once.")]

    assert ask("Hi") == "The paid model, once."
    assert ask("And now?") == "The free model again."
    assert not models.free_quota.exhausted()


def test_an_overloaded_free_model_falls_back(two_models):
    free, fallback = two_models
    free.steps = [TimeoutError("503 The model is overloaded")]
    fallback.steps = [paid("Answered by the paid model.")]

    assert ask("Hi") == "Answered by the paid model."



def test_a_free_call_that_times_out_falls_back(two_models):
    # What the Gemini client raises when a call outlasts PERRY_CHAT_TIMEOUT.
    free, fallback = two_models
    free.steps = [httpx.ReadTimeout("The read operation timed out")]
    fallback.steps = [paid("Answered by the paid model.")]

    assert ask("Hi") == "Answered by the paid model."

def test_a_bug_in_the_free_call_is_not_hidden_by_the_paid_model(two_models):
    free, fallback = two_models
    free.steps = [ValueError("bad request")]

    with pytest.raises(ValueError):
        ask("Hi")
    assert fallback.calls == []


def test_agents_and_their_tools_run_on_the_paid_model_after_the_fallback(two_models, tmp_path):
    from perry.ingest import ingest_file

    (tmp_path / "budget.md").write_text("The marketing budget for Q3 is 40,000 dollars.")
    ingest_file(tmp_path / "budget.md")
    free, fallback = two_models
    free.steps = [daily_quota_used_up()]
    fallback.steps = [
        call("ask_knowledge_agent", question="What is the Q3 marketing budget?"),  # Supervisor
        call("hybrid_search", question="Q3 marketing budget"),  # Knowledge Agent
        last_message,  # Knowledge Agent reports the Chunks it found
        last_message,  # Supervisor passes the findings on
    ]

    reply = ask("How big is my Q3 marketing budget?")

    assert reply.startswith("The marketing budget for Q3 is 40,000 dollars.")
    assert "Citations: [budget.md]" in reply


def test_paid_calls_are_priced_by_their_model_in_the_usage_ledger(two_models, prices):
    prices({"gpt-5.4-mini": {"input": 0.75, "output": 4.50}})
    free, fallback = two_models
    free.steps = [daily_quota_used_up()]
    fallback.steps = [paid("Hello.", input_tokens=1_000_000, output_tokens=0)]

    ask("Hi")

    assert usage.today().cost_usd == pytest.approx(0.75)


def test_without_a_paid_model_a_used_up_quota_stops_the_turn_and_says_so(owner):
    free = ScriptedChatModel(steps=[daily_quota_used_up()])
    provider = fake_provider(free)
    provider = provider.__class__(
        **{**provider.__dict__, "chat_model": lambda: models.free_first(free, None)}
    )
    with models.use(provider):
        reply = ask("Hi")

    assert "quota" in reply


def test_the_free_daily_quota_comes_back_at_midnight_pacific_time():
    pacific = ZoneInfo("America/Los_Angeles")
    quota = models.FreeQuota()

    quota.use_up(datetime(2026, 10, 9, 23, 30, tzinfo=pacific))

    assert quota.exhausted(datetime(2026, 10, 9, 23, 59, tzinfo=pacific))
    assert not quota.exhausted(datetime(2026, 10, 10, 0, 0, tzinfo=pacific))


def test_the_benchmark_pins_every_chat_call_to_the_paid_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # never called: no request is made

    with models.use(fake_provider(ScriptedChatModel())), models.paid_chat_only("gpt-5.4-nano"):
        model = models.chat_model()

    assert model.model_name == "gpt-5.4-nano"


def test_the_benchmark_needs_the_paid_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"), models.paid_chat_only():
        pass
