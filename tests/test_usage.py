"""The usage ledger and the per-Turn token ceiling, seen through Turns and today's usage."""

import pytest
from fakes import call, last_message, with_usage


@pytest.fixture
def prices(tmp_path, monkeypatch):
    """Write a price table (USD per million tokens) and point Perry at it."""
    path = tmp_path / "prices.toml"
    monkeypatch.setenv("PERRY_PRICES_FILE", str(path))

    def write(table: dict[str, dict[str, float]]) -> None:
        lines = []
        for model, rates in table.items():
            lines.append(f'[models."{model}"]')
            lines += [f"{kind}_per_million = {rate}" for kind, rate in rates.items()]
        path.write_text("\n".join(lines), encoding="utf-8")

    return write


def ask_about_budget(perry, *, tokens: list[tuple[int, int]]) -> str:
    """A Turn that consults the Knowledge Agent: 4 chat calls with the given (input, output)
    tokens, and one embedding of the search query."""
    perry.script(
        with_usage(call("ask_knowledge_agent", question="Q3 budget?"), *tokens[0]),
        with_usage(call("search_documents", query="Q3 budget"), *tokens[1]),
        with_usage(last_message, *tokens[2]),
        with_usage(last_message, *tokens[3]),
    )
    return perry.send("How big is my Q3 marketing budget?")


def test_every_chat_and_embedding_call_of_a_turn_counts_in_todays_usage(perry, prices):
    from perry import usage

    prices({"scripted": {"input": 1.0, "output": 2.0}, "fake-embedding": {"input": 0.5}})
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")

    ask_about_budget(perry, tokens=[(1000, 100), (2000, 50), (3000, 200), (4000, 300)])

    today = usage.today()
    # Embedding tokens are estimated at 4 characters each: "Q3 budget" is 3 tokens.
    assert today.tokens == 10_650 + 3
    assert today.cost_usd == pytest.approx(10_000 * 1e-6 + 650 * 2e-6 + 3 * 0.5e-6)


def test_editing_the_price_table_prices_later_calls_without_repricing_earlier_ones(perry, prices):
    from perry import usage

    prices({"scripted": {"input": 1.0, "output": 1.0}})
    perry.script(with_usage("Hi!", 1000, 0))
    perry.send("Hello")

    prices({"scripted": {"input": 3.0, "output": 1.0}})
    perry.script(with_usage("Still here.", 1000, 0))
    perry.send("Are you there?")

    assert usage.today().cost_usd == pytest.approx(1000 * 1e-6 + 1000 * 3e-6)


def test_turn_that_hits_the_token_ceiling_stops_and_reports_what_it_found(perry, monkeypatch):
    monkeypatch.setenv("PERRY_TURN_TOKEN_CEILING", "1000")
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    perry.script(
        with_usage(call("ask_knowledge_agent", question="Q3 budget?"), 400),  # Supervisor
        with_usage(call("search_documents", query="Q3 budget"), 300),  # Knowledge Agent
        with_usage("The Q3 marketing budget is 40,000 dollars [budget.md].", 300),
        # The Supervisor's next call would go over the ceiling, so it never happens.
    )

    reply = perry.send("How big is my Q3 marketing budget?")

    assert reply.startswith("I stopped because this Turn reached its limit of 1,000 tokens.")
    assert "The Q3 marketing budget is 40,000 dollars [budget.md]." in reply
    assert perry.history() == [
        {"role": "user", "content": "How big is my Q3 marketing budget?"},
        {"role": "assistant", "content": reply},
    ]


def test_token_ceiling_counts_only_the_current_turn_and_can_stop_inside_an_agent(
    perry, monkeypatch
):
    monkeypatch.setenv("PERRY_TURN_TOKEN_CEILING", "1000")
    perry.script(with_usage("Hi! How can I help?", 900))
    perry.send("Hello")  # 900 tokens: under the ceiling, and they don't carry over
    perry.script(
        with_usage(call("ask_knowledge_agent", question="Q3 budget?"), 600),  # Supervisor
        with_usage(call("search_documents", query="Q3 budget"), 500),  # Knowledge Agent
        # The Knowledge Agent's next call would go over the ceiling, so it never happens.
    )

    reply = perry.send("How big is my Q3 marketing budget?")

    assert reply == (
        "I stopped because this Turn reached its limit of 1,000 tokens. "
        "I hadn't found anything yet. Try a narrower question."
    )
    assert perry.history() == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": "How big is my Q3 marketing budget?"},
        {"role": "assistant", "content": reply},
    ]
