"""The daily cap: new Turns are refused once today's usage reaches it, until local midnight."""

import pytest
from fakes import call, last_message, with_usage


@pytest.fixture
def cap(monkeypatch, prices):
    """A daily cap of $0.01, with the scripted model at $1 per million tokens."""
    prices({"scripted": {"input": 1.0, "output": 1.0}})
    monkeypatch.setenv("PERRY_DAILY_CAP_USD", "0.01")


def test_new_turn_after_the_daily_cap_is_refused_with_when_it_resets(perry, cap):
    perry.script(with_usage("Hi! How can I help?", 10_000))  # $0.01: the whole cap
    perry.send("Hello")

    reply = perry.send("How big is my Q3 marketing budget?")  # no model call is scripted

    assert reply == (
        "Perry has used today's budget of $0.01, so it can't take new requests "
        "until the budget resets at midnight."
    )
    assert perry.history() == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
    ]


def test_turn_that_crosses_the_daily_cap_midway_finishes(perry, cap):
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    perry.script(
        with_usage(call("ask_knowledge_agent", question="Q3 budget?"), 6_000),  # Supervisor
        with_usage(call("search_documents", query="Q3 budget"), 6_000),  # cap crossed here
        with_usage("The Q3 marketing budget is 40,000 dollars [budget.md].", 6_000),
        with_usage(last_message, 6_000),
    )

    reply = perry.send("How big is my Q3 marketing budget?")

    assert reply == "The Q3 marketing budget is 40,000 dollars.\n\nCitations: [budget.md]"
    assert perry.send("And for Q4?").startswith("Perry has used today's budget of $0.01")


def test_daily_cap_defaults_to_one_dollar(perry, prices, monkeypatch):
    monkeypatch.delenv("PERRY_DAILY_CAP_USD", raising=False)
    prices({"scripted": {"input": 1.0, "output": 1.0}})
    perry.script(with_usage("Hi!", 999_999))  # just under $1
    perry.send("Hello")
    perry.script("Still here.")
    assert perry.send("Are you there?") == "Still here."

    perry.script(with_usage("Yes.", 1))  # brings today to exactly $1
    perry.send("Sure?")

    assert perry.send("Hello again").startswith("Perry has used today's budget of $1.00")


def test_new_message_after_the_daily_cap_still_cancels_the_pending_confirmation(
    perry, cap, workspace
):
    from perry.supervisor import Approve, Confirmation

    perry.script(
        with_usage(call("ask_task_agent", request="Email Sam that I'm late."), 10_000),
        call("send_email", to="sam@example.com", subject="Late", body="Running late."),
    )
    assert isinstance(perry.send("Tell Sam I'm late"), Confirmation)

    reply = perry.send("Hello?")  # no model call is scripted

    assert "cancelled" in reply.lower()
    assert reply.endswith("until the budget resets at midnight.")
    assert "nothing waiting" in perry.resume(Approve()).lower()
    assert workspace.sent == []
