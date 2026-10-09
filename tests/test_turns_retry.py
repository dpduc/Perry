"""Turns where an Agent step fails: retried once, then reported, never made up."""

from fakes import call, last_message

from perry import models


def rate_limited() -> Exception:
    return models.RateLimitError("429 RESOURCE_EXHAUSTED: quota exceeded")


def test_step_that_fails_once_then_succeeds_gives_a_normal_answer(perry):
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    perry.script(
        call("ask_knowledge_agent", question="What is the Q3 marketing budget?"),  # Supervisor
        rate_limited(),  # Knowledge Agent's first attempt fails
        call("hybrid_search", question="Q3 marketing budget"),  # retry
        last_message,  # Knowledge Agent reports the passages it found
        last_message,  # Supervisor passes the findings on
    )

    reply = perry.send("How big is my Q3 marketing budget?")

    assert reply.startswith("The marketing budget for Q3 is 40,000 dollars.")
    assert reply.endswith(
        "Citations: [budget.md]\nFound with: Hybrid search (picked logically)"
    )
    assert perry.consulted() == ["ask_knowledge_agent"]


def report_every_step(messages) -> str:
    """A Supervisor that answers by passing on what each step of this Turn returned."""
    last_user = max(i for i, m in enumerate(messages) if m.type == "human")
    return "\n\n".join(m.text for m in messages[last_user:] if m.type == "tool")


def two_step_turn_where_the_second_step_fails_twice(perry):
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    perry.add_document("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")
    perry.script(
        # Step 1 works.
        call("ask_knowledge_agent", question="What is the Q3 marketing budget?"),
        call("hybrid_search", question="Q3 marketing budget"),
        "The Q3 marketing budget is 40,000 dollars [budget.md].",
        # Step 2: the Knowledge Agent finds passages, then its model fails, twice.
        call("ask_knowledge_agent", question="Is the ridge path steep?"),
        call("hybrid_search", question="ridge path steep"),
        rate_limited(),
        call("hybrid_search", question="ridge path steep"),
        rate_limited(),
        report_every_step,
    )
    return perry.send("What's my Q3 budget, and is the ridge path steep?")


def test_step_that_fails_twice_is_named_and_the_other_results_are_kept(perry):
    reply = two_step_turn_where_the_second_step_fails_twice(perry)

    assert "The Q3 marketing budget is 40,000 dollars." in reply
    assert reply.endswith(
        "Citations: [budget.md]\nFound with: Hybrid search (picked logically)"
    )
    assert "ask_knowledge_agent(question='Is the ridge path steep?') failed twice" in reply
    assert perry.consulted() == ["ask_knowledge_agent", "ask_knowledge_agent"]


def test_failed_step_contributes_nothing_it_found_before_failing(perry):
    reply = two_step_turn_where_the_second_step_fails_twice(perry)

    assert "hiking.md" not in reply
    assert "waterfall" not in reply
    assert "no result" in reply
