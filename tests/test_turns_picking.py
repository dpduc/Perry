"""Picking a Retrieval Technique, seen through Turns: semantic picking by example
questions, cascade falling back to logical picking, the mode setting and the override.

The registry is replaced with two stand-in techniques whose example questions are far
apart under the fake (bag-of-words) embeddings, so which one is nearest is plain.
"""

import pytest
from fakes import FakeSummaryModel, call, fake_provider, last_message

from perry import models, summaries, techniques
from perry.techniques import Technique, hybrid

BUDGET = Technique(
    name="budget_search",
    label="Budget search",
    description="Search for money figures.",
    examples=(
        "How much is the marketing budget?",
        "What is the Q4 marketing budget?",
        "Which budget pays for marketing?",
    ),
    chain=hybrid,
)
TRAILS = Technique(
    name="trail_search",
    label="Trail search",
    description="Search hiking notes.",
    examples=(
        "Is the ridge path steep?",
        "Where does the waterfall trail start?",
        "How long is the ridge hike?",
    ),
    chain=hybrid,
)


SUMMARIES = Technique(
    name="summary_search",
    label="Summary search",
    description="Find the Documents about a topic from their Summaries.",
    examples=(
        "Which Document covers the marketing budget?",
        "Which file talks about hiking?",
        "Which Document mentions the ridge path?",
    ),
    chain=hybrid,
    summary_kind="document",
)


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(techniques, "TECHNIQUES", (BUDGET, TRAILS, SUMMARIES))
    monkeypatch.delenv("PERRY_PICKING_MODE", raising=False)


def ask(perry, question: str, technique_call: str, **send) -> str:
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    perry.add_document("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")
    perry.script(
        call("ask_knowledge_agent", question=question),  # Supervisor
        call(technique_call, question=question),  # Knowledge Agent searches
        last_message,  # Knowledge Agent reports the Chunks it found
        last_message,  # Supervisor passes the findings on
    )
    return perry.send(question, **send)


def found_with(reply: str) -> str:
    return reply.splitlines()[-1]


def test_semantic_picking_runs_the_technique_with_the_nearest_examples(
    perry, registry, monkeypatch
):
    monkeypatch.setenv("PERRY_PICKING_MODE", "semantic")

    reply = ask(perry, "What is the Q3 marketing budget?", "budget_search")

    assert "40,000" in reply
    assert found_with(reply) == "Found with: Budget search (picked semantically)"


def test_semantic_picking_is_recorded_in_the_usage_ledger_as_picking(
    perry, registry, monkeypatch
):
    from perry import usage

    monkeypatch.setenv("PERRY_PICKING_MODE", "semantic")

    ask(perry, "What is the Q3 marketing budget?", "budget_search")

    # The question's embedding (32 characters: 8 tokens), plus any example questions
    # not embedded yet in this process.
    assert usage.today("pick").tokens >= 8


# Under the fake embeddings this is a little nearer Trail search's examples than Budget
# search's (by less than the default margin): a close call.
CLOSE_CALL = "Any notes on marketing and the ridge?"


def test_cascade_picks_semantically_when_one_technique_is_clearly_nearest(perry, registry):
    reply = ask(perry, "What is the Q3 marketing budget?", "budget_search")

    assert found_with(reply) == "Found with: Budget search (picked semantically)"


def test_cascade_falls_back_to_logical_picking_on_a_close_call(perry, registry):
    reply = ask(perry, CLOSE_CALL, "budget_search")

    assert found_with(reply) == "Found with: Budget search (picked logically)"


def test_logical_mode_leaves_even_a_clear_question_to_the_model(perry, registry, monkeypatch):
    monkeypatch.setenv("PERRY_PICKING_MODE", "logical")

    reply = ask(perry, "What is the Q3 marketing budget?", "trail_search")

    assert found_with(reply) == "Found with: Trail search (picked logically)"


def test_semantic_mode_picks_even_on_a_close_call(perry, registry, monkeypatch):
    monkeypatch.setenv("PERRY_PICKING_MODE", "semantic")

    reply = ask(perry, CLOSE_CALL, "trail_search")

    assert found_with(reply) == "Found with: Trail search (picked semantically)"


SUMMARY_QUESTION = "Which Document covers the marketing budget?"


def test_a_summary_based_technique_is_picked_once_a_documents_summary_is_ready(
    perry, registry, monkeypatch
):
    monkeypatch.setenv("PERRY_PICKING_MODE", "semantic")
    perry.add_document("plan.md", "The marketing plan for Q4.")
    with models.use(fake_provider(perry.model, summary=FakeSummaryModel())):
        summaries.summarise_pending()  # plan.md's Summary is ready; the next two pend

    reply = ask(perry, SUMMARY_QUESTION, "summary_search")

    assert found_with(reply) == "Found with: Summary search (picked semantically)"


@pytest.mark.parametrize("mode", ["semantic", "logical"])
def test_summary_based_techniques_are_not_picked_while_no_summary_is_ready(
    perry, registry, monkeypatch, mode
):
    monkeypatch.setenv("PERRY_PICKING_MODE", mode)
    # The Documents `ask` uploads are pending: this provider has no summary model.

    reply = ask(perry, SUMMARY_QUESTION, "summary_search")

    assert "Summary search" not in reply


@pytest.mark.parametrize("mode", ["semantic", "logical", "cascade"])
def test_the_override_beats_every_picking_mode(perry, registry, monkeypatch, mode):
    monkeypatch.setenv("PERRY_PICKING_MODE", mode)

    reply = ask(
        perry, "What is the Q3 marketing budget?", "trail_search", technique="trail_search"
    )

    assert found_with(reply) == "Found with: Trail search (forced)"
