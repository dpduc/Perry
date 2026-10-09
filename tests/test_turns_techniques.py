"""Retrieval Techniques seen through Turns: which technique found the answer, the
"Found with" line, and the per-request override that forces one."""

import pytest
from fakes import call, last_message, with_usage


def ask_about_budget(perry, technique_call: str = "hybrid_search", **send) -> str:
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    perry.add_document("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")
    perry.script(
        call("ask_knowledge_agent", question="What is the Q3 marketing budget?"),  # Supervisor
        call(technique_call, question="Q3 marketing budget"),  # Knowledge Agent picks
        last_message,  # Knowledge Agent reports the passages it found
        last_message,  # Supervisor passes the findings on
    )
    return perry.send("How big is my Q3 marketing budget?", **send)


def test_question_about_documents_is_answered_with_hybrid_search_picked_logically(perry):
    reply = ask_about_budget(perry)

    assert reply.startswith("The marketing budget for Q3 is 40,000 dollars.")
    *_, citations, found_with = reply.splitlines()
    assert citations.startswith("Citations: [budget.md]")
    assert found_with == "Found with: Hybrid search (picked logically)"
    assert perry.history()[-1] == {"role": "assistant", "content": reply}


def test_a_setting_turns_the_found_with_line_off(perry, monkeypatch):
    monkeypatch.setenv("PERRY_SHOW_FOUND_WITH", "false")

    reply = ask_about_budget(perry)

    assert reply.splitlines()[-1].startswith("Citations: [budget.md]")
    assert "Found with" not in reply


def test_a_forced_technique_is_used_and_reported_as_forced(perry):
    reply = ask_about_budget(perry, technique="hybrid_search")

    *_, citations, found_with = reply.splitlines()
    assert citations.startswith("Citations: [budget.md]")
    assert found_with == "Found with: Hybrid search (forced)"


def test_a_forced_technique_leaves_the_supervisor_free_to_not_ask_for_documents(perry):
    perry.script("Hi! I'm Perry. How can I help?")

    reply = perry.send("Hello there", technique="hybrid_search")

    assert reply == "Hi! I'm Perry. How can I help?"
    assert perry.consulted() == []


def test_the_forced_technique_applies_only_to_the_turn_it_came_with(perry):
    ask_about_budget(perry, technique="hybrid_search")
    perry.script(
        call("ask_knowledge_agent", question="What is the Q3 marketing budget?"),
        call("hybrid_search", question="Q3 marketing budget"),
        last_message,
        last_message,
    )

    reply = perry.send("And the Q3 marketing budget again?")

    assert reply.splitlines()[-1] == "Found with: Hybrid search (picked logically)"


def ask_with_rewrites(perry, technique: str, *rewrite_steps) -> str:
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    perry.add_document("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")
    perry.script(
        call("ask_knowledge_agent", question="What is the Q3 marketing budget?"),  # Supervisor
        call(technique, question="Q3 marketing budget"),  # Knowledge Agent
        *rewrite_steps,  # the technique rewrites the question
        last_message,  # Knowledge Agent reports the passages it found
        last_message,  # Supervisor passes the findings on
    )
    return perry.send("How big is my Q3 marketing budget?", technique=technique)


@pytest.mark.parametrize(
    ("technique", "label"), [("multi_query", "Multi-query"), ("rag_fusion", "RAG fusion")]
)
def test_a_forced_query_rewriting_technique_answers_with_citations(perry, technique, label):
    reply = ask_with_rewrites(
        perry, technique, "Q3 marketing budget\nmarketing spend in the third quarter"
    )

    *_, citations, found_with = reply.splitlines()
    assert citations.startswith("Citations: [budget.md]")
    assert found_with == f"Found with: {label} (forced)"


@pytest.mark.parametrize("technique", ["multi_query", "rag_fusion"])
def test_rewrite_calls_are_recorded_as_rewrites_and_count_toward_the_token_ceiling(
    perry, monkeypatch, technique
):
    from perry import usage

    monkeypatch.setenv("PERRY_TURN_TOKEN_CEILING", "1000")
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    perry.script(
        with_usage(call("ask_knowledge_agent", question="Q3 budget?"), 300),  # Supervisor
        with_usage(call(technique, question="Q3 budget"), 300),  # Knowledge Agent
        with_usage("Q3 marketing budget\nmarketing spend", 400, 50),  # the rewrite
        # The rewrite took the Turn over its ceiling: no further call happens.
    )

    reply = perry.send("How big is my Q3 marketing budget?")

    assert reply.startswith("I stopped because this Turn reached its limit of 1,000 tokens.")
    assert usage.today("rewrite").tokens == 450
