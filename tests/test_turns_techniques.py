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


@pytest.mark.parametrize(
    ("technique", "label", "model_reply"),
    [
        ("step_back", "Step-back", "What drives marketing budgets?"),
        ("hyde", "HyDE", "The marketing budget for Q3 is 10,000 dollars."),
    ],
)
def test_a_forced_query_translation_technique_answers_with_citations(
    perry, technique, label, model_reply
):
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    perry.add_document("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")
    perry.script(
        call("ask_knowledge_agent", question="What is the Q3 marketing budget?"),  # Supervisor
        call(technique, question="Q3 marketing budget"),  # Knowledge Agent
        model_reply,  # the technique's own model call (broader question / hypothetical)
        last_message,  # Knowledge Agent reports the passages it found
        last_message,  # Supervisor passes the findings on
    )

    reply = perry.send("How big is my Q3 marketing budget?", technique=technique)

    assert reply.startswith("The marketing budget for Q3 is 40,000 dollars.")
    assert model_reply not in reply
    *_, citations, found_with = reply.splitlines()
    assert citations.startswith("Citations: [budget.md]")
    assert found_with == f"Found with: {label} (forced)"


def compare_budget_and_hike(perry, **send) -> str:
    """A Turn whose question chains two facts, from two Documents, answered with
    Question Decomposition."""
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    perry.add_document("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")
    question = "What is the Q3 marketing budget, and how steep is the ridge path?"
    perry.script(
        call("ask_knowledge_agent", question=question),  # Supervisor
        call("question_decomposition", question=question),  # Knowledge Agent
        "What is the Q3 marketing budget?\nHow steep is the ridge path?",  # sub-questions
        last_message,  # Knowledge Agent reports the passages it found
        last_message,  # Supervisor passes the findings on
    )
    return perry.send("How big is my Q3 budget, and is the ridge path steep?", **send)


def test_forced_question_decomposition_answers_with_every_sub_questions_citations(perry):
    reply = compare_budget_and_hike(perry, technique="question_decomposition")

    assert "40,000 dollars" in reply and "ridge path is steep" in reply
    *_, citations, found_with = reply.splitlines()
    assert citations.startswith("Citations: ")
    assert "[budget.md]" in citations and "[hiking.md]" in citations
    assert found_with == "Found with: Question Decomposition (forced)"


def test_a_chained_turn_is_still_split_into_agent_steps_by_the_supervisor(perry, workspace):
    """Task Decomposition, not Question Decomposition: even with Question Decomposition
    forced, the Supervisor asks the Knowledge Agent, then the Task Agent."""
    from perry.supervisor import Approve, Confirmation

    perry.add_document("q3-notes.md", "Q3 notes: revenue grew 12%. Hiring freeze until January.")

    def email_the_summary(messages):
        assert "revenue grew 12%" in messages[-1].text
        return call("ask_task_agent", request="Email sam@example.com: revenue grew 12%.")

    perry.script(
        call("ask_knowledge_agent", question="What do the Q3 notes say?"),  # Supervisor
        call("question_decomposition", question="What do the Q3 notes say?"),  # Knowledge Agent
        "What do the Q3 notes say?",  # one fact: the question stays whole
        last_message,  # Knowledge Agent reports
        email_the_summary,  # Supervisor chains to the Task Agent
        call("send_email", to="sam@example.com", subject="Q3", body="Revenue grew 12%."),
    )

    pending = perry.send(
        "Find my Q3 notes and email a summary to Sam", technique="question_decomposition"
    )

    assert isinstance(pending, Confirmation)
    perry.script(last_message, "Done: I emailed Sam the Q3 summary.")
    reply = perry.resume(Approve())

    assert perry.consulted() == ["ask_knowledge_agent", "ask_task_agent"]
    assert [e.subject for e in workspace.sent] == ["Q3"]
    assert reply.splitlines()[-1] == "Found with: Question Decomposition (forced)"


def test_sub_questions_count_toward_the_token_ceiling_and_the_turn_reports_its_findings(
    perry, monkeypatch
):
    monkeypatch.setenv("PERRY_TURN_TOKEN_CEILING", "1000")
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    perry.add_document("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")
    question = "What is the Q3 marketing budget, and how steep is the ridge path?"
    perry.script(
        with_usage(call("ask_knowledge_agent", question=question), 400),  # Supervisor
        with_usage(call("question_decomposition", question=question), 100),  # Knowledge Agent
        with_usage("What is the Q3 marketing budget?\nHow steep is the ridge path?", 100),
        # 990 chat tokens are under the ceiling, but each sub-question's search embeds it
        # too (15 tokens here), so the Supervisor's next call would go over and never happens.
        with_usage(last_message, 390),  # Knowledge Agent reports
    )

    reply = perry.send("How big is my Q3 budget, and is the ridge path steep?")

    assert reply.startswith("I stopped because this Turn reached its limit of 1,000 tokens.")
    assert "40,000 dollars" in reply and "ridge path is steep" in reply
