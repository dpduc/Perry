"""Turns end to end: what the user sees, with a scripted fake chat model and fake embeddings."""

import pytest
from fakes import call, last_message


def test_small_talk_is_answered_without_consulting_any_agent(perry):
    perry.script("Hi! I'm Perry. How can I help?")

    reply = perry.send("Hello there")

    assert reply == "Hi! I'm Perry. How can I help?"
    assert perry.consulted() == []


def test_question_about_documents_is_answered_by_the_knowledge_agent_with_citations(perry):
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    perry.add_document("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")
    perry.script(
        call("ask_knowledge_agent", question="What is the Q3 marketing budget?"),  # Supervisor
        call("hybrid_search", question="Q3 marketing budget"),  # Knowledge Agent
        last_message,  # Knowledge Agent reports the Chunks it found
        last_message,  # Supervisor passes the findings on
    )

    reply = perry.send("How big is my Q3 marketing budget?")

    assert perry.consulted() == ["ask_knowledge_agent"]
    assert reply.startswith("The marketing budget for Q3 is 40,000 dollars.")
    assert "\nCitations: [budget.md]" in reply  # at the end, then the "Found with" line


def test_citations_are_moved_out_of_the_answer_to_its_end(perry):
    perry.script(
        call("ask_knowledge_agent", question="Q3 budget and headcount?"),  # Supervisor
        # Knowledge Agent's findings
        "Q3 budget is 40,000 dollars [budget.md]; headcount is 12 [report.pdf#page=2].",
        "Your Q3 budget is 40,000 dollars [budget.md], and headcount is 12 [report.pdf#page=2].",
    )

    reply = perry.send("What are my Q3 budget and headcount?")

    assert reply == (
        "Your Q3 budget is 40,000 dollars, and headcount is 12.\n\n"
        "Citations: [budget.md] [report.pdf#page=2]"
    )
    assert perry.history()[-1] == {"role": "assistant", "content": reply}


def test_citations_the_supervisors_answer_drops_are_put_back_at_the_end(perry):
    perry.script(
        call("ask_knowledge_agent", question="Q3 budget and headcount?"),
        "Q3 budget is 40,000 dollars [budget.md]; headcount is 12 [report.pdf#page=2].",
        "Your Q3 budget is 40,000 dollars [budget.md] and headcount is 12.",  # one dropped
    )

    reply = perry.send("What are my Q3 budget and headcount?")

    assert reply == (
        "Your Q3 budget is 40,000 dollars and headcount is 12.\n\n"
        "Citations: [budget.md] [report.pdf#page=2]"
    )


def test_brackets_that_are_not_citations_stay_where_they_are(perry):
    perry.script(
        call("ask_knowledge_agent", question="Q3 budget?"),
        "Q3 budget is 40,000 dollars [budget.md].",
        "Your Q3 budget is 40,000 dollars [budget.md] (see [the table](x.md)).",
    )

    assert perry.send("What is my Q3 budget?") == (
        "Your Q3 budget is 40,000 dollars (see [the table](x.md)).\n\nCitations: [budget.md]"
    )


def test_turn_that_exceeds_the_step_cap_stops_with_the_step_cap_message(perry):
    from perry.supervisor import MAX_STEPS_PER_TURN

    # The Supervisor keeps asking the Knowledge Agent and never answers.
    for _ in range(MAX_STEPS_PER_TURN + 1):
        perry.script(
            call("ask_knowledge_agent", question="Anything about llamas?"),
            "Nothing about llamas.",
        )

    reply = perry.send("Tell me everything about llamas")

    assert reply.startswith(f"I stopped after {MAX_STEPS_PER_TURN} steps without finishing.")
    assert perry.history() == []  # unfinished, so a retry won't show the question twice


def test_turn_that_crashes_midway_leaves_no_dangling_message_and_retry_shows_it_once(perry):
    perry.script("Hi! How can I help?")
    perry.send("Hello")
    perry.script(
        call("ask_knowledge_agent", question="What is the Q3 marketing budget?"),
        RuntimeError("Knowledge Agent crashed"),
    )

    with pytest.raises(RuntimeError):
        perry.send("How big is my Q3 marketing budget?")

    assert perry.history() == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
    ]

    perry.script("I couldn't find a budget in your Documents.")
    perry.send("How big is my Q3 marketing budget?")

    assert perry.history() == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": "How big is my Q3 marketing budget?"},
        {"role": "assistant", "content": "I couldn't find a budget in your Documents."},
    ]


def test_first_turn_of_a_thread_that_crashes_leaves_the_thread_empty(perry):
    perry.script(RuntimeError("model unavailable"))

    with pytest.raises(RuntimeError):
        perry.send("Hello")

    assert perry.history() == []
