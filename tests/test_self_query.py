"""Self-query: the model turns the question into a filter over Document metadata plus a
search query, and only Chunks of matching Documents come back."""

import json
from datetime import datetime, timedelta

from fakes import (
    FakeSummaryModel,
    ScriptedChatModel,
    call,
    fake_embed,
    fake_provider,
    last_message,
)

from perry import models, summaries, techniques
from perry.ingest import ingest_file


def ingest(tmp_path, documents: dict[str, str]) -> None:
    for name, text in documents.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
        ingest_file(tmp_path / name, embed=fake_embed)


def self_query(question: str, reply: dict | str, **provider) -> list:
    """Run Self-query for `question`, the model replying with `reply` (a filter as JSON)."""
    reply = reply if isinstance(reply, str) else json.dumps(reply)
    with models.use(fake_provider(ScriptedChatModel(steps=[reply]), **provider)):
        return techniques.get("self_query").chain.invoke(question)


BUDGETS = {
    "budget-2024.md": "Budget notes: the marketing budget is 30,000 dollars.",
    "budget-2025.md": "Budget notes: the marketing budget is 40,000 dollars.",
    "hiking.md": "Trail notes: the ridge path is steep after the waterfall.",
}


def test_a_question_naming_a_document_searches_only_that_document(database, tmp_path):
    ingest(tmp_path, BUDGETS)

    hits = self_query(
        "In the 2025 budget notes, what is the marketing budget?",
        {"query": "marketing budget", "name": "2025 budget"},
    )

    assert [hit.citation for hit in hits] == ["budget-2025.md"]
    assert "40,000" in hits[0].text


def write_pdf(path, pages: list[str]) -> None:
    import pymupdf

    with pymupdf.open() as pdf:
        for text in pages:
            pdf.new_page().insert_text((72, 72), text)
        pdf.save(path)


def test_a_question_naming_a_file_type_and_page_searches_only_those_pages(
    database, tmp_path
):
    ingest(tmp_path, BUDGETS)
    write_pdf(
        tmp_path / "plan.pdf",
        [
            "Plan overview: the marketing budget is reviewed yearly.",
            "Plan details: the marketing budget is 50,000 dollars.",
        ],
    )
    ingest_file(tmp_path / "plan.pdf", embed=fake_embed)

    hits = self_query(
        "On page 2 of my PDFs, what is the marketing budget?",
        {"query": "marketing budget", "file_type": "pdf", "page_from": 2, "page_to": 2},
    )

    assert [hit.citation for hit in hits] == ["plan.pdf#page=2"]
    assert "50,000" in hits[0].text


def summarised(documents: dict[str, tuple[str | None, str]]):
    """A summary model that gives each Document (by name) its Document date and topic."""

    def reply_for(name: str) -> str:
        dated, topic = documents[name]
        return json.dumps(
            {"summary": f"Summary of {name}", "document_date": dated, "topic": topic}
        )

    # The summarising job takes Documents in upload order.
    return FakeSummaryModel(replies=[reply_for(name) for name in documents])


def test_filters_on_document_date_and_topic_use_the_summarising_jobs_metadata(
    database, tmp_path
):

    ingest(tmp_path, BUDGETS)
    summary = summarised(
        {
            "budget-2024.md": ("2024-11-02", "marketing budget"),
            "budget-2025.md": ("2025-03-15", "marketing budget"),
            "hiking.md": ("2025-06-01", "hiking trails"),
        }
    )
    with models.use(fake_provider(ScriptedChatModel(), summary=summary)):
        summaries.summarise_pending()

    dated = self_query(
        "What did the 2025 marketing documents say the budget is?",
        {
            "query": "budget",
            "dated_from": "2025",
            "dated_to": "2025",
            "topic": "marketing",
        },
    )
    on_topic = self_query(
        "What do my hiking notes say?", {"query": "path", "topic": "hiking"}
    )

    assert [hit.citation for hit in dated] == ["budget-2025.md"]
    assert [hit.citation for hit in on_topic] == ["hiking.md"]


def test_a_filter_on_the_upload_date_keeps_only_documents_uploaded_then(
    database, tmp_path
):

    ingest(tmp_path, BUDGETS)
    today = datetime.now().astimezone().date()
    yesterday = today - timedelta(days=1)

    uploaded_today = self_query(
        "What budget did I upload today?",
        {"query": "marketing budget", "uploaded_from": today.isoformat()},
    )
    uploaded_before = self_query(
        "What budget did I upload before today?",
        {"query": "marketing budget", "uploaded_to": yesterday.isoformat()},
    )

    assert {hit.citation for hit in uploaded_today} >= {
        "budget-2024.md",
        "budget-2025.md",
    }
    assert uploaded_before == []


def test_a_question_with_no_usable_filter_returns_the_same_chunks_as_hybrid_search(
    database, tmp_path
):
    ingest(tmp_path, BUDGETS)
    question = "What is the marketing budget?"
    with models.use(fake_provider(ScriptedChatModel())):
        plain = techniques.get("hybrid_search").chain.invoke(question)

    no_filter = self_query(question, {"query": "budget", "name": None, "topic": None})
    not_json = self_query(question, "I can't tell which Documents you mean.")

    assert no_filter == plain
    assert not_json == plain


def test_forced_self_query_answers_with_citations_and_its_found_with_line(perry):

    for name, text in BUDGETS.items():
        perry.add_document(name, text)
    question = "In the 2025 budget notes, what is the marketing budget?"
    perry.script(
        call("ask_knowledge_agent", question=question),  # Supervisor
        call("self_query", question=question),  # Knowledge Agent
        json.dumps({"query": "marketing budget", "name": "2025 budget"}),  # the filter
        last_message,  # Knowledge Agent reports the Chunks it found
        last_message,  # Supervisor passes the findings on
    )

    reply = perry.send(question, technique="self_query")

    assert reply.startswith("Budget notes: the marketing budget is 40,000 dollars.")
    assert "30,000" not in reply
    *_, citations, found_with = reply.splitlines()
    assert citations == "Citations: [budget-2025.md]"
    assert found_with == "Found with: Self-query (forced)"


def test_a_filter_no_document_matches_is_reported_and_another_technique_can_answer(perry):
    for name, text in BUDGETS.items():
        perry.add_document(name, text)
    question = "In the 2019 budget notes, what is the marketing budget?"

    def told_no_document_matches(messages):
        # Self-query keeps its strict filter, and tells the model which one matched nothing.
        assert "No Document matches the filter" in messages[-1].text
        assert "name '2019 budget'" in messages[-1].text
        return call("hybrid_search", question="marketing budget")

    perry.script(
        call("ask_knowledge_agent", question=question),  # Supervisor
        call("self_query", question=question),  # Knowledge Agent
        json.dumps({"query": "marketing budget", "name": "2019 budget"}),  # the filter
        told_no_document_matches,  # Knowledge Agent tries another technique
        last_message,  # Knowledge Agent reports the Chunks it found
        last_message,  # Supervisor passes the findings on
    )

    reply = perry.send(question)

    assert "marketing budget is" in reply
    assert reply.splitlines()[-1] == "Found with: Hybrid search (picked logically)"
