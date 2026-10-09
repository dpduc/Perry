"""Background Summaries and Document metadata, written by the local summary model
(a fake here), against a real Postgres with fake embeddings."""

import json
from datetime import date

import httpx
import pymupdf
import pytest
from fakes import (
    FakeSummaryModel,
    ScriptedChatModel,
    fake_embed,
    fake_embed_one,
    fake_provider,
)

from perry import models, summaries, techniques, usage
from perry.ingest import estimate_ingestion, ingest_file
from perry.search import hybrid_search


def summary_reply(summary: str, document_date: str | None = None, topic: str = "general") -> str:
    return json.dumps({"summary": summary, "document_date": document_date, "topic": topic})


@pytest.fixture
def summary_model(database):
    """The local summary model every model call in the test goes to, with fake embeddings."""
    model = FakeSummaryModel()
    with models.use(fake_provider(ScriptedChatModel(), summary=model)):
        yield model


@pytest.fixture
def upload(tmp_path):
    def write(name: str, text: str) -> None:
        (tmp_path / name).write_text(text, encoding="utf-8")
        ingest_file(tmp_path / name, embed=fake_embed)

    return write


def test_background_job_writes_a_summary_document_date_and_topic(summary_model, upload):
    summary_model.replies.append(
        summary_reply("Plan for Q3 marketing spend.", "2025-07-01", "Marketing budget")
    )
    upload("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    assert summaries.document("budget.md").status == summaries.PENDING

    summaries.summarise_pending()

    assert summaries.document("budget.md") == summaries.DocumentSummary(
        status=summaries.READY,
        summary="Plan for Q3 marketing spend.",
        document_date=date(2025, 7, 1),
        topic="Marketing budget",
    )
    # The local model read the Document itself.
    assert "40,000 dollars" in summary_model.calls[0][-1].text


def test_a_failing_summary_model_leaves_the_document_searchable_and_the_next_start_retries_it(
    summary_model, upload
):
    summary_model.replies.append(httpx.ConnectError("Ollama isn't running"))
    upload("budget.md", "The marketing budget for Q3 is 40,000 dollars.")

    summaries.summarise_pending()

    assert summaries.document("budget.md").status == summaries.FAILED
    assert hybrid_search("marketing budget", embed=fake_embed_one)[0].citation == "budget.md"

    summaries.retry_failed()  # what the next start does before summarising again
    summaries.summarise_pending()

    assert summaries.document("budget.md").status == summaries.READY


def test_reupload_replaces_the_summary_and_metadata(summary_model, upload):
    summary_model.replies += [
        summary_reply("Launch plan for March.", "2025-03-01", "Launch"),
        summary_reply("Revised launch plan for June.", "2025-06-01", "Launch revision"),
    ]
    upload("plan.md", "The launch is in March.")
    summaries.summarise_pending()

    upload("plan.md", "The launch moved to June.")
    assert summaries.document("plan.md") == summaries.DocumentSummary(status=summaries.PENDING)

    summaries.summarise_pending()
    assert summaries.document("plan.md") == summaries.DocumentSummary(
        status=summaries.READY,
        summary="Revised launch plan for June.",
        document_date=date(2025, 6, 1),
        topic="Launch revision",
    )


def test_local_summary_calls_are_in_the_usage_ledger_at_zero_cost(summary_model, upload, caplog):
    upload("budget.md", "The marketing budget for Q3 is 40,000 dollars.")

    summaries.summarise_pending()

    # The fake reports 100 tokens in and 20 out, as qwen3:8b, priced by prices.toml.
    assert usage.today(kind="summary") == usage.Usage(tokens=120, cost_usd=0.0)
    assert "'qwen3:8b'" not in caplog.text  # priced at zero, not just unpriced


def test_the_upload_estimate_counts_summarising_on_the_local_model_as_free(
    summary_model, tmp_path, prices
):
    prices({"fake-embedding": {"input": 1000.0}, "qwen3:8b": {"input": 0.0, "output": 0.0}})
    file = tmp_path / "budget.md"
    file.write_text("The marketing budget for Q3 is 40,000 dollars.", encoding="utf-8")

    estimate = estimate_ingestion([file])

    summary, summary_embedding = estimate.background
    assert (summary.kind, summary.model, summary.cost_usd) == ("summary", "qwen3:8b", 0)
    # Embedding the Summary is paid, and counts toward the Daily Cap.
    assert summary_embedding.kind == "embedding"
    assert estimate.cost_usd == pytest.approx(
        (12 + summary_embedding.input_tokens) * 1000 / 1_000_000
    )


def document_summaries(question: str):
    return techniques.get("document_summaries").chain.invoke(question)


def test_document_summaries_technique_returns_the_matching_documents_chunks_with_page_citations(
    summary_model, upload, tmp_path
):
    summary_model.replies += [
        summary_reply("Annual report with the Q3 marketing budget and spending plans."),
        summary_reply("Trail notes about a steep ridge path and a waterfall."),
    ]
    report = tmp_path / "report.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Annual report of the company.")
        pdf.new_page().insert_text((72, 72), "The marketing budget for Q3 is 40,000 dollars.")
        pdf.save(report)
    ingest_file(report, embed=fake_embed)
    upload("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")
    summaries.summarise_pending()

    hits = document_summaries("Which document has the Q3 marketing budget?")

    assert hits[0].citation == "report.pdf#page=2"
    assert "40,000" in hits[0].text
    # Only Chunks come back, never a Summary.
    chunks = {
        "Annual report of the company.",
        "The marketing budget for Q3 is 40,000 dollars.",
        "Trail notes: the ridge path is steep after the waterfall.",
    }
    assert {hit.text for hit in hits} <= chunks


def test_document_summaries_technique_skips_documents_whose_summaries_are_not_ready(
    summary_model, upload
):
    summary_model.replies += [
        summary_reply("The Q3 marketing budget."),
        httpx.ConnectError("Ollama isn't running"),
    ]
    upload("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    upload("draft.md", "Draft: the marketing budget for Q3 might be 35,000 dollars.")
    summaries.summarise_pending()  # budget.md is ready, draft.md failed
    upload("latest.md", "Latest: the marketing budget for Q3 is 45,000 dollars.")  # pending

    hits = document_summaries("What is the Q3 marketing budget?")

    assert {hit.citation for hit in hits} == {"budget.md"}


def test_document_summaries_is_not_offered_for_picking_while_summaries_are_pending(
    summary_model, upload
):
    upload("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    assert "document_summaries" not in {t.name for t in techniques.available()}

    summaries.summarise_pending()

    assert "document_summaries" in {t.name for t in techniques.available()}
