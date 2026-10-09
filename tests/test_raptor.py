"""RAPTOR: per-Document trees and a cross-Document level, built in the background by the
local summary model (a fake here), against a real Postgres with fake embeddings."""

import httpx
import pymupdf
import pytest
from fakes import FakeSummaryModel, ScriptedChatModel, fake_embed, fake_provider

from perry import models, raptor, summaries, techniques
from perry.ingest import estimate_ingestion, ingest_file


@pytest.fixture
def summary_model(database):
    model = FakeSummaryModel()
    with models.use(fake_provider(ScriptedChatModel(), summary=model)):
        yield model


@pytest.fixture
def upload(tmp_path):
    def write(name: str, text: str) -> None:
        (tmp_path / name).write_text(text, encoding="utf-8")
        ingest_file(tmp_path / name, embed=fake_embed)

    return write


PAGES = [
    "The marketing budget for Q3 is 40,000 dollars.",
    "Marketing spend on social ads is 12,000 dollars in Q3.",
    "The Q3 marketing budget covers print adverts too.",
    "Marketing events in Q3 cost 8,000 dollars of the budget.",
    "The office kitchen gets a new coffee machine in May.",
    "Kitchen cleaning happens every Friday afternoon.",
    "The coffee machine in the kitchen needs descaling monthly.",
    "Kitchen supplies are ordered by the office manager.",
]


@pytest.fixture
def report(tmp_path):
    """A PDF of eight one-Chunk pages on two subjects: too many Chunks for one cluster."""
    path = tmp_path / "report.pdf"
    with pymupdf.open() as pdf:
        for text in PAGES:
            pdf.new_page().insert_text((72, 72), text)
        pdf.save(path)
    ingest_file(path, embed=fake_embed)
    return path


def raptor_search(question: str):
    return techniques.get("raptor").chain.invoke(question)


def test_background_job_builds_a_tree_per_document_and_a_cross_document_level(
    summary_model, upload, report
):
    upload("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")
    assert raptor.status("report.pdf") == raptor.PENDING

    summaries.summarise_pending()

    assert raptor.status("report.pdf") == raptor.READY
    tree = raptor.tree("report.pdf")
    # Eight Chunks are clustered under several nodes, summarised again up to one root.
    assert len([node for node in tree if node.level == 1]) > 1
    *_, root = tree
    assert [node for node in tree if node.level == root.level] == [root]
    assert root.citations == tuple(f"report.pdf#page={page}" for page in range(1, 9))
    # The cross-Document level sits over both Documents' trees.
    (top,) = raptor.cross_document()
    assert set(top.citations) == {*root.citations, "hiking.md"}
    # Every node was summarised on the local model, after each Document's Summary.
    assert len(summary_model.calls) == 2 + len(tree) + len(raptor.tree("hiking.md")) + 1


def test_a_match_at_any_level_returns_leaf_chunks_with_page_citations_never_node_text(
    summary_model, upload, report
):
    upload("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")
    summaries.summarise_pending()

    hits = raptor_search("What does the Q3 marketing budget cover?")

    assert hits[0].citation.startswith("report.pdf#page=")
    assert "marketing" in hits[0].text.lower()
    chunks = {*PAGES, "Trail notes: the ridge path is steep after the waterfall."}
    assert {hit.text for hit in hits} <= chunks  # never a node's Summary


def test_reupload_replaces_the_documents_nodes_and_rebuilds_the_cross_document_level(
    summary_model, upload
):
    upload("plan.md", "The launch is in March.")
    upload("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")
    summaries.summarise_pending()

    upload("plan.md", "The launch moved to June.")
    assert raptor.status("plan.md") == raptor.PENDING
    assert raptor.tree("plan.md") == []
    summaries.summarise_pending()

    (root,) = raptor.tree("plan.md")
    assert "June" in root.text and "March" not in root.text
    (top,) = raptor.cross_document()
    assert "June" in top.text and "March" not in top.text
    assert set(top.citations) == {"plan.md", "hiking.md"}


def test_documents_whose_tree_is_not_ready_are_skipped(summary_model, upload):
    upload("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    upload("draft.md", "Draft: the marketing budget for Q3 might be 35,000 dollars.")
    # Both Summaries succeed; then draft.md's tree fails (budget.md's tree is built first).
    summary_model.replies += [
        "Summary of the budget.",
        "Summary of the draft.",
        "Tree node: the marketing budget for Q3 is 40,000 dollars.",
        httpx.ConnectError("Ollama isn't running"),
    ]
    summaries.summarise_pending()
    assert raptor.status("draft.md") == raptor.FAILED
    upload("latest.md", "Latest: the marketing budget for Q3 is 45,000 dollars.")  # pending

    hits = raptor_search("What is the Q3 marketing budget?")

    assert {hit.citation for hit in hits} == {"budget.md"}

    summaries.retry_failed()  # what the next start does
    summaries.summarise_pending()
    assert raptor.status("draft.md") == raptor.READY


def test_raptor_is_not_offered_for_picking_while_trees_are_pending(summary_model, upload):
    upload("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    assert "raptor" not in {t.name for t in techniques.available()}

    summaries.summarise_pending()

    assert "raptor" in {t.name for t in techniques.available()}


def test_the_upload_estimate_counts_building_the_tree_on_the_local_model_as_free(
    summary_model, tmp_path, prices
):
    prices({"fake-embedding": {"input": 1000.0}, "qwen3:8b": {"input": 0.0, "output": 0.0}})
    file = tmp_path / "budget.md"
    file.write_text("The marketing budget for Q3 is 40,000 dollars.", encoding="utf-8")

    estimate = estimate_ingestion([file])

    *_, tree, tree_embedding = estimate.background
    assert (tree.kind, tree.model, tree.cost_usd) == ("raptor", "qwen3:8b", 0)
    assert tree.input_tokens > 12  # it reads the whole Document
    # Embedding the nodes' Summaries is paid, and counts toward the Daily Cap.
    assert tree_embedding.kind == "embedding"
    assert tree_embedding.cost_usd > 0
