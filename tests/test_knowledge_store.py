"""Ingestion and hybrid search against a real Postgres, with fake embeddings."""

from conftest import fake_embed, fake_embed_one

from perry.ingest import ingest_file
from perry.search import hybrid_search


def search(query: str):
    return hybrid_search(query, embed=fake_embed_one)


def test_search_finds_the_right_document_with_its_citation(database, tmp_path):
    (tmp_path / "budget.md").write_text(
        "The marketing budget for Q3 is 40,000 dollars.", encoding="utf-8"
    )
    (tmp_path / "hiking.md").write_text(
        "Trail notes: the ridge path is steep after the waterfall.", encoding="utf-8"
    )
    for name in ("budget.md", "hiking.md"):
        ingest_file(tmp_path / name, embed=fake_embed)

    hits = search("marketing budget")

    assert hits[0].citation == "budget.md"
    assert "40,000" in hits[0].text


def test_unchanged_document_is_skipped(database, tmp_path):
    file = tmp_path / "a.md"
    file.write_text("alpha", encoding="utf-8")

    assert ingest_file(file, embed=fake_embed).status == "ingested"
    assert ingest_file(file, embed=fake_embed).status == "unchanged"


def test_reupload_with_same_name_replaces_the_document(database, tmp_path):
    file = tmp_path / "plan.md"
    file.write_text("The launch is in March.", encoding="utf-8")
    ingest_file(file, embed=fake_embed)

    file.write_text("The launch moved to June.", encoding="utf-8")
    ingest_file(file, embed=fake_embed)

    texts = [hit.text for hit in search("launch")]
    assert texts == ["The launch moved to June."]
