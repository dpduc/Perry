"""Deleting, replacing (under a new name) and renaming a Document: afterwards no
Retrieval Technique returns the old Chunks or text derived from them, without a
restart; a failure partway leaves Postgres and the ColBERT index as they were."""

import pytest
from fakes import (
    FakeColbert,
    FakeSummaryModel,
    ScriptedChatModel,
    fake_embed,
    fake_provider,
)

from perry import ingest, models, raptor, summaries, techniques
from perry.ingest import ingest_file

OLD = "The Kestrel launch is in March."
NEW = "The Kestrel launch moved to June."
OTHERS = {
    "hiking.md": "Trail notes: the ridge path is steep after the waterfall.",
    "garden.md": "Garden log: the tomatoes went in along the south fence.",
}


@pytest.fixture
def library(database, tmp_path, monkeypatch):
    """report.md and two other Documents, summarised, with RAPTOR trees and the
    cross-Document level built, and indexed for ColBERT (fake models throughout)."""
    monkeypatch.setenv("PERRY_APP_DATA_DIR", str(tmp_path / "app-data"))
    folder = tmp_path / "documents"
    folder.mkdir()

    def write(name: str, text: str):
        path = folder / name
        path.write_text(text, encoding="utf-8")
        return path

    provider = fake_provider(ScriptedChatModel(), summary=FakeSummaryModel(), colbert=FakeColbert())
    with models.use(provider):
        for name, text in {"report.md": OLD, **OTHERS}.items():
            ingest_file(write(name, text), embed=fake_embed)
        summaries.summarise_pending()
        yield write


def found(question: str = "When is the Kestrel launch?") -> dict[str, set[str]]:
    """What every Retrieval Technique that needs no chat model finds, by technique."""
    names = ("hybrid_search", "colbert", "document_summaries", "raptor")
    return {
        name: {hit.text for hit in techniques.get(name).chain.invoke(question)}
        for name in names
    }


def derived_text() -> str:
    """Every Summary's text, the cross-Document RAPTOR level's included."""
    nodes = [node.text for node in raptor.cross_document()]
    for name in ["report.md", "report-v2.md", *OTHERS]:
        nodes += [node.text for node in raptor.tree(name)]
        if (summary := summaries.document(name)) and summary.summary:
            nodes.append(summary.summary)
    return "\n".join(nodes)


def test_a_deleted_document_is_found_by_no_technique_at_once(library):
    assert all(OLD in texts for texts in found().values())  # the setup

    assert ingest.delete_document("report.md") == 1  # its one Chunk

    assert not any(OLD in texts for texts in found().values())
    assert "Kestrel" not in derived_text()


def test_the_cross_document_level_is_rebuilt_from_the_documents_left(library):
    ingest.delete_document("report.md")

    summaries.summarise_pending()  # what the background job does next

    (top,) = raptor.cross_document()
    assert set(top.citations) == set(OTHERS)
    assert "Kestrel" not in top.text


def test_a_failed_index_write_leaves_the_document_in_place(library, monkeypatch):
    from fast_plaid.search import FastPlaid

    def disk_full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as patch, pytest.raises(OSError):
        patch.setattr(FastPlaid, "create", disk_full)
        ingest.delete_document("report.md")

    assert all(OLD in texts for texts in found().values())


def test_an_unknown_name_deletes_nothing(library):
    with pytest.raises(ingest.DocumentNotFound):
        ingest.delete_document("no-such-file.md")

    assert all(OLD in texts for texts in found().values())


def test_a_document_deleted_while_the_cross_document_level_is_built_leaves_no_trace(
    library, monkeypatch
):
    build = raptor._build

    def delete_meanwhile(level0):
        nodes = build(level0)
        if len(level0) > 1 and summaries.document("report.md"):  # the cross level
            ingest.delete_document("report.md")
        return nodes

    # A changed Document, so the background job builds the cross-Document level again.
    ingest_file(library("hiking.md", "Trail notes: the ridge path is muddy."), embed=fake_embed)
    monkeypatch.setattr(raptor, "_build", delete_meanwhile)
    summaries.summarise_pending()

    assert summaries.document("report.md") is None  # the setup: deleted mid-build
    assert "Kestrel" not in derived_text()
    (top,) = raptor.cross_document()  # built again from the Documents left
    assert set(top.citations) == set(OTHERS)


def test_a_newer_version_under_a_new_name_replaces_the_old_one(library):
    newer = library("report-v2.md", NEW)

    ingest_file(newer, embed=fake_embed, replaces="report.md")

    results = found()
    assert all(NEW in texts for texts in (results["hybrid_search"], results["colbert"]))
    assert not any(OLD in texts for texts in results.values())
    assert "Kestrel launch is in March" not in derived_text()


def test_a_failed_replacement_keeps_the_old_version(library):
    broken = library("report-v2.md", NEW + "\x00")  # Postgres refuses a NUL byte

    with pytest.raises(Exception, match="NUL"):
        ingest_file(broken, embed=fake_embed, replaces="report.md")

    assert all(OLD in texts for texts in found().values())


def test_a_renamed_document_is_cited_by_its_new_name(library):
    ingest.rename_document("report.md", "kestrel-plan.md")

    hits = techniques.get("hybrid_search").chain.invoke("When is the Kestrel launch?")
    assert hits[0].citation == "kestrel-plan.md"
    assert summaries.document("kestrel-plan.md") is not None  # its Summary went with it


def test_a_rename_onto_an_existing_name_is_refused(library):
    with pytest.raises(ingest.NameTaken):
        ingest.rename_document("report.md", "hiking.md")

    hits = techniques.get("hybrid_search").chain.invoke("When is the Kestrel launch?")
    assert hits[0].citation == "report.md"
