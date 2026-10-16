"""ColBERT: a late-interaction Retrieval Technique with its own on-disk index (ADR-0002), kept in
step with Postgres by ingestion. Uses the real index with the fake ColBERT encoder."""

import pytest
from fakes import (
    FakeColbert,
    ScriptedChatModel,
    call,
    fake_embed,
    fake_embed_one,
    fake_provider,
    last_message,
)

from perry import models, techniques
from perry.ingest import ingest_file
from perry.search import hybrid_search


@pytest.fixture
def documents(database, tmp_path, monkeypatch):
    """Write and ingest Documents with the fake ColBERT encoder, the index in a temp folder."""
    monkeypatch.setenv("PERRY_APP_DATA_DIR", str(tmp_path / "app-data"))
    folder = tmp_path / "documents"
    folder.mkdir()

    def add(name: str, text: str):
        path = folder / name
        path.write_text(text, encoding="utf-8")
        return ingest_file(path, embed=fake_embed)

    with models.use(fake_provider(ScriptedChatModel(), colbert=FakeColbert())):
        yield add


def colbert(question: str):
    return techniques.get("colbert").chain.invoke(question)


def test_colbert_returns_chunks_with_page_citations(documents):
    documents("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    documents("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")

    hits = colbert("Kestrel budget for Q3?")

    assert hits[0].citation == "budget.md"
    assert "40,000" in hits[0].text


def test_reupload_replaces_the_documents_entries_in_the_index(documents):
    documents("plan.md", "The Kestrel launch is in March.")
    documents("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")

    documents("plan.md", "The Kestrel launch moved to June.")

    texts = [hit.text for hit in colbert("Kestrel launch")]
    assert texts[0] == "The Kestrel launch moved to June."
    assert "The Kestrel launch is in March." not in texts


def test_colbert_encoding_is_estimated_and_recorded_in_the_usage_ledger(
    documents, tmp_path, prices
):
    from perry import ingest, usage

    prices({"fake-embedding": {"input": 1000.0}})  # fake-colbert unpriced: free, as it's local
    path = tmp_path / "budget.md"
    path.write_text("The marketing budget for Q3 is 40,000 dollars.", encoding="utf-8")

    estimate = ingest.estimate_ingestion([path])
    ingest.ingest_files([path])

    assert [(c.kind, c.model, c.tokens) for c in estimate.calls] == [
        ("embedding", "fake-embedding", 12),
        ("colbert", "fake-colbert", 12),
    ]
    assert estimate.cost_usd == pytest.approx(0.012)
    # ColBERT encodes the same text locally: shown once, not twice.
    assert estimate.describe() == "1 file: about 12 tokens, $0.0120."
    assert usage.today("colbert").tokens == 12
    assert usage.today().cost_usd == pytest.approx(estimate.cost_usd)


def hybrid(question: str):
    return hybrid_search(question, embed=fake_embed_one)


def test_a_failed_postgres_write_leaves_the_document_as_it_was_in_both(documents):
    documents("plan.md", "The Kestrel launch is in March.")
    documents("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")

    # Postgres refuses text containing a NUL byte, after the new index has been built.
    with pytest.raises(Exception, match="NUL"):
        documents("plan.md", "The Kestrel launch moved to June.\x00")

    assert colbert("Kestrel launch")[0].text == "The Kestrel launch is in March."
    assert hybrid("Kestrel launch")[0].text == "The Kestrel launch is in March."


def test_a_failed_index_write_leaves_the_document_as_it_was_in_both(documents, monkeypatch):
    from fast_plaid.search import FastPlaid

    documents("plan.md", "The Kestrel launch is in March.")
    documents("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")

    def disk_full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as patch, pytest.raises(OSError):
        patch.setattr(FastPlaid, "create", disk_full)
        documents("plan.md", "The Kestrel launch moved to June.")

    assert colbert("Kestrel launch")[0].text == "The Kestrel launch is in March."
    assert hybrid("Kestrel launch")[0].text == "The Kestrel launch is in March."


def test_an_index_swap_that_fails_after_the_commit_is_rebuilt_at_once(documents, monkeypatch):
    from perry import colbert as colbert_index

    documents("plan.md", "The Kestrel launch is in March.")
    documents("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")
    swap = colbert_index._swap
    failures = [PermissionError(32, "The file is in use by another process")]

    def swap_once_locked():
        if failures:
            raise failures.pop()
        swap()

    monkeypatch.setattr(colbert_index, "_swap", swap_once_locked)

    documents("plan.md", "The Kestrel launch moved to June.")  # Postgres has committed

    assert not failures  # the swap did fail
    assert colbert("Kestrel launch")[0].text == "The Kestrel launch moved to June."
    assert hybrid("Kestrel launch")[0].text == "The Kestrel launch moved to June."


def test_an_index_swap_that_keeps_failing_is_reported_and_repaired_at_start(
    documents, monkeypatch
):
    from perry import colbert as colbert_index

    documents("plan.md", "The Kestrel launch is in March.")
    documents("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")

    def locked():
        raise PermissionError(32, "The file is in use by another process")

    with monkeypatch.context() as patch, pytest.raises(PermissionError):
        patch.setattr(colbert_index, "_swap", locked)
        documents("plan.md", "The Kestrel launch moved to June.")

    colbert_index.ensure_in_step()  # what the next start does

    assert colbert("Kestrel launch")[0].text == "The Kestrel launch moved to June."


def test_documents_stored_without_colbert_are_encoded_and_indexed_at_start(documents, tmp_path):
    from perry import colbert as colbert_index

    old = tmp_path / "notes.md"
    old.write_text("The Kestrel launch is in March.", encoding="utf-8")
    with models.use(fake_provider(ScriptedChatModel())):  # no ColBERT encoder yet
        ingest_file(old, embed=fake_embed)
    documents("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")

    colbert_index.ensure_in_step()

    assert colbert("Kestrel launch")[0].citation == "notes.md"


def test_an_index_lost_after_a_commit_is_rebuilt_at_start(documents):
    import shutil

    from perry import colbert as colbert_index

    documents("plan.md", "The Kestrel launch is in March.")
    documents("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")
    colbert("warm up")  # the index is open, as in a running Perry
    colbert_index.close()
    shutil.rmtree(colbert_index.folder())

    colbert_index.ensure_in_step()

    assert colbert("Kestrel launch")[0].citation == "plan.md"


def test_forcing_colbert_in_a_turn_answers_with_citations_and_found_with(
    perry, tmp_path, monkeypatch
):
    monkeypatch.setenv("PERRY_APP_DATA_DIR", str(tmp_path / "app-data"))
    with models.use(fake_provider(perry.model, colbert=FakeColbert())):
        perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
        perry.add_document("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")
        perry.script(
            call("ask_knowledge_agent", question="What is the Q3 marketing budget?"),
            call("colbert", question="Q3 marketing budget"),
            last_message,  # Knowledge Agent reports the Chunks it found
            last_message,  # Supervisor passes the findings on
        )

        reply = perry.send("How big is my Q3 marketing budget?", technique="colbert")

    assert reply.startswith("The marketing budget for Q3 is 40,000 dollars.")
    *_, citations, found_with = reply.splitlines()
    assert citations.startswith("Citations: [budget.md]")
    assert found_with == "Found with: ColBERT (forced)"


@pytest.fixture
def real_colbert():
    """The real ColBERT model. Skips unless it is installed and downloaded."""
    pytest.importorskip("pylate")
    from huggingface_hub import try_to_load_from_cache

    model = models.DEFAULT_COLBERT_MODEL
    if not isinstance(try_to_load_from_cache(model, "config.json"), str):
        pytest.skip(f"{model} isn't downloaded (Perry fetches it on first use)")
    return models.LocalColbert(model)


def test_the_real_colbert_model_finds_an_exact_name(
    database, tmp_path, monkeypatch, real_colbert
):
    monkeypatch.setenv("PERRY_APP_DATA_DIR", str(tmp_path / "app-data"))
    texts = {
        "suppliers.md": "Our lidar sensors come from Brightwave Optics, based in Eindhoven.",
        "kestrel.md": "Project Kestrel-7 is owned by Priya Raman and ships in May.",
        "hiking.md": "Trail notes: the ridge path is steep after the waterfall.",
    }
    with models.use(fake_provider(ScriptedChatModel(), colbert=real_colbert)):
        for name, text in texts.items():
            (tmp_path / name).write_text(text, encoding="utf-8")
            ingest_file(tmp_path / name, embed=fake_embed)

        hits = colbert("Who owns Kestrel-7?")

    assert hits[0].citation == "kestrel.md"


def test_each_database_keeps_its_own_index(monkeypatch, tmp_path):
    from perry import colbert, db

    monkeypatch.setenv("PERRY_APP_DATA_DIR", str(tmp_path))
    server = "postgresql://perry:perry@127.0.0.1:5433"
    folders = {}
    for name in ("perry", "perry_eval", "perry_benchmark"):
        monkeypatch.setattr(db, "DATABASE_URL", f"{server}/{name}")
        folders[name] = colbert.folder()

    assert len(set(folders.values())) == 3  # an eval run never rebuilds the app's index
    assert folders["perry"] == tmp_path / "colbert"  # the app's index stays where it was


CHANGE_IN_ANOTHER_PROCESS = """
import sys
from pathlib import Path

sys.path[:0] = [sys.argv[1], str(Path(sys.argv[1]).parent)]
from fakes import FakeColbert, ScriptedChatModel, fake_embed, fake_provider

from perry import models
from perry.ingest import ingest_file

with models.use(fake_provider(ScriptedChatModel(), colbert=FakeColbert())):
    ingest_file(Path(sys.argv[2]), embed=fake_embed)
"""


def test_a_change_from_the_command_line_while_perry_runs_reaches_perrys_searches(
    documents, tmp_path
):
    import subprocess
    import sys
    from pathlib import Path

    documents("plan.md", "The Kestrel launch is in March.")
    documents("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")
    colbert("Kestrel launch")  # the index is open, as in a running Perry
    newer = tmp_path / "plan.md"
    newer.write_text("The Kestrel launch moved to June.", encoding="utf-8")

    script = tmp_path / "change.py"
    script.write_text(CHANGE_IN_ANOTHER_PROCESS, encoding="utf-8")
    tests = Path(__file__).parent
    done = subprocess.run(
        [sys.executable, str(script), str(tests), str(newer)],
        cwd=tests.parent, capture_output=True, text=True, timeout=300, check=False,
    )

    assert done.returncode == 0, done.stderr[-2000:]
    assert colbert("Kestrel launch")[0].text == "The Kestrel launch moved to June."
