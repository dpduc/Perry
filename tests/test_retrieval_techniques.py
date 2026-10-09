"""Every registered Retrieval Technique, as a chain over a real Postgres with fake embeddings."""

import pytest
from fakes import FakeColbert, ScriptedChatModel, fake_embed, fake_provider

from perry import models
from perry.ingest import ingest_file
from perry.techniques import TECHNIQUES


@pytest.mark.parametrize("technique", TECHNIQUES, ids=lambda t: t.name)
def test_technique_returns_chunks_with_page_citations(
    database, tmp_path, monkeypatch, technique
):
    monkeypatch.setenv("PERRY_APP_DATA_DIR", str(tmp_path / "app-data"))
    (tmp_path / "budget.md").write_text(
        "The marketing budget for Q3 is 40,000 dollars.", encoding="utf-8"
    )
    (tmp_path / "hiking.md").write_text(
        "Trail notes: the ridge path is steep after the waterfall.", encoding="utf-8"
    )
    with models.use(fake_provider(ScriptedChatModel(), colbert=FakeColbert())):
        for name in ("budget.md", "hiking.md"):
            ingest_file(tmp_path / name, embed=fake_embed)

        hits = technique.chain.invoke("What is the Q3 marketing budget?")

    assert hits[0].citation == "budget.md"
    assert "40,000" in hits[0].text


@pytest.mark.parametrize("technique", TECHNIQUES, ids=lambda t: t.name)
def test_technique_registers_a_label_description_and_example_questions(technique):
    assert technique.label and technique.description
    assert len(technique.examples) >= 5
    assert all(example.endswith("?") for example in technique.examples)


def test_technique_names_are_unique_tool_names():
    names = [t.name for t in TECHNIQUES]
    assert len(set(names)) == len(names)
    assert all(name.isidentifier() for name in names)
