"""Every registered Retrieval Technique, as a chain over a real Postgres with fake embeddings."""

import pytest
from fakes import ScriptedChatModel, fake_embed, fake_provider

from perry import models
from perry.ingest import ingest_file
from perry.techniques import TECHNIQUES


@pytest.mark.parametrize("technique", TECHNIQUES, ids=lambda t: t.name)
def test_technique_returns_chunks_with_page_citations(database, tmp_path, technique):
    (tmp_path / "budget.md").write_text(
        "The marketing budget for Q3 is 40,000 dollars.", encoding="utf-8"
    )
    (tmp_path / "hiking.md").write_text(
        "Trail notes: the ridge path is steep after the waterfall.", encoding="utf-8"
    )
    for name in ("budget.md", "hiking.md"):
        ingest_file(tmp_path / name, embed=fake_embed)

    # A technique that asks the model (e.g. to rewrite the question) gets the question back.
    restated = ["What is the Q3 marketing budget?"] * 5
    with models.use(fake_provider(ScriptedChatModel(steps=restated))):
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


def test_question_decomposition_returns_every_sub_questions_chunks(database, tmp_path):
    from perry import techniques

    documents = {
        "budget.md": "The marketing budget for Q3 is 40,000 dollars.",
        "hiking.md": "Trail notes: the ridge path is steep after the waterfall.",
    }
    # Documents that match the question's own wording, but answer neither part of it.
    for n in range(8):
        documents[f"memo-{n}.md"] = f"Memo {n}: how the two plans compare is still open."
    for name, text in documents.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
        ingest_file(tmp_path / name, embed=fake_embed)
    sub_questions = "What is the Q3 marketing budget?\nHow steep is the ridge path?"

    with models.use(fake_provider(ScriptedChatModel(steps=[sub_questions]))):
        hits = techniques.get("question_decomposition").chain.invoke(
            "How do the two plans compare: the Q3 budget and the ridge hike?"
        )

    best = {hit.citation: hit.text for hit in hits[:2]}
    assert "40,000" in best["budget.md"]
    assert "ridge path is steep" in best["hiking.md"]
