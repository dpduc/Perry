"""Every registered Retrieval Technique, as a chain over a real Postgres with fake embeddings."""

import pytest
from fakes import ScriptedChatModel, fake_embed, fake_provider

from perry import models, techniques
from perry.ingest import ingest_file
from perry.techniques import TECHNIQUES

# What the fake model replies to each technique's own model calls, in call order.
SCRIPTS = {
    "multi_query": ["Q3 marketing budget\nmarketing spend in the third quarter"],
    "rag_fusion": ["Q3 marketing budget\nmarketing spend in the third quarter"],
}


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

    script = ScriptedChatModel(steps=list(SCRIPTS.get(technique.name, [])))
    with models.use(fake_provider(script)):
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


def ingest_fruit(tmp_path):
    for name, text in [
        ("alpha.md", "alpha apples"),
        ("beta.md", "beta bananas"),
        ("gamma.md", "gamma grapes"),
    ]:
        (tmp_path / name).write_text(text, encoding="utf-8")
        ingest_file(tmp_path / name, embed=fake_embed)


def test_rag_fusion_ranks_first_the_chunks_most_rewrites_agree_on(database, tmp_path):
    ingest_fruit(tmp_path)
    script = ScriptedChatModel(steps=["alpha\nbeta\nbeta bananas"])

    with models.use(fake_provider(script)):
        hits = techniques.get("rag_fusion").chain.invoke("Which fruit?")

    # The first rewrite ranks alpha first, but two rewrites agree on beta.
    assert hits[0].document == "beta.md"
    assert hits[0].score > hits[1].score


def test_multi_query_returns_the_union_of_every_rewrites_chunks_once_each(database, tmp_path):
    ingest_fruit(tmp_path)
    script = ScriptedChatModel(steps=["alpha\nbeta\nbeta bananas"])

    with models.use(fake_provider(script)):
        hits = techniques.get("multi_query").chain.invoke("Which fruit?")

    documents = [hit.document for hit in hits]
    assert documents[0] == "alpha.md"  # the first rewrite's best Chunk
    assert "beta.md" in documents  # the other rewrites' best Chunk
    assert len(set(documents)) == len(documents)  # each Chunk once
