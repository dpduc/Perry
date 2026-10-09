"""Every registered Retrieval Technique, as a chain over a real Postgres with fake embeddings."""

import pytest
from fakes import (
    FakeColbert,
    FakeSummaryModel,
    ScriptedChatModel,
    fake_embed,
    fake_provider,
    with_usage,
)

from perry import models, summaries, techniques
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
    # A technique that asks the model (e.g. to rewrite the question) gets the question back.
    restated = ["What is the Q3 marketing budget?"] * 5
    chat, summary = ScriptedChatModel(steps=restated), FakeSummaryModel()
    with models.use(fake_provider(chat, summary=summary, colbert=FakeColbert())):
        for name in ("budget.md", "hiking.md"):
            ingest_file(tmp_path / name, embed=fake_embed)
        summaries.summarise_pending()  # the background job, for Summary-based techniques
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


def ingest(tmp_path, documents: dict[str, str]) -> None:
    for name, text in documents.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
        ingest_file(tmp_path / name, embed=fake_embed)


def ingest_fruit(tmp_path):
    ingest(tmp_path, {"alpha.md": "alpha apples", "beta.md": "beta bananas", "gamma.md": "gamma grapes"})


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


def test_step_back_finds_chunks_for_the_original_and_the_broader_question(database, tmp_path):
    from perry import techniques, usage

    question = "Why did the Q3 marketing budget rise?"
    ingest(
        tmp_path,
        {
            "q3.md": "The Q3 marketing budget rose to 40,000 dollars.",
            "policy.md": "Company policy: spending follows expected customer growth.",
            # Notes sharing a word with the question, which the policy doesn't, so plain
            # hybrid search ranks them above it.
            **{f"note{n}.md": f"Q3 note {n}: plants watered." for n in range(8)},
        },
    )
    broader = with_usage("What principles drive company spending and customer growth?", 30, 10)
    model = ScriptedChatModel(steps=[broader])

    with models.use(fake_provider(model)):
        plain = [hit.citation for hit in techniques.get("hybrid_search").chain.invoke(question)]
        hits = techniques.get("step_back").chain.invoke(question)

    assert "policy.md" not in plain
    citations = [hit.citation for hit in hits]
    assert "q3.md" in citations and "policy.md" in citations
    assert usage.today("step_back").tokens == 40


def test_hyde_searches_with_a_hypothetical_answer_and_never_cites_it(database, tmp_path):
    from perry import techniques, usage

    # Worded nothing like the Document it should find.
    question = "How much can promo spend this summer?"
    ingest(
        tmp_path,
        {
            "budget.md": "The marketing budget for Q3 is 40,000 dollars.",
            "hiking.md": "Trail notes: the ridge path is steep after the waterfall.",
            "summer.md": "Summer hours: the office closes at 3pm this summer.",
        },
    )
    hypothetical = "The marketing budget for Q3 is about 25,000 dollars."
    model = ScriptedChatModel(steps=[with_usage(hypothetical, 20, 15)])

    with models.use(fake_provider(model)):
        plain = techniques.get("hybrid_search").chain.invoke(question)
        hits = techniques.get("hyde").chain.invoke(question)

    assert plain[0].citation != "budget.md"
    assert hits[0].citation == "budget.md"
    assert "40,000" in hits[0].text
    assert all("25,000" not in hit.text for hit in hits)
    assert usage.today("hyde").tokens == 35
