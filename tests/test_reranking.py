"""Reranking: a local cross-encoder scores each Chunk a Retrieval Technique found against
the question, so the best-answering Chunks reach the Knowledge Agent first."""

import pytest
from fakes import (
    FakeReranker,
    ScriptedChatModel,
    call,
    fake_embed,
    fake_provider,
    last_message,
)

from perry import models, techniques, usage
from perry.ingest import ingest_file
from perry.search import hybrid_search

QUESTION = "When was the budget approved by Dana?"
ANSWER = "minutes.md"


@pytest.fixture
def library(database, tmp_path):
    """Nine notes that repeat the question's key words, and the meeting minutes that
    answer it in passing. Hybrid search ranks the minutes below the notes; a reranker
    that reads the whole question ranks them first."""
    texts = {
        f"note-{n}.md": f"Note {n}: budget approved Dana. Budget approved Dana. Budget Dana."
        for n in range(1, 10)
    }
    texts[ANSWER] = (
        "Minutes of the finance meeting. The room was booked for the afternoon and the "
        "projector was broken again. Dana opened with travel, then hiring, then the "
        "office move. The budget was approved by Dana on 3 May, when the meeting ended."
    )
    for name, text in texts.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        with models.use(fake_provider(ScriptedChatModel())):
            ingest_file(path, embed=fake_embed)


def search_with(reranker, technique: str = "hybrid_search"):
    with models.use(fake_provider(ScriptedChatModel(), reranker=reranker)):
        return techniques.search(techniques.get(technique), QUESTION)


def test_reranking_brings_a_chunk_from_beyond_the_usual_six_to_the_top(library):
    with models.use(fake_provider(ScriptedChatModel())):
        usual = [hit.document for hit in hybrid_search(QUESTION, limit=6)]
    assert ANSWER not in usual  # the setup: plain hybrid search misses it

    found = search_with(FakeReranker())

    assert found.reranked
    assert found.hits[0].document == ANSWER
    assert len(found.hits) == 6



def test_every_candidate_keeps_its_rerank_score_and_its_rank_before_reranking(library):
    found = search_with(FakeReranker())

    assert len(found.candidates) == 10  # the whole pool, not just the 6 kept
    best = found.candidates[0]
    assert best.citation.startswith(ANSWER)
    assert best.rank_before == 10  # hybrid search ranked the minutes last
    scores = [c.score for c in found.candidates]
    assert scores == sorted(scores, reverse=True)
    assert [c.kept for c in found.candidates] == [True] * 6 + [False] * 4
    assert [c.citation for c in found.candidates if c.kept] == [
        hit.citation for hit in found.hits
    ]


class ScoresBy(FakeReranker):
    """Scores a text by the first of `scores` whose key is in it."""

    def __init__(self, **scores: float):
        self.scores = scores

    def score(self, question, texts):
        return [next(s for key, s in self.scores.items() if key in t) for t in texts]


def test_chunks_scored_below_the_cutoff_are_left_out(library, monkeypatch):
    monkeypatch.setattr(models, "RERANK_MIN_SCORE", -8.0)
    low = models.RERANK_MIN_SCORE - 1
    found = search_with(ScoresBy(Minutes=5.0, **{"Note 1:": -1.0, "Note 2:": -2.0, "Note": low}))

    assert [hit.document for hit in found.hits] == [ANSWER, "note-1.md", "note-2.md"]
    assert sum(c.kept for c in found.candidates) == 3


def test_the_best_chunk_is_kept_even_when_every_chunk_scores_below_the_cutoff(
    library, monkeypatch
):
    monkeypatch.setattr(models, "RERANK_MIN_SCORE", -8.0)
    low = models.RERANK_MIN_SCORE - 1
    found = search_with(ScoresBy(Minutes=low, Note=low - 1))

    assert [hit.document for hit in found.hits] == [ANSWER]


@pytest.fixture
def real_reranker():
    """The real reranker. Skips unless it is installed and downloaded."""
    pytest.importorskip("sentence_transformers")
    from huggingface_hub import try_to_load_from_cache

    model = models.DEFAULT_RERANK_MODEL
    if not isinstance(try_to_load_from_cache(model, "config.json"), str):
        pytest.skip(f"{model} isn't downloaded (Perry fetches it on first use)")
    return models.LocalReranker(model)


def test_the_real_reranker_scores_the_text_that_answers_highest(real_reranker):
    texts = [
        "Trail notes: the ridge path is steep after the waterfall.",
        "Project Kestrel-7 is owned by Priya Raman and ships in May.",
        "Kestrels are small falcons that hover before they dive.",
    ]

    scores = real_reranker.score("Who owns Project Kestrel?", texts)

    assert max(range(len(texts)), key=scores.__getitem__) == 1


def test_reranking_is_recorded_in_the_usage_ledger_as_a_local_call(database):
    with models.use(fake_provider(ScriptedChatModel(), reranker=FakeReranker())):
        models.rerank("Who owns it?", ["Priya owns Kestrel.", "Trail notes."])

    # The question is read with each text: "Who owns it? Priya owns Kestrel." is 32
    # characters (8 tokens at 4 a token) and "Who owns it? Trail notes." is 25 (7).
    assert usage.today("rerank").tokens == 15
    assert usage.today("rerank").cost_usd == 0


def test_without_a_reranker_a_search_keeps_the_techniques_own_six(library):
    with models.use(fake_provider(ScriptedChatModel())):
        usual = hybrid_search(QUESTION, limit=6)

    found = search_with(reranker=None)

    assert not found.reranked
    assert found.hits == usual


class BrokenReranker(FakeReranker):
    def score(self, question: str, texts: list[str]) -> list[float]:
        raise RuntimeError("out of memory")


def test_a_failed_reranker_keeps_the_techniques_best_six_and_says_why(library):
    with models.use(fake_provider(ScriptedChatModel())):
        usual = hybrid_search(QUESTION, limit=6)

    found = search_with(BrokenReranker())

    assert not found.reranked
    assert found.hits == usual
    assert found.problem == "Reranking failed (out of memory)"


def found_with_line(perry, reranker) -> str:
    """The Found-with line of a Turn whose question the Knowledge Agent searches with
    hybrid search (forced), with `reranker` in use."""
    with models.use(fake_provider(perry.model, reranker=reranker)):
        perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
        perry.script(
            call("ask_knowledge_agent", question="What is the Q3 marketing budget?"),
            call("hybrid_search", question="Q3 marketing budget"),
            last_message,  # Knowledge Agent reports the Chunks it found
            last_message,  # Supervisor passes the findings on
        )
        reply = perry.send("How big is my Q3 marketing budget?", technique="hybrid_search")
    return reply.splitlines()[-1]


def test_the_answer_says_its_chunks_were_reranked(perry):
    assert found_with_line(perry, FakeReranker()) == "Found with: Hybrid search, reranked (forced)"


def test_the_answer_says_when_reranking_failed(perry):
    assert found_with_line(perry, BrokenReranker()) == (
        "Found with: Hybrid search, reranking failed (forced)"
    )


def test_without_a_reranker_the_answer_names_only_the_technique(perry):
    assert found_with_line(perry, None) == "Found with: Hybrid search (forced)"


def test_perry_loads_the_reranker_before_the_first_question():
    class CountingReranker(FakeReranker):
        calls = 0

        def score(self, question, texts):
            CountingReranker.calls += 1
            return super().score(question, texts)

    with models.use(fake_provider(ScriptedChatModel(), reranker=CountingReranker())):
        models.warm_up_reranker().join()

    assert CountingReranker.calls == 1


def test_without_a_reranker_there_is_nothing_to_load():
    with models.use(fake_provider(ScriptedChatModel())):
        assert models.warm_up_reranker() is None
