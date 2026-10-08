"""The benchmark question set is consistent with the Documents both systems ingest."""

from collections import Counter

import pymupdf

from benchmark.corpus import DOCUMENTS_DIR
from benchmark.run import QUESTIONS_PATH
from benchmark.scoring import QUESTION_TYPES, _normalise, load_questions

QUESTIONS = load_questions(QUESTIONS_PATH)


def page_texts() -> dict[str, str]:
    texts = {}
    for path in sorted(DOCUMENTS_DIR.glob("*.pdf")):
        with pymupdf.open(path) as pdf:
            for number, page in enumerate(pdf, start=1):
                texts[f"{path.name}#page={number}"] = _normalise(page.get_text())
    return texts


PAGES = page_texts()


def test_at_least_20_questions_covering_every_question_type():
    counts = Counter(q.type for q in QUESTIONS)

    assert len(QUESTIONS) >= 20
    assert set(counts) == set(QUESTION_TYPES)
    assert min(counts.values()) >= 4


def test_question_ids_are_unique():
    assert len({q.id for q in QUESTIONS}) == len(QUESTIONS)


def test_every_source_page_exists_in_the_document_set():
    for q in QUESTIONS:
        for page in q.pages + q.related_pages:
            assert page in PAGES, f"{q.id}: no such page {page}"


def test_every_expected_fact_is_written_on_one_of_its_source_pages():
    for q in QUESTIONS:
        sources = [PAGES[p] for p in q.pages + q.related_pages]
        for fact in q.facts:
            assert any(
                _normalise(phrasing) in text for phrasing in fact for text in sources
            ), f"{q.id}: fact {fact} is on none of its pages"


def test_committed_documents_match_their_sources(tmp_path):
    from benchmark.corpus import build

    for fresh in build(tmp_path):
        committed = DOCUMENTS_DIR / fresh.name
        assert committed.exists(), f"rebuild with `python -m benchmark.corpus`: {fresh.name}"
        with pymupdf.open(fresh) as a, pymupdf.open(committed) as b:
            assert [p.get_text() for p in a] == [p.get_text() for p in b], fresh.name


def test_multi_hop_questions_need_more_than_one_document():
    for q in QUESTIONS:
        if q.type == "multi_hop":
            documents = {page.split("#")[0] for page in q.pages}
            assert len(documents) >= 2, q.id
