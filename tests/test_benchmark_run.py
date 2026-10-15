"""Benchmark runner: asks every question once, survives quota errors, compares systems."""

import pytest

from benchmark.run import (
    answer_all,
    load_answers,
    render_report,
    retrieval_report,
    summarise,
)
from benchmark.scoring import Question


def make_question(id: str, type: str, fact: str, page: str) -> Question:
    return Question(id=id, type=type, question=f"question {id}?", facts=((fact,),), pages=(page,))


QUESTIONS = [
    make_question("L1", "lookup", "84.2", "a.pdf#page=1"),
    make_question("L2", "lookup", "Brno", "a.pdf#page=3"),
    make_question("M1", "multi_hop", "Tampere", "b.pdf#page=2"),
]


def test_every_question_is_answered_and_saved(tmp_path):
    path = tmp_path / "answers.jsonl"

    answer_all(QUESTIONS, lambda q: f"answer to {q}", path)

    answers = load_answers(path)
    assert set(answers) == {"L1", "L2", "M1"}
    assert answers["L1"]["answer"] == "answer to question L1?"


def test_a_failed_question_stops_the_run_and_a_rerun_resumes_from_it(tmp_path):
    path = tmp_path / "answers.jsonl"
    asked = []

    def quota_runs_out_on_l2(question: str) -> str:
        asked.append(question)
        if question == "question L2?":
            raise RuntimeError("429 quota exhausted")
        return "ok"

    with pytest.raises(RuntimeError):
        answer_all(QUESTIONS, quota_runs_out_on_l2, path)
    assert set(load_answers(path)) == {"L1"}

    asked.clear()
    answer_all(QUESTIONS, lambda q: asked.append(q) or "ok", path)

    assert asked == ["question L2?", "question M1?"]
    assert set(load_answers(path)) == {"L1", "L2", "M1"}


def test_summary_compares_systems_per_question_type():
    answers = {
        "perry": {
            "L1": {"answer": "84.2 [a.pdf#page=1]"},
            "L2": {"answer": "I don't know."},
            "M1": {"answer": "Tampere [b.pdf#page=2]"},
        },
        "qabot": {
            "L1": {"answer": "84.2 [a.pdf#page=1]"},
            "L2": {"answer": "Brno [a.pdf#page=3]"},
            # M1 not answered yet.
        },
    }

    summary = summarise(QUESTIONS, answers)

    perry_lookup = summary[("lookup", "perry")]
    assert perry_lookup.answered == 2
    assert perry_lookup.correct == 0.5
    assert perry_lookup.citation_recall == 0.5
    assert summary[("lookup", "qabot")].correct == 1.0
    assert summary[("multi_hop", "qabot")].answered == 0
    assert summary[("multi_hop", "perry")].correct == 1.0


def test_report_shows_the_per_type_comparison_and_every_answer():
    answers = {"perry": {"L1": {"answer": "84.2 [a.pdf#page=1]"}}, "qabot": {}}

    report = render_report(QUESTIONS, answers)

    assert "| lookup | perry | 1/2 | 100% |" in report
    assert "| lookup | qabot | 0/2 |" in report
    assert "question L1?" in report


def test_the_retrieval_report_compares_hybrid_search_with_reranked_page_recall():
    plain = {"L1": {"a.pdf#page=1"}, "L2": {"x.pdf#page=9"}, "M1": set()}
    reranked = {"L1": {"a.pdf#page=1"}, "L2": {"a.pdf#page=3"}, "M1": {"b.pdf#page=2"}}

    report = retrieval_report(QUESTIONS, 6, plain, reranked)

    assert "| lookup | 2 | 50% | 1/2 | 100% | 2/2 |" in report
    assert "| all | 3 | 33% | 1/3 | 100% | 3/3 |" in report
    assert "| L2 | 0% | a.pdf#page=3 | 100% | - |" in report


def test_without_a_reranker_the_retrieval_report_shows_hybrid_search_alone():
    plain = {"L1": {"a.pdf#page=1"}, "L2": set(), "M1": set()}

    report = retrieval_report(QUESTIONS, 6, plain, None)

    assert "| lookup | 2 | 50% | 1/2 |" in report
    assert "Reranked" not in report
