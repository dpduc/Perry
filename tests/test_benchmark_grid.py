"""The benchmark grid: every Retrieval Technique forced and every picking mode, on the
same questions, scored for correctness, Citation accuracy and cost."""

from datetime import date

import pytest
from fakes import (
    FakeColbert,
    FakeSummaryModel,
    ScriptedChatModel,
    call,
    fake_provider,
    last_message,
    with_usage,
)

from benchmark import grid, run
from benchmark.grid import Answer, load_grid, render_grid, run_grid, summarise_grid
from benchmark.scoring import Question, load_questions
from perry import models, picking, raptor, summaries, techniques
from perry.techniques import Technique, hybrid


def make_question(id: str, type: str, fact: str, page: str) -> Question:
    return Question(id=id, type=type, question=f"question {id}?", facts=((fact,),), pages=(page,))


QUESTIONS = [
    make_question("L1", "lookup", "84.2", "a.pdf#page=1"),
    make_question("T1", "thematic", "Tampere", "b.pdf#page=2"),
]


def scripted_ask(answers: dict[tuple[str, str], str], asked: list | None = None):
    """An `ask` that answers from `answers` by (setup, question), costing 1 cent each."""

    def ask(setup: str, question: str) -> Answer:
        if asked is not None:
            asked.append((setup, question))
        return Answer(
            text=answers.get((setup, question), "I don't know."),
            found_with=(f"{setup} (forced)",),
            cost_usd=0.01,
            tokens=1200,
            model="gemini-test",
        )

    return ask


def test_a_run_records_a_scored_row_per_setup_and_question(tmp_path):
    path = tmp_path / "grid.jsonl"
    ask = scripted_ask({("hyde", "question L1?"): "Revenue was 84.2 [a.pdf#page=1]."})

    run_grid(QUESTIONS, ["hyde", "cascade"], path, ask, today=date(2026, 10, 9))

    rows = load_grid(path)
    assert set(rows) == {("hyde", "L1"), ("hyde", "T1"), ("cascade", "L1"), ("cascade", "T1")}
    right = rows[("hyde", "L1")]
    assert right["correct"] is True
    assert right["citation_recall"] == 1.0
    assert right["citation_precision"] == 1.0
    assert right["cost_usd"] == 0.01
    assert right["date"] == "2026-10-09"
    assert right["model"] == "gemini-test"
    assert right["type"] == "lookup"
    assert right["found_with"] == ["hyde (forced)"]
    wrong = rows[("cascade", "T1")]
    assert wrong["correct"] is False
    assert wrong["citation_recall"] == 0.0


def test_a_run_limited_to_one_technique_reruns_only_that_technique(tmp_path):
    path = tmp_path / "grid.jsonl"
    run_grid(QUESTIONS, ["hyde", "cascade"], path, scripted_ask({}), today=date(2026, 10, 1))
    asked = []
    better = scripted_ask({("hyde", "question L1?"): "84.2 [a.pdf#page=1]"}, asked)

    run_grid(QUESTIONS, ["hyde"], path, better, rerun=True, today=date(2026, 10, 9))

    assert asked == [("hyde", "question L1?"), ("hyde", "question T1?")]
    rows = load_grid(path)
    assert rows[("hyde", "L1")]["correct"] is True
    assert rows[("hyde", "L1")]["date"] == "2026-10-09"
    assert rows[("cascade", "L1")]["date"] == "2026-10-01"  # kept, not rerun
    assert len(rows) == 4


def test_a_stopped_run_resumes_from_the_first_unanswered_row(tmp_path):
    path = tmp_path / "grid.jsonl"
    answers = scripted_ask({})

    def quota_runs_out_on_cascade(setup: str, question: str) -> Answer:
        if setup == "cascade":
            raise RuntimeError("429 quota exhausted")
        return answers(setup, question)

    with pytest.raises(RuntimeError):
        run_grid(QUESTIONS, ["hyde", "cascade"], path, quota_runs_out_on_cascade)
    asked = []
    run_grid(QUESTIONS, ["hyde", "cascade"], path, scripted_ask({}, asked))

    assert asked == [("cascade", "question L1?"), ("cascade", "question T1?")]
    assert len(load_grid(path)) == 4


QUESTIONS_3 = [*QUESTIONS, make_question("L2", "lookup", "Brno", "a.pdf#page=3")]


def graded_run(tmp_path):
    """hyde: L1 right, L2 right but uncited, T1 wrong. cascade: only L1 answered, right."""
    path = tmp_path / "grid.jsonl"
    ask = scripted_ask(
        {
            ("hyde", "question L1?"): "84.2 [a.pdf#page=1]",
            ("hyde", "question L2?"): "Brno",
            ("cascade", "question L1?"): "84.2 [a.pdf#page=1] [c.pdf#page=9]",
        }
    )
    run_grid(QUESTIONS_3, ["hyde"], path, ask, today=date(2026, 10, 9))
    run_grid(QUESTIONS_3[:1], ["cascade"], path, ask, today=date(2026, 10, 8))
    return load_grid(path)


def test_results_are_summarised_per_setup_and_question_type(tmp_path):
    summary = summarise_grid(QUESTIONS_3, graded_run(tmp_path))

    hyde_lookup = summary[("hyde", "lookup")]
    assert hyde_lookup.answered == 2
    assert hyde_lookup.correct == 1.0
    assert hyde_lookup.citation_recall == 0.5
    assert hyde_lookup.cost_usd == pytest.approx(0.01)  # per question
    hyde_all = summary[("hyde", "all")]
    assert hyde_all.answered == 3
    assert hyde_all.correct == pytest.approx(2 / 3)
    assert hyde_all.total_cost_usd == pytest.approx(0.03)
    assert hyde_all.dates == ("2026-10-09",)
    assert hyde_all.models == ("gemini-test",)
    cascade_lookup = summary[("cascade", "lookup")]
    assert cascade_lookup.answered == 1
    assert cascade_lookup.citation_precision == 0.5
    assert summary[("cascade", "thematic")].answered == 0


def test_the_table_shows_each_setup_per_question_type_with_cost_date_and_model(tmp_path):
    report = render_grid(QUESTIONS_3, graded_run(tmp_path))

    assert "| hyde | 3/3 | 67% |" in report
    assert "| picking: cascade | 1/3 | 100% |" in report
    assert "2026-10-09" in report and "gemini-test" in report
    assert "| hyde | 100% | - | 0% | - | 67% |" in report  # correct per question type


# --- Asking Perry: a forced technique or a picking mode, with its cost -----------------

BUDGET_QUESTION = "What is the Q3 marketing budget?"
BUDGET = Technique(
    name="budget_search",
    label="Budget search",
    description="Search for money figures.",
    examples=("How much is the marketing budget?", "What is the Q4 marketing budget?"),
    chain=hybrid,
)
TRAILS = Technique(
    name="trail_search",
    label="Trail search",
    description="Search hiking notes.",
    examples=("Is the ridge path steep?", "Where does the waterfall trail start?"),
    chain=hybrid,
)


@pytest.fixture
def budget(perry, prices):
    prices({"scripted": {"input": 1000, "output": 2000}})
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    return perry


def test_a_forced_technique_answers_with_its_citations_cost_and_model(budget):
    budget.script(
        with_usage(call("hybrid_search", question=BUDGET_QUESTION), 100, 50),
        with_usage(last_message, 100, 50),  # the Knowledge Agent reports the Chunks
    )

    answer = grid.ask_perry("hybrid_search", BUDGET_QUESTION)

    assert "40,000 dollars" in answer.text
    assert "[budget.md]" in answer.text
    assert answer.found_with == ("Hybrid search (forced)",)
    assert answer.cost_usd == pytest.approx(0.4)  # two calls of 0.1 in + 0.1 out
    assert answer.tokens >= 300
    assert answer.model == "scripted"


def test_a_picking_mode_setup_picks_in_that_mode_whatever_the_setting(
    budget, monkeypatch
):
    monkeypatch.setattr(techniques, "TECHNIQUES", (BUDGET, TRAILS))
    monkeypatch.setenv("PERRY_PICKING_MODE", "logical")
    budget.script(call("budget_search", question=BUDGET_QUESTION), last_message)

    answer = grid.ask_perry("semantic", BUDGET_QUESTION)

    assert answer.found_with == ("Budget search (picked semantically)",)

    monkeypatch.setenv("PERRY_PICKING_MODE", "semantic")
    budget.script(call("trail_search", question=BUDGET_QUESTION), last_message)

    answer = grid.ask_perry("logical", BUDGET_QUESTION)

    assert answer.found_with == ("Trail search (picked logically)",)


def test_the_grid_has_every_technique_forced_and_every_picking_mode():
    every = grid.setups()

    assert every == [*(t.name for t in techniques.TECHNIQUES), *picking.MODES]
    assert len(every) == 13


def test_a_run_can_be_limited_to_named_setups_and_always_hybrid_is_hybrid_search():
    assert grid.select(["raptor", "cascade"]) == ["raptor", "cascade"]
    assert grid.select(["always-hybrid"]) == ["hybrid_search"]
    assert grid.select(["hybrid_search", "always_hybrid"]) == ["hybrid_search"]
    assert grid.select(None) == grid.setups()
    with pytest.raises(ValueError, match="no_such"):
        grid.select(["no_such"])


# --- Preparing the benchmark Documents --------------------------------------------------


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("PERRY_APP_DATA_DIR", str(tmp_path / "app-data"))  # ColBERT index
    folder = tmp_path / "corpus"
    folder.mkdir()
    (folder / "budget.md").write_text(
        "The marketing budget for Q3 is 40,000 dollars.", encoding="utf-8"
    )
    (folder / "hiking.md").write_text(
        "Trail notes: the ridge path is steep after the waterfall.", encoding="utf-8"
    )
    return sorted(folder.iterdir())


def using(chat=None, **models_kwargs):
    return models.use(fake_provider(chat or ScriptedChatModel(), **models_kwargs))


def test_preparing_summary_based_setups_summarises_and_builds_raptor_trees_first(
    database, corpus
):
    with using(summary=FakeSummaryModel()):
        grid.prepare(["document_summaries", "raptor"], corpus)

        assert summaries.document("budget.md").status == "ready"
        assert raptor.status("hiking.md") == "ready"
        hits = techniques.get("raptor").chain.invoke("How steep is the ridge path?")
    assert [hit.document for hit in hits][:1] == ["hiking.md"]


def test_picking_modes_need_summaries_too_as_they_may_pick_a_summary_based_technique(
    database, corpus
):
    summary_model = FakeSummaryModel()
    with using(summary=summary_model, colbert=FakeColbert()):
        grid.prepare(["cascade"], corpus)

    assert summary_model.calls  # summarised


def test_a_run_that_needs_summaries_refuses_to_start_without_the_local_model(database, corpus):
    with using(), pytest.raises(grid.NotReady, match="Ollama"):
        grid.prepare(["raptor"], corpus)


def test_a_run_that_needs_summaries_refuses_to_start_if_summarising_fails(database, corpus):
    failing = FakeSummaryModel(replies=[ConnectionError("Ollama is not running")] * 10)
    with using(summary=failing), pytest.raises(grid.NotReady, match="budget.md"):
        grid.prepare(["document_summaries"], corpus)


def test_colbert_needs_its_encoder(database, corpus):
    with using(), pytest.raises(grid.NotReady, match="ColBERT"):
        grid.prepare(["colbert"], corpus)


def test_a_hybrid_search_run_does_not_wait_for_summaries(database, corpus):
    summary_model = FakeSummaryModel()
    with using(summary=summary_model):
        grid.prepare(["hybrid_search", "hyde"], corpus)

    assert summary_model.calls == []
    assert summaries.document("budget.md").status == "pending"


# --- The command ------------------------------------------------------------------------


def test_the_grid_command_runs_the_named_setups_in_the_benchmark_database(
    monkeypatch, tmp_path
):
    events = []
    monkeypatch.setattr(run, "_use_benchmark_database", lambda: events.append("database"))
    monkeypatch.setattr(grid, "prepare", lambda setups, docs: events.append(("prepare", setups)))
    monkeypatch.setattr(grid, "ask_perry", scripted_ask({}))
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)

    run.main(["grid", "--only", "hyde,always-hybrid"])

    assert run.BENCHMARK_DATABASE_URL.endswith("/perry_benchmark")
    assert events == ["database", ("prepare", ["hybrid_search", "hyde"])]
    rows = load_grid(tmp_path / "grid.jsonl")
    assert {setup for setup, _ in rows} == {"hybrid_search", "hyde"}
    assert len(rows) == 2 * len(load_questions(run.QUESTIONS_PATH))
    assert "| hyde | 23/23 |" in (tmp_path / "GRID.md").read_text(encoding="utf-8")
