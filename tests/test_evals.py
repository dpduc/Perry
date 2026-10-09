"""The eval harness's hard checks (routing, Citations) and its non-gating judge, with fakes.

The eval itself runs on demand against the real model: `uv run python -m evals`.
"""

from dataclasses import replace

from fakes import call, last_message

from evals.harness import EvalTurn, cites, feedback, run_eval_turn

BUDGET_TURN = EvalTurn(
    id="q3-budget",
    message="How big is my Q3 marketing budget?",
    agents=("knowledge",),
    citations=("budget.md",),
    facts=("The Q3 marketing budget is 40,000 dollars.",),
)


def ask_knowledge_agent_and_echo(perry):
    perry.script(
        call("ask_knowledge_agent", question="What is the Q3 marketing budget?"),  # Supervisor
        call("hybrid_search", question="Q3 marketing budget"),  # Knowledge Agent
        last_message,  # Knowledge Agent reports the passages it found
        last_message,  # Supervisor passes the findings on
    )


def test_turn_that_consults_the_expected_agent_and_cites_the_document_passes(perry):
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    ask_knowledge_agent_and_echo(perry)
    perry.script('{"score": 5, "comment": "Correct and cited."}')  # judge

    result = run_eval_turn(BUDGET_TURN, perry)

    assert result.passed
    assert result.consulted == ["knowledge"]
    assert result.quality == 1.0


def test_turn_answered_without_the_expected_agent_fails_routing(perry):
    perry.script("Your Q3 marketing budget is probably around 40,000 dollars [budget.md].")
    perry.script('{"score": 4, "comment": "Plausible."}')

    result = run_eval_turn(BUDGET_TURN, perry)

    assert not result.routing_ok
    assert not result.passed


def test_agent_consulted_when_none_was_expected_fails_routing(perry):
    small_talk = EvalTurn(id="hello", message="Hello there", agents=())
    perry.script(
        call("ask_knowledge_agent", question="Hello"),
        call("hybrid_search", question="Hello"),
        "The Documents don't mention that.",
        "Hi! How can I help?",
        '{"score": 5, "comment": "Friendly."}',
    )

    result = run_eval_turn(small_talk, perry)

    assert result.consulted == ["knowledge"]
    assert not result.passed


def test_reply_without_the_required_citation_fails_the_citation_check(perry):
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    perry.script(
        call("ask_knowledge_agent", question="What is the Q3 marketing budget?"),
        call("hybrid_search", question="Q3 marketing budget"),
        "The Q3 marketing budget is 40,000 dollars.",  # the Knowledge Agent drops the Citation
        "Your Q3 marketing budget is 40,000 dollars.",
        '{"score": 5, "comment": "Correct."}',
    )

    result = run_eval_turn(BUDGET_TURN, perry)

    assert result.routing_ok
    assert result.missing_citations == ["budget.md"]
    assert not result.passed


def test_citation_of_a_page_counts_for_its_document_but_not_for_another_page():
    reply = "Revenue grew 12% [report.pdf#page=4; notes.md]."

    assert cites(reply, "report.pdf")
    assert cites(reply, "report.pdf#page=4")
    assert cites(reply, "notes.md")
    assert not cites(reply, "report.pdf#page=5")
    assert not cites(reply, "report")
    assert not cites("report.pdf#page=4 without brackets", "report.pdf")


def test_a_poor_judge_score_or_an_unusable_judge_never_fails_the_turn(perry):
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    ask_knowledge_agent_and_echo(perry)
    perry.script('{"score": 1, "comment": "Just pastes the passage."}')
    poor = run_eval_turn(BUDGET_TURN, perry)

    ask_knowledge_agent_and_echo(perry)
    perry.script("I think it is fine.")  # no JSON
    unusable = run_eval_turn(BUDGET_TURN, perry)

    assert (poor.passed, poor.quality) == (True, 0.0)
    assert (unusable.passed, unusable.quality) == (True, None)


def test_turn_that_crashes_fails_without_stopping_the_run(perry):
    perry.script(RuntimeError("model unavailable"))

    result = run_eval_turn(BUDGET_TURN, perry)

    assert not result.passed
    assert result.error == "RuntimeError: model unavailable"


def test_eval_turns_have_unique_ids_and_cite_only_eval_documents():
    from evals.turns import DOCUMENTS_DIR, all_turns

    turns = all_turns()
    documents = {path.name for path in DOCUMENTS_DIR.iterdir()}

    assert len({t.id for t in turns}) == len(turns)
    for turn in turns:
        for citation in turn.citations:
            assert citation.split("#")[0] in documents, (turn.id, citation)


def test_knowledge_turns_include_questions_the_documents_cannot_answer():
    from evals.turns import all_turns

    knowledge = [t for t in all_turns() if "knowledge" in t.agents]
    not_in_documents = [t for t in knowledge if not t.citations]

    assert len(knowledge) >= 8
    assert len(not_in_documents) >= 2


def test_eval_turns_name_only_agents_the_supervisor_has():
    from evals.turns import all_turns
    from perry import supervisor

    for turn in all_turns():
        for agent in turn.agents:
            assert hasattr(supervisor, f"ask_{agent}_agent"), (turn.id, agent)


def test_task_turns_run_against_the_eval_workspace_fake(perry):
    from evals.turns import all_turns
    from evals.workspace import fake_workspace
    from perry import workspace

    turn = next(t for t in all_turns() if t.id == "task-tomorrow")
    tomorrow = fake_workspace().events[0].start.date().isoformat()
    perry.script(
        call("ask_task_agent", request=turn.message),
        call("list_events", first_day=tomorrow, last_day=tomorrow),
        last_message,
        last_message,
        '{"score": 5, "comment": "Lists both events."}',
    )

    with workspace.use(fake_workspace()):
        result = run_eval_turn(turn, perry)

    assert result.passed
    assert "Roadmap review" in result.reply and "Lunch with Priya" in result.reply


EMAIL_SAM = EvalTurn(
    id="task-write-email",
    message="Email sam@example.com that I'll be 10 minutes late.",
    agents=("task",),
    write_actions=("send_email",),
)


def test_write_action_turn_passes_when_it_waits_for_the_confirmation_and_is_approved(
    perry, workspace
):
    perry.script(
        call("ask_task_agent", request=EMAIL_SAM.message),
        call("send_email", to="sam@example.com", subject="Late", body="10 minutes late."),
        last_message,  # after the harness approves: Task Agent reports
        last_message,  # Supervisor passes it on
        '{"score": 5, "comment": "Sent."}',
    )

    result = run_eval_turn(EMAIL_SAM, perry)

    assert result.confirmations == ["send_email"]
    assert result.passed
    assert [e.subject for e in workspace.sent] == ["Late"]


def test_write_action_turn_that_never_asks_for_a_confirmation_fails(perry, workspace):
    perry.script(
        call("ask_task_agent", request=EMAIL_SAM.message),
        "I can't send email.",
        "I can't send email.",
        '{"score": 1, "comment": "Did nothing."}',
    )

    result = run_eval_turn(EMAIL_SAM, perry)

    assert result.routing_ok
    assert result.confirmations == []
    assert not result.passed


def load_eval_datasets() -> None:
    from evals.turns import DATASETS_DIR
    from perry import datasets

    for file in DATASETS_DIR.iterdir():
        datasets.upload(file)


def test_analysis_sql_turns_run_against_the_eval_datasets(perry):
    from evals.turns import all_turns

    load_eval_datasets()
    turn = next(t for t in all_turns() if t.id == "analysis-revenue-per-region")
    perry.script(
        call("ask_analysis_agent", question=turn.message),
        call("run_sql", query="SELECT region, sum(revenue) FROM sales GROUP BY 1 ORDER BY 1"),
        last_message,
        last_message,
        '{"score": 5, "comment": "Correct totals."}',
    )

    result = run_eval_turn(turn, perry)

    assert result.passed
    assert "North | 181300" in result.reply and "West | 220850" in result.reply


BUDGET_BY_DECOMPOSITION = EvalTurn(
    id="q3-budget-decomposed",
    message="How big is my Q3 marketing budget?",
    agents=("knowledge",),
    citations=("budget.md",),
    technique="question_decomposition",
)


def test_turn_found_with_the_expected_technique_matches_the_soft_check(perry):
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    turn = replace(BUDGET_TURN, technique="hybrid_search")
    ask_knowledge_agent_and_echo(perry)
    perry.script('{"score": 5, "comment": "Correct."}')

    result = run_eval_turn(turn, perry)

    assert result.techniques == ["hybrid_search"]
    assert result.technique_ok is True
    assert result.passed


def test_technique_mismatch_is_reported_but_never_fails_the_turn(perry):
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    ask_knowledge_agent_and_echo(perry)  # found with hybrid search
    perry.script('{"score": 5, "comment": "Correct."}')

    result = run_eval_turn(BUDGET_BY_DECOMPOSITION, perry)

    assert result.techniques == ["hybrid_search"]
    assert result.technique_ok is False
    assert result.passed


def test_turn_without_an_expected_technique_has_no_technique_check(perry):
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    ask_knowledge_agent_and_echo(perry)
    perry.script('{"score": 5, "comment": "Correct."}')

    result = run_eval_turn(BUDGET_TURN, perry)

    assert result.technique_ok is None
    assert "technique" not in feedback(result)


def test_technique_mismatch_is_logged_as_feedback_with_the_technique_used(perry):
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    ask_knowledge_agent_and_echo(perry)
    perry.script('{"score": 5, "comment": "Correct."}')

    result = run_eval_turn(BUDGET_BY_DECOMPOSITION, perry)
    scores = feedback(result)

    assert scores["technique"] == (
        0.0,
        "expected question_decomposition, found with hybrid_search",
    )
    assert scores["routing"] == (1.0, None)
    assert scores["quality"] == (1.0, "Correct.")


def test_eval_turns_expect_every_retrieval_technique_by_its_registered_name():
    from evals.turns import all_turns
    from perry.techniques import TECHNIQUES

    expected = [t.technique for t in all_turns() if t.technique]

    assert set(expected) == {t.name for t in TECHNIQUES}


def test_eval_turns_are_neither_benchmark_questions_nor_router_examples():
    """The cascade margin is tuned on the eval Turns: reusing the benchmark would grade
    the benchmark on its own answer key, and reusing router examples would make the
    tuning trivially optimistic."""
    from benchmark.run import QUESTIONS_PATH
    from benchmark.scoring import _question_key, load_questions
    from evals.turns import all_turns
    from perry.techniques import TECHNIQUES

    taken = {_question_key(q.question) for q in load_questions(QUESTIONS_PATH)}
    taken |= {_question_key(e) for t in TECHNIQUES for e in t.examples}

    assert [t.id for t in all_turns() if _question_key(t.message) in taken] == []


def test_analysis_sandbox_turns_read_the_eval_datasets(perry, monkeypatch, tmp_path):
    from evals.turns import all_turns
    from perry import sandbox

    monkeypatch.setenv("PERRY_APP_DATA_DIR", str(tmp_path))
    load_eval_datasets()
    turn = next(t for t in all_turns() if t.id == "analysis-python-chart")
    inputs = []

    def fake_sandbox(code, files):
        inputs.append(files["sales.csv"].decode())
        return sandbox.Result(output="plotted", charts=[b"\x89PNG fake"])

    monkeypatch.setattr(sandbox, "run", fake_sandbox)
    perry.script(
        call("ask_analysis_agent", question=turn.message),
        call("run_python", code="...", inputs={"sales.csv": "SELECT * FROM sales"}),
        "West is highest every month.",
        last_message,
        '{"score": 5, "comment": "Chart shown."}',
    )

    result = run_eval_turn(turn, perry)

    assert result.passed
    assert len(inputs[0].splitlines()) == 37  # header and 36 monthly rows
    assert "[chart: " in result.reply
