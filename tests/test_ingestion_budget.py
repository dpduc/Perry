"""Ingestion counts toward the daily cap: it is estimated up front and recorded as it runs."""

import pytest

BUDGET_TEXT = "The marketing budget for Q3 is 40,000 dollars."  # 46 characters: 12 tokens


@pytest.fixture
def documents(tmp_path):
    def write(name: str, text: str):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path

    return write


def test_embedding_calls_during_ingestion_are_recorded_in_the_usage_ledger(
    perry, prices, documents
):
    from perry import ingest, usage

    prices({"fake-embedding": {"input": 1000.0}})

    ingest.ingest_files([documents("budget.md", BUDGET_TEXT)])

    assert usage.today().tokens == 12
    assert usage.today().cost_usd == pytest.approx(12 * 1000e-6)


def test_estimate_shows_tokens_and_cost_of_what_ingestion_will_embed(perry, prices, documents):
    from perry import ingest, usage

    prices({"fake-embedding": {"input": 1000.0}})
    budget = documents("budget.md", BUDGET_TEXT)
    ingest.ingest_files([budget])
    spent_before = usage.today()
    hiking = documents("hiking.md", "Trail notes.")  # 12 characters: 3 tokens

    estimate = ingest.estimate_ingestion([budget, hiking])  # budget.md is unchanged

    assert estimate.tokens == 3
    assert estimate.cost_usd == pytest.approx(0.003)
    assert estimate.describe() == (
        "2 files: about 3 tokens, $0.0030 (1 unchanged, skipped)."
    )
    ingest.ingest_files([budget, hiking])
    assert usage.today().tokens - spent_before.tokens == estimate.tokens


def test_ingestion_that_would_exceed_the_remaining_daily_budget_is_refused_before_any_call(
    perry, prices, documents, monkeypatch
):
    from perry import ingest, usage

    monkeypatch.setenv("PERRY_DAILY_CAP_USD", "0.01")
    prices({"fake-embedding": {"input": 1000.0}})
    hiking = documents("hiking.md", "Trail notes.")  # $0.003
    budget = documents("budget.md", BUDGET_TEXT)  # $0.012
    ingest.ingest_files([hiking])

    with pytest.raises(ingest.OverBudget) as refused:
        ingest.ingest_files([budget])

    assert str(refused.value) == (
        "Ingesting these files would cost about $0.0120, "
        "but only $0.0070 of today's $0.01 budget is left."
    )
    assert usage.today().tokens == 3  # only hiking.md's embedding
    assert ingest.estimate_ingestion([budget]).unchanged == 0  # budget.md wasn't stored


def test_a_file_that_fails_is_reported_and_the_rest_are_still_ingested(perry, documents):
    from perry import ingest

    broken = documents("broken.pdf", "not really a PDF")
    budget = documents("budget.md", BUDGET_TEXT)

    assert ingest.estimate_ingestion([broken, budget]).tokens == 12
    results = ingest.ingest_files([broken, budget])

    assert results[1].describe() == "budget.md: 1 chunks"
    assert results[0].describe().startswith("broken.pdf: failed (")


def test_cli_shows_the_estimate_and_ingests_only_after_the_user_says_yes(
    perry, prices, documents, capsys
):
    from perry import ingest, usage

    prices({"fake-embedding": {"input": 1000.0}})
    budget = documents("budget.md", BUDGET_TEXT)
    questions = []

    def answer(reply):
        def ask(question):
            questions.append(question)
            return reply

        return ask

    ingest.main([str(budget.parent)], ask=answer("n"))
    assert questions == ["Ingest 1 file: about 12 tokens, $0.0120? [y/N] "]
    assert usage.today().tokens == 0
    assert "Nothing ingested." in capsys.readouterr().out

    ingest.main([str(budget.parent)], ask=answer("y"))
    assert usage.today().tokens == 12
    assert "budget.md: 1 chunks" in capsys.readouterr().out


def test_cli_yes_flag_skips_the_question_but_not_the_budget(
    perry, prices, documents, capsys, monkeypatch
):
    from perry import ingest, usage

    monkeypatch.setenv("PERRY_DAILY_CAP_USD", "0.01")
    prices({"fake-embedding": {"input": 1000.0}})
    budget = documents("budget.md", BUDGET_TEXT)

    def never(question):
        raise AssertionError("--yes must not ask")

    with pytest.raises(SystemExit) as stopped:
        ingest.main([str(budget), "--yes"], ask=never)

    assert "would cost about $0.0120" in str(stopped.value)
    assert usage.today().tokens == 0
