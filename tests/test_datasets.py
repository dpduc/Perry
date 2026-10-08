"""Datasets against a real Postgres: upload, replacement, and the read-only query role."""

import pandas as pd
import pytest

from perry import datasets


def write_csv(folder, name: str, text: str):
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def test_uploading_a_csv_creates_one_dataset_with_its_columns(database, tmp_path):
    path = write_csv(tmp_path, "Sales Q3.csv", "Region,Amount\nNorth,100\nSouth,250\n")

    datasets.upload(path)

    assert [(d.name, d.columns) for d in datasets.list_datasets()] == [
        ("sales_q3", ["region", "amount"])
    ]
    result = datasets.run_query("SELECT region, amount FROM sales_q3 ORDER BY amount DESC")
    assert result.rows == [("South", 250), ("North", 100)]


def test_uploading_an_xlsx_creates_one_dataset_per_sheet(database, tmp_path):
    path = tmp_path / "budget.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"Team": ["Ops"], "Spend": [1.5]}).to_excel(
            writer, sheet_name="2025", index=False
        )
        pd.DataFrame({"Month": ["Jan"], "Hires": [3]}).to_excel(
            writer, sheet_name="Hiring", index=False
        )

    datasets.upload(path)

    assert [(d.name, d.columns, d.origin) for d in datasets.list_datasets()] == [
        ("budget_2025", ["team", "spend"], "budget.xlsx"),
        ("budget_hiring", ["month", "hires"], "budget.xlsx"),
    ]


def test_uploading_a_file_with_an_existing_dataset_name_replaces_that_dataset(database, tmp_path):
    datasets.upload(write_csv(tmp_path, "sales.csv", "region,amount\nNorth,100\n"))

    results = datasets.upload(write_csv(tmp_path, "sales.csv", "region,units\nEast,7\nWest,9\n"))

    assert [r.replaced for r in results] == [True]
    assert [(d.name, d.columns) for d in datasets.list_datasets()] == [
        ("sales", ["region", "units"])
    ]
    assert datasets.run_query("SELECT count(*) FROM sales").rows == [(2,)]


FORBIDDEN = [
    # Writes, anywhere.
    "INSERT INTO analytics.sales VALUES ('East', 1)",
    "UPDATE analytics.sales SET amount = 0",
    "DELETE FROM analytics.sales",
    "DROP TABLE analytics.sales",
    "CREATE TABLE analytics.extra (x int)",
    "CREATE TABLE public.extra (x int)",
    "CREATE TEMP TABLE scratch (x int)",
    "INSERT INTO public.threads (id, title) VALUES ('x', 'y')",
    "TRUNCATE public.documents",
    # Reads outside the analytics schema: Documents, Chunks, Threads, checkpoints.
    "SELECT * FROM public.documents",
    "SELECT * FROM public.chunks",
    "SELECT * FROM public.threads",
    "SELECT * FROM public.checkpoints",
    "SELECT * FROM public.checkpoint_writes",
    "SELECT * FROM public.datasets",
    # Escalation.
    "SET ROLE perry",
]


@pytest.mark.parametrize("statement", FORBIDDEN)
def test_the_analysis_role_cannot_write_anywhere_or_read_outside_datasets(
    perry, tmp_path, statement
):
    datasets.upload(write_csv(tmp_path, "sales.csv", "region,amount\nNorth,100\n"))
    perry.script("Hi!")
    perry.send("Hello")  # so a Thread, its checkpoints and a Document row exist
    perry.add_document("notes.md", "Private notes.")

    with pytest.raises(datasets.QueryError, match="failed"):
        datasets.run_query(statement)

    assert datasets.run_query("SELECT region, amount FROM sales").rows == [("North", 100)]


def test_a_query_over_the_timeout_is_stopped_with_a_clear_message(database):
    with pytest.raises(datasets.QueryError, match="took longer than 0.2 seconds"):
        datasets.run_query("SELECT pg_sleep(5)", timeout_ms=200)


def test_query_results_are_cut_at_the_row_limit(database):
    result = datasets.run_query("SELECT generate_series(1, 10) AS n", row_limit=3)

    assert result.columns == ["n"]
    assert result.rows == [(1,), (2,), (3,)]
    assert result.truncated


def test_a_failing_query_names_the_problem(database):
    with pytest.raises(datasets.QueryError, match='The query failed: relation "nope"'):
        datasets.run_query("SELECT * FROM nope")
