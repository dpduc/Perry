"""Turns that involve the Analysis Agent, with a scripted fake chat model and real Datasets."""

from pathlib import Path

from fakes import call, last_message

from perry import datasets

TOTALS_SQL = "SELECT region, sum(amount) AS total FROM sales GROUP BY region ORDER BY region"


def add_sales_dataset(tmp_path) -> None:
    path = tmp_path / "sales.csv"
    path.write_text("region,amount\nNorth,100\nSouth,250\nNorth,50\n", encoding="utf-8")
    datasets.upload(path)


def test_data_question_is_answered_by_the_analysis_agent_showing_the_query_used(perry, tmp_path):
    add_sales_dataset(tmp_path)
    perry.script(
        call("ask_analysis_agent", question="What are total sales per region?"),  # Supervisor
        call("run_sql", query=TOTALS_SQL),  # Analysis Agent
        lambda messages: f"Totals: {messages[-1].text}",  # Analysis Agent reports
        last_message,  # Supervisor passes the findings on
    )

    reply = perry.send("What were total sales per region?")

    assert perry.consulted() == ["ask_analysis_agent"]
    assert "North | 150" in reply
    assert "South | 250" in reply
    assert TOTALS_SQL in reply


def test_the_analysis_agent_sees_which_datasets_exist_and_their_columns(perry, tmp_path):
    add_sales_dataset(tmp_path)
    perry.script(
        call("ask_analysis_agent", question="Which Datasets do I have?"),
        call("list_datasets"),
        last_message,
        last_message,
    )

    reply = perry.send("Which Datasets do I have?")

    assert "sales" in reply
    assert "region (text)" in reply
    assert "amount (bigint)" in reply


def test_a_failed_query_gives_a_clear_message_instead_of_crashing(perry, tmp_path):
    add_sales_dataset(tmp_path)
    perry.script(
        call("ask_analysis_agent", question="Total sales?"),
        call("run_sql", query="DELETE FROM sales"),
        last_message,  # Analysis Agent reports the failure
        last_message,
    )

    reply = perry.send("Delete my sales data and tell me the total")

    assert reply.startswith("The query failed: ")
    assert datasets.run_query("SELECT count(*) FROM sales").rows == [(3,)]


FAKE_PNG = b"\x89PNG\r\n\x1a\nfake chart"

CHART_CODE = """import pandas as pd
sales = pd.read_csv("sales.csv")
sales.groupby("region")["amount"].sum().plot.bar().figure.savefig("totals.png")"""


def test_a_computation_question_runs_python_and_the_answer_shows_the_chart(
    perry, tmp_path, monkeypatch
):
    from perry import charts, sandbox

    monkeypatch.setenv("PERRY_APP_DATA_DIR", str(tmp_path / "app-data"))
    add_sales_dataset(tmp_path)
    ran = []

    def fake_sandbox(code, files):
        ran.append((code, files))
        return sandbox.Result(output="North 150\nSouth 250\n", charts=[FAKE_PNG])

    monkeypatch.setattr(sandbox, "run", fake_sandbox)
    perry.script(
        call("ask_analysis_agent", question="Chart total sales per region"),  # Supervisor
        call("run_python", code=CHART_CODE, inputs={"sales.csv": "SELECT * FROM sales"}),
        "Totals: North 150, South 250.",  # Analysis Agent, not mentioning the chart
        "North sold 150 and South 250.",  # Supervisor, not mentioning it either
    )

    reply = perry.send("Plot my total sales per region")

    assert perry.consulted() == ["ask_analysis_agent"]
    code, files = ran[0]
    assert code == CHART_CODE
    assert files["sales.csv"].decode().splitlines() == [
        "region,amount",
        "North,100",
        "South,250",
        "North,50",
    ]
    shown = charts.chat_messages(reply)
    assert shown[-1] == {"role": "assistant", "content": "North sold 150 and South 250."}
    chart_files = [m["content"]["path"] for m in shown[:-1]]
    assert len(chart_files) == 1
    assert Path(chart_files[0]).read_bytes() == FAKE_PNG
    # Reopening the Thread shows the chart again.
    assert charts.chat_messages(perry.history()[-1]["content"]) == shown


def test_a_failing_script_is_reported_so_the_user_knows_to_rephrase(perry, tmp_path, monkeypatch):
    from perry import sandbox

    add_sales_dataset(tmp_path)
    monkeypatch.setattr(
        sandbox,
        "run",
        lambda code, files: sandbox.Result(output="", error="The script failed:\nKeyError: 'x'"),
    )
    perry.script(
        call("ask_analysis_agent", question="Forecast sales"),
        call("run_python", code="print(1)", inputs={"sales.csv": "SELECT * FROM sales"}),
        last_message,
        last_message,
    )

    reply = perry.send("Forecast my sales")

    assert "KeyError: 'x'" in reply
