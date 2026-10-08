"""Turns that involve the Analysis Agent, with a scripted fake chat model and real Datasets."""

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
