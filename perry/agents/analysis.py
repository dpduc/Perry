"""Analysis Agent: answers questions about Datasets with read-only SQL, and with Python
run in the Sandbox when SQL isn't enough.

Interface: question in, findings out, always followed by the SQL and Python that
produced them, so the user can check the logic. Charts travel as chart markers.
"""

import csv
import io
import os

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage
from langchain_core.tools import ToolException, tool

from perry import charts, datasets, models, sandbox

SYSTEM_PROMPT = """You are Perry's Analysis Agent. You answer questions about the user's \
Datasets (tables uploaded from CSV or spreadsheet files) by querying them with SQL.

- Call list_datasets first unless you already know the Dataset and its columns.
- Write one PostgreSQL SELECT statement per run_sql call. Refer to Datasets by their name, \
which is the table name. Let SQL do the computing (sums, averages, grouping, filtering).
- Your access is read-only. If a query fails, read the error, fix the query and try again \
(at most 3 attempts). If it still fails, say so plainly with the error.
- Results are cut at {row_limit} rows; aggregate rather than listing many rows.
- When SQL isn't enough (statistics, forecasts, reshaping, charts), call run_python. \
Give it `inputs`: CSV file names mapped to SELECT statements; each result is saved as \
that file in the script's working directory. The script has pandas, numpy, scipy and \
matplotlib, but no network and no other files. Print the results you need and save each \
chart with plt.savefig("<name>.png"); charts are shown to the user automatically.
- Answer only from query and script results. Never invent numbers.
- Be concise: report the figures, not a polished essay."""

# Python inputs can be larger than the rows the model reads back from run_sql.
INPUT_ROW_LIMIT = int(os.getenv("PERRY_SANDBOX_INPUT_ROW_LIMIT", "100000"))


@tool
def list_datasets() -> str:
    """List the user's Datasets with their columns and types."""
    found = datasets.list_datasets()
    if not found:
        return "The user has not uploaded any Datasets yet."
    return "\n".join(d.describe() for d in found)


@tool(response_format="content")
def run_sql(query: str) -> str:
    """Run one read-only SQL SELECT statement over the Datasets and return the rows."""
    try:
        result = datasets.run_query(query)
    except datasets.QueryError as error:
        raise ToolException(str(error)) from error
    return format_table(result)


run_sql.handle_tool_error = True  # the Agent sees the error and can fix its query


@tool(response_format="content")
def run_python(code: str, inputs: dict[str, str] | None = None) -> str:
    """Run a Python script in an isolated sandbox. `inputs` maps CSV file names to
    read-only SQL SELECT statements whose results the script can read. Returns what the
    script printed, any CSV tables it saved, and its charts (PNG files it saved)."""
    files, notes = {}, []
    for name, query in (inputs or {}).items():
        try:
            result = datasets.run_query(query, row_limit=INPUT_ROW_LIMIT)
        except datasets.QueryError as error:
            raise ToolException(f"Input {name}: {error}") from error
        files[name] = _csv(result)
        if result.truncated:
            notes.append(f"{name} was cut at {INPUT_ROW_LIMIT} rows.")
    try:
        outcome = sandbox.run(code, files)
    except sandbox.SandboxUnavailable as error:
        raise ToolException(f"Python can't run right now: {error}") from error
    if not outcome.ok:
        printed = f"\nIt printed before failing:\n{outcome.output}" if outcome.output else ""
        raise ToolException(outcome.error + printed)
    parts = [*notes, outcome.output.strip() or "(the script printed nothing)"]
    parts += [f"{name}:\n{table.strip()}" for name, table in outcome.tables.items()]
    if outcome.charts:
        parts.append("Charts (shown to the user): " + " ".join(map(charts.save, outcome.charts)))
    return "\n\n".join(parts)


run_python.handle_tool_error = True  # the Agent sees the error and can fix its script


def _csv(result: datasets.QueryResult) -> bytes:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(result.columns)
    writer.writerows(result.rows)
    return out.getvalue().encode()


def format_table(result: datasets.QueryResult) -> str:
    if not result.columns:
        return "The query returned no columns."
    lines = [" | ".join(result.columns)]
    lines += [" | ".join("" if v is None else str(v) for v in row) for row in result.rows]
    if not result.rows:
        lines.append("(no rows)")
    if result.truncated:
        lines.append(f"(only the first {len(result.rows)} rows are shown)")
    return "\n".join(lines)


def analysis_agent():
    # Built per call (cheap) so it always uses the current model provider.
    return create_agent(
        models.chat_model(),
        tools=[list_datasets, run_sql, run_python],
        system_prompt=SYSTEM_PROMPT.format(row_limit=datasets.ROW_LIMIT),
        name="analysis_agent",
    )


def ask(question: str) -> str:
    result = analysis_agent().invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]
    # Code, not the model, keeps every chart made in this run in the findings.
    findings = charts.keep(messages[-1].text, [m.text for m in messages if m.type == "tool"])
    shown = [findings]
    queries = [args["query"] for args in calls_that_ran(messages, "run_sql")]
    if queries:
        label = "Query used" if len(queries) == 1 else "Queries used"
        shown.append(f"{label}:\n" + "\n\n".join(f"```sql\n{q}\n```" for q in queries))
    for args in calls_that_ran(messages, "run_python"):
        reads = "".join(f"\n- {name}: `{q}`" for name, q in (args.get("inputs") or {}).items())
        reading = f", reading:{reads}" if reads else ""
        shown.append(f"Python used{reading}\n```python\n{args['code']}\n```")
    return "\n\n".join(shown)


def calls_that_ran(messages: list[BaseMessage], tool_name: str) -> list[dict]:
    """The arguments of every successful call to the tool, in order."""
    succeeded = {m.tool_call_id for m in messages if m.type == "tool" and m.status != "error"}
    return [
        call["args"]
        for m in messages
        if m.type == "ai"
        for call in m.tool_calls
        if call["name"] == tool_name and call["id"] in succeeded
    ]
