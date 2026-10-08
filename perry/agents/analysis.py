"""Analysis Agent: answers questions about Datasets with read-only SQL.

Interface: question in, findings out, always followed by the SQL that produced them,
so the user can check the logic.
"""

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage
from langchain_core.tools import ToolException, tool

from perry import datasets, models, retries

SYSTEM_PROMPT = """You are Perry's Analysis Agent. You answer questions about the user's \
Datasets (tables uploaded from CSV or spreadsheet files) by querying them with SQL.

- Call list_datasets first unless you already know the Dataset and its columns.
- Write one PostgreSQL SELECT statement per run_sql call. Refer to Datasets by their name, \
which is the table name. Let SQL do the computing (sums, averages, grouping, filtering).
- Your access is read-only. If a query fails, read the error, fix the query and try again \
(at most 3 attempts). If it still fails, say so plainly with the error.
- Results are cut at {row_limit} rows; aggregate rather than listing many rows.
- Answer only from query results. Never invent numbers.
- Be concise: report the figures, not a polished essay."""


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
        tools=[list_datasets, run_sql],
        middleware=[retries.RetryFailedSteps()],
        system_prompt=SYSTEM_PROMPT.format(row_limit=datasets.ROW_LIMIT),
        name="analysis_agent",
    )


def ask(question: str) -> str:
    result = analysis_agent().invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]
    findings = messages[-1].text
    queries = queries_that_ran(messages)
    if not queries:
        return findings
    label = "Query used" if len(queries) == 1 else "Queries used"
    shown = "\n\n".join(f"```sql\n{q}\n```" for q in queries)
    return f"{findings}\n\n{label}:\n{shown}"


def queries_that_ran(messages: list[BaseMessage]) -> list[str]:
    """The SQL of every run_sql call that succeeded, in order."""
    succeeded = {m.tool_call_id for m in messages if m.type == "tool" and m.status != "error"}
    return [
        call["args"]["query"]
        for m in messages
        if m.type == "ai"
        for call in m.tool_calls
        if call["name"] == "run_sql" and call["id"] in succeeded
    ]
