"""Retry a failed step once, then report it.

A step is one tool call made by the Supervisor or an Agent (e.g. asking the
Knowledge Agent). When it fails for a reason that may pass (rate limit, timeout,
provider error, see `models.is_transient`), it is tried once more after a short
backoff. If that fails too, the step's result becomes a failure report, so the
Turn goes on and the answer can say what didn't work instead of making it up.
Other errors are not retried and end the Turn as before: bugs, and deliberate
stops such as `usage.TurnTokenCeilingReached`, `workspace.ReconnectNeeded` and
`privacy.Refused`.

Excluding a tool from retries
-----------------------------
Mark the tool with `never_retry` where it is defined:

    @never_retry
    @tool
    def send_email(...): ...

A marked tool runs at most once per call; a transient failure is reported
straight away. Use it for every Write Action (a Confirmation covers a single
attempt) and for any Agent tool whose run may include a Write Action, since
retrying it would repeat the Agent's run. The mark lives on the tool, so the
rule holds wherever the tool is used with `RetryFailedSteps`.
"""

import os
import time

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphBubbleUp

from perry import models

_NEVER_RETRY = "never_retry"

# Wait before the retry. Tests set it to 0.
RETRY_DELAY_SECONDS = float(os.getenv("PERRY_RETRY_DELAY_SECONDS", "2"))


def never_retry(tool: BaseTool) -> BaseTool:
    """Mark `tool` so a failure is reported at once, never retried."""
    tool.metadata = {**(tool.metadata or {}), _NEVER_RETRY: True}
    return tool


def is_retried(tool: BaseTool) -> bool:
    return not (tool.metadata or {}).get(_NEVER_RETRY, False)


class RetryFailedSteps(AgentMiddleware):
    """Agent middleware: retry a transiently failed tool call once, then report it."""

    def wrap_tool_call(self, request, handler):
        tool = request.tool
        attempts = 2 if tool is None or is_retried(tool) else 1
        for attempt in range(1, attempts + 1):
            try:
                return handler(request)
            except GraphBubbleUp:  # interrupts (e.g. a Confirmation) are not failures
                raise
            except Exception as error:
                if not models.is_transient(error):
                    raise
                if attempt == attempts:
                    return _failure_report(request.tool_call, error, attempts)
                time.sleep(RETRY_DELAY_SECONDS)
        raise AssertionError("unreachable")


def _failure_report(tool_call: dict, error: Exception, attempts: int) -> ToolMessage:
    tried = "twice" if attempts == 2 else "once"
    args = ", ".join(f"{k}={v!r}" for k, v in tool_call["args"].items())
    return ToolMessage(
        f"FAILED: the step {tool_call['name']}({args}) failed {tried} "
        f"({type(error).__name__}: {error}). It produced no result. Tell the user this "
        "step failed; do not guess what it would have returned.",
        tool_call_id=tool_call["id"],
        name=tool_call["name"],
        status="error",
    )
