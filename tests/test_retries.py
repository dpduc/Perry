"""Tools marked `never_retry` (e.g. Write Actions) run at most once per call."""

import pytest
from fakes import ScriptedChatModel, call
from langchain.agents import create_agent
from langchain_core.tools import tool

from perry import models
from perry.retries import RetryFailedSteps, never_retry


def agent_with(tool_):
    model = ScriptedChatModel()
    model.steps.extend([call(tool_.name), lambda messages: messages[-1].text])
    return create_agent(model, tools=[tool_], middleware=[RetryFailedSteps()])


def flaky(name: str):
    """A tool that fails transiently on its first run, then works. Records each run."""
    runs = []

    def run() -> str:
        runs.append(1)
        if len(runs) == 1:
            raise TimeoutError("provider timed out")
        return "done"

    return tool(name, run, description="Does a thing."), runs


def ask(agent) -> str:
    return agent.invoke({"messages": [{"role": "user", "content": "go"}]})["messages"][-1].text


def test_unmarked_tool_is_retried_after_a_transient_failure():
    lookup, runs = flaky("lookup")

    assert ask(agent_with(lookup)) == "done"
    assert len(runs) == 2


def test_tool_marked_never_retry_runs_once_and_its_failure_is_reported():
    send_email, runs = flaky("send_email")
    never_retry(send_email)

    reply = ask(agent_with(send_email))

    assert len(runs) == 1
    assert "send_email() failed once" in reply
    # It may have done something before failing, so nothing is claimed about its effect.
    assert "no result" not in reply
    assert "outcome is unknown" in reply
    assert "may already have been performed" in reply
    assert "check" in reply


def test_retried_tools_failure_says_it_produced_no_result():
    def run() -> str:
        raise TimeoutError("provider timed out")

    reply = ask(agent_with(tool("lookup", run, description="Does a thing.")))

    assert "lookup() failed twice" in reply
    assert "produced no result" in reply


def test_never_retry_works_as_a_decorator_on_a_tool():
    @never_retry
    @tool
    def delete_event(event_id: str) -> str:
        """Delete a calendar event."""
        return "deleted"

    assert delete_event.invoke({"event_id": "e1"}) == "deleted"


def ceiling_reached():
    from perry import usage

    return usage.TurnTokenCeilingReached(1000)


def google_needs_reconnecting():
    from perry import workspace

    return workspace.expired()


def privacy_refusal():
    from perry import privacy

    return privacy.Refused("The Task Agent is off.")


@pytest.mark.parametrize(
    "stop", [ceiling_reached, google_needs_reconnecting, privacy_refusal]
)
def test_deliberate_stops_end_the_turn_instead_of_being_retried_or_reported(stop):
    runs = []

    def run() -> str:
        runs.append(1)
        raise stop()

    lookup = tool("lookup", run, description="Does a thing.")

    with pytest.raises(type(stop())):
        ask(agent_with(lookup))
    assert len(runs) == 1


def test_only_transient_errors_count_as_failures_worth_retrying():
    assert models.is_transient(models.RateLimitError("429"))
    assert models.is_transient(TimeoutError())
    assert not models.is_transient(ValueError("bad argument"))
