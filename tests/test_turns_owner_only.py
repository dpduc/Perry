"""Email and calendar belong to the Owner: other Users' Turns never reach the Task Agent."""

from datetime import datetime

import pytest
from fakes import call, last_message

from perry.workspace import Email


@pytest.fixture
def sam(database):
    from perry import users

    return users.create_user("sam", "sam-password-1", "Sam", "UTC")


@pytest.fixture
def owners_mail(workspace):
    workspace.emails.append(
        Email(
            "Alex Kim <alex@example.com>",
            "Salaries",
            "The salary review is on Friday.",
            datetime(2026, 10, 1, 9).astimezone(),
            id="m1",
            email_thread_id="t1",
        )
    )
    return workspace


def supervisor_sees(model, messages) -> tuple[list[str], str]:
    """The tools and system prompt of the Supervisor's model call."""
    return model.bound_tools, messages[0].text


def test_a_non_owners_supervisor_has_no_task_agent_and_says_email_isnt_theirs(
    perry, sam, owners_mail
):
    perry.user = sam
    seen = {}

    def supervisor_step(messages):
        seen["tools"], seen["prompt"] = supervisor_sees(perry.model, messages)
        return "Email and calendar aren't available to you."

    perry.script(supervisor_step)

    perry.send("Any new email from Alex?")

    assert "ask_task_agent" not in seen["tools"]
    assert "ask_knowledge_agent" in seen["tools"] and "ask_analysis_agent" in seen["tools"]
    assert "ask_task_agent" not in seen["prompt"]
    assert "email and calendar aren't available" in seen["prompt"].lower()


def test_a_non_owner_never_reaches_the_workspace_even_if_the_model_tries(
    perry, sam, owners_mail
):
    perry.user = sam
    perry.script(
        call("ask_task_agent", request="Show my latest email."),  # Supervisor tries anyway
        "Email and calendar aren't available to you.",  # ...and is refused, so it says so
    )

    reply = perry.send("Show my latest email.")

    assert reply == "Email and calendar aren't available to you."
    assert "salary" not in str(perry.model.calls).lower()  # the Owner's mail was never read
    assert perry.model.steps == []


def test_the_owners_turns_still_reach_the_task_agent(perry, owners_mail):
    seen = {}

    def supervisor_step(messages):
        seen["tools"], seen["prompt"] = supervisor_sees(perry.model, messages)
        return call("ask_task_agent", request="Search email for salaries.")

    perry.script(
        supervisor_step,
        call("search_email", query="salary"),  # Task Agent
        last_message,  # Task Agent reports
        last_message,  # Supervisor passes it on
    )

    reply = perry.send("What did Alex say about salaries?")

    assert "ask_task_agent" in seen["tools"]
    assert "aren't available" not in seen["prompt"]
    assert perry.consulted() == ["ask_task_agent"]
    assert "Friday" in reply


def test_a_supervisor_cant_be_built_without_knowing_whose_turn_it_is(database):
    """No default to the Owner's tools: forgetting the User fails, rather than giving
    the Task Agent to whoever's Turn it is."""
    from perry import supervisor

    with pytest.raises(TypeError):
        supervisor.supervisor()
