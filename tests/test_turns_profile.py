"""The Supervisor is given the User's Profile on each Turn: name, about-me and their local date."""

from datetime import datetime
from zoneinfo import ZoneInfo

from fakes import call, last_message


def supervisor_prompt(perry, model_call: int = 0) -> str:
    """The system prompt the Supervisor's model was given on that model call."""
    first = perry.model.calls[model_call][0]
    assert first.type == "system"
    return first.text


def local_today(time_zone: str) -> str:
    return f"{datetime.now(ZoneInfo(time_zone)):%Y-%m-%d}"


def test_the_supervisor_is_given_the_users_profile(perry):
    from perry import users

    perry.user = users.create_user(
        "sam", "sam-password-1", "Sam Student", "Pacific/Kiritimati"
    )
    perry.user = users.update_profile(
        perry.user, "Sam Student", "Pacific/Kiritimati", "A second-year chemistry student."
    )
    perry.script("Hi Sam!")

    perry.send("Hello there")

    prompt = supervisor_prompt(perry)
    assert "Sam Student" in prompt
    assert "A second-year chemistry student." in prompt
    assert "Pacific/Kiritimati" in prompt
    assert local_today("Pacific/Kiritimati") in prompt


def test_relative_dates_use_the_users_time_zone(perry):
    """At any moment these two time zones are on different dates (25 hours apart)."""
    from perry import users

    perry.user = users.create_user("ahead", "ahead-password", "Ahead", "Pacific/Kiritimati")
    perry.script("Hi!")
    perry.send("What's today's date?")
    perry.user = users.create_user("behind", "behind-password", "Behind", "Pacific/Pago_Pago")
    perry.thread_id = None
    perry.script("Hi!")
    perry.send("What's today's date?")

    assert local_today("Pacific/Kiritimati") in supervisor_prompt(perry, 0)
    assert local_today("Pacific/Pago_Pago") in supervisor_prompt(perry, 1)
    assert local_today("Pacific/Pago_Pago") not in supervisor_prompt(perry, 0)


def test_an_empty_about_me_is_fine(perry, owner):
    perry.script("Hi Olive!")

    reply = perry.send("Hello there")

    assert reply == "Hi Olive!"
    prompt = supervisor_prompt(perry)
    assert "Olive Owner" in prompt
    assert "About them" not in prompt


def test_every_model_call_of_the_turn_gets_the_profile(perry):
    """Also the Supervisor's call after an Agent answers, not just its first."""
    perry.script(
        call("ask_knowledge_agent", question="Anything about hiking?"),  # Supervisor
        call("hybrid_search", question="hiking"),  # Knowledge Agent
        last_message,  # Knowledge Agent
        last_message,  # Supervisor
    )

    perry.send("Do my notes mention hiking?")

    assert "Olive Owner" in supervisor_prompt(perry, 0)
    assert "Olive Owner" in supervisor_prompt(perry, 3)
    # The Agents aren't told about the User; the Supervisor gives them exact requests.
    assert "Olive Owner" not in supervisor_prompt(perry, 1)


def test_a_turn_resumed_after_a_confirmation_still_gets_the_profile(perry, workspace):
    from perry.supervisor import Approve

    perry.script(
        call("ask_task_agent", request="Email Sam that I'm running late."),  # Supervisor
        call("send_email", to="sam@example.com", subject="Late", body="Running late."),
    )
    perry.send("Tell Sam I'm running late")
    perry.script(last_message, last_message)  # Task Agent reports, Supervisor passes it on

    perry.resume(Approve())

    assert "Olive Owner" in supervisor_prompt(perry, -1)


# The Agents work out "today" in the User's time zone too. At any moment these two time
# zones are on different dates, so one of them is always not this machine's date.
APART = ("Pacific/Kiritimati", "Pacific/Pago_Pago")


def prompt_text(messages) -> str:
    return "\n".join(message.text for message in messages)


def test_the_task_agent_knows_today_in_the_users_time_zone(perry, owner, workspace):
    from perry import users

    for zone in APART:
        perry.user = users.update_profile(owner, "Olive Owner", zone)
        perry.thread_id = None
        perry.script(
            call("ask_task_agent", request="What's on my calendar today?"),  # Supervisor
            "Nothing is on today.",  # Task Agent
            last_message,  # Supervisor
        )

        perry.send("What's on today?")

        task_agents_prompt = prompt_text(perry.model.calls[-2])
        assert local_today(zone) in task_agents_prompt, zone


def test_self_query_knows_today_in_the_users_time_zone(perry):
    from perry import users

    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    for zone in APART:
        perry.user = users.update_profile(perry.user, "Olive Owner", zone)
        perry.thread_id = None
        perry.script(
            call("ask_knowledge_agent", question="What is the marketing budget?"),
            call("self_query", question="marketing budget"),  # Knowledge Agent
            '{"query": "marketing budget"}',  # Self-query's model call: no filter
            last_message,  # Knowledge Agent
            last_message,  # Supervisor
        )

        perry.send("What's my marketing budget?", technique="self_query")

        self_query_prompt = prompt_text(perry.model.calls[-3])
        assert f"Today is {local_today(zone)}" in self_query_prompt, zone
