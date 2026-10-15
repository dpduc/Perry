"""Turns that need the Task Agent, against the in-memory Workspace fake."""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from fakes import call, last_message

# The Owner's Profile time zone: the Task Agent works out dates in it.
OWNERS_ZONE = ZoneInfo("Europe/London")


def at(day: date, hour: int) -> datetime:
    return datetime.combine(day, time(hour), tzinfo=OWNERS_ZONE)


def test_whats_on_tomorrow_lists_tomorrows_events(perry, workspace):
    from perry.workspace import Event

    today = datetime.now(OWNERS_ZONE).date()
    tomorrow = today + timedelta(days=1)
    workspace.events += [
        Event("Dentist", at(tomorrow, 9), at(tomorrow, 10), location="High Street"),
        Event("Team lunch", at(tomorrow, 12), at(tomorrow, 13)),
        Event("Yoga", at(today, 18), at(today, 19)),
        Event("Flight to Lisbon", at(tomorrow + timedelta(days=1), 7), at(tomorrow + timedelta(days=1), 9)),
    ]
    perry.script(
        call("ask_task_agent", request="What's on my calendar tomorrow?"),  # Supervisor
        call("list_events", first_day=tomorrow.isoformat(), last_day=tomorrow.isoformat()),
        last_message,  # Task Agent reports the events it found
        last_message,  # Supervisor passes them on
    )

    reply = perry.send("What's on tomorrow?")

    assert perry.consulted() == ["ask_task_agent"]
    assert "Dentist" in reply and "High Street" in reply and "Team lunch" in reply
    assert "Yoga" not in reply and "Lisbon" not in reply


def test_what_did_sam_say_about_the_budget_searches_and_reads_mail(perry, workspace):
    from perry.workspace import Email

    workspace.emails += [
        Email(
            "Sam Lee <sam@example.com>",
            "Q3 budget",
            "Hi, the Q3 budget is approved at 40k, but travel is capped at 5k.",
            at(date(2026, 9, 28), 10),
            id="m1",
            email_thread_id="t-budget",
        ),
        Email(
            "Alex Kim <alex@example.com>",
            "Re: Q3 budget",
            "Thanks Sam. Can we move 2k from travel to tooling?",
            at(date(2026, 9, 28), 14),
            id="m2",
            email_thread_id="t-budget",
        ),
        Email(
            "Sam Lee <sam@example.com>",
            "Hiking on Saturday",
            "Shall we take the ridge path?",
            at(date(2026, 9, 30), 9),
            id="m3",
            email_thread_id="t-hike",
        ),
    ]

    def read_what_was_found(messages):
        found = messages[-1].text
        assert "Q3 budget" in found and "Hiking" not in found
        return call("read_email_thread", email_thread_id="t-budget")

    perry.script(
        call("ask_task_agent", request="What did Sam say about the budget?"),  # Supervisor
        call("search_email", query="from:sam budget"),  # Task Agent searches
        read_what_was_found,  # ...sees only the budget email thread, and reads it
        lambda messages: f"Sam said: {messages[-1].text}",  # Task Agent summarises
        last_message,  # Supervisor passes it on
    )

    reply = perry.send("What did Sam say about the budget?")

    assert perry.consulted() == ["ask_task_agent"]
    assert "travel is capped at 5k" in reply
    assert "ridge path" not in reply


def test_expired_google_sign_in_answers_that_google_needs_reconnecting(perry, workspace):
    workspace.disconnect()  # e.g. the refresh token was revoked
    perry.script(
        call("ask_task_agent", request="What's on my calendar tomorrow?"),
        call("list_events", first_day="2026-10-09", last_day="2026-10-09"),
    )

    reply = perry.send("What's on tomorrow?")

    assert reply.startswith("Google needs reconnecting")
    assert perry.history() == []  # a retry after reconnecting won't show the question twice


def test_after_google_needs_reconnecting_the_ui_shows_a_reconnect_button(perry, workspace):
    from perry import app

    assert app.google_button()["value"] == "Google connected"

    workspace.disconnect()
    perry.script(
        call("ask_task_agent", request="Any email from Sam?"),
        call("search_email", query="from:sam"),
    )
    perry.send("Any email from Sam?")

    assert app.google_button()["value"] == "Reconnect Google"
