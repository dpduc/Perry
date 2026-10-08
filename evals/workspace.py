"""The calendar and email the eval Turns see: an in-memory Workspace, never real Google.

Dates are relative to the run, so "tomorrow" always has something on it.
"""

from datetime import date, datetime, time, timedelta

from perry.workspace import Email, Event, InMemoryWorkspace


def _at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute)).astimezone()


def fake_workspace() -> InMemoryWorkspace:
    today = datetime.now().astimezone().date()
    tomorrow = today + timedelta(days=1)
    last_week = today - timedelta(days=6)
    return InMemoryWorkspace(
        events=[
            Event(
                "Roadmap review",
                _at(tomorrow, 10),
                _at(tomorrow, 11),
                location="Room 4B",
                attendees=("Sam Patel",),
                id="e1",
            ),
            Event("Lunch with Priya", _at(tomorrow, 12, 30), _at(tomorrow, 13, 30), id="e2"),
            Event("Yoga", _at(today, 18), _at(today, 19), id="e3"),
        ],
        emails=[
            Email(
                "Sam Patel <sam@example.com>",
                "Q3 budget",
                "Hi, I'd like to move 2,000 dollars from online ads to the Lyon trade fair "
                "stand. Ads are underperforming and the stand needs a bigger screen.",
                _at(last_week, 9),
                to=("me@example.com",),
                id="m1",
                email_thread_id="t-budget",
            ),
            Email(
                "Me <me@example.com>",
                "Re: Q3 budget",
                "Sounds reasonable, let's discuss at the roadmap review.",
                _at(last_week, 11),
                to=("sam@example.com",),
                id="m2",
                email_thread_id="t-budget",
            ),
            Email(
                "Priya Shah <priya@example.com>",
                "Lunch tomorrow?",
                "Shall we try the new noodle place?",
                _at(today, 8),
                to=("me@example.com",),
                id="m3",
                email_thread_id="t-lunch",
            ),
        ],
    )
