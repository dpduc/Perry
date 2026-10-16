"""The calendar and email the eval Turns see: an in-memory Workspace, never real Google.

Dates are relative to the run, so "tomorrow" always has something on it. They are in
the eval Owner's time zone, not this machine's: Perry reads "tomorrow" in the Owner's
time zone, and the two dates differ for part of every day.
"""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from perry.workspace import Email, Event, InMemoryWorkspace

# The eval Owner's Profile time zone.
EVAL_TIME_ZONE = "UTC"
_ZONE = ZoneInfo(EVAL_TIME_ZONE)


def _at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=_ZONE)


def fake_workspace() -> InMemoryWorkspace:
    today = datetime.now(_ZONE).date()
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
