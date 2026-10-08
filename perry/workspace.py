"""The Workspace: your calendar and email, as the Task Agent sees them.

The Task Agent depends only on the `Workspace` interface below. `google_workspace`
implements it on Google Calendar and Gmail; `InMemoryWorkspace` is the fake used by
tests and evals. The interface has Read Actions (look only) and Write Actions (send,
reply, create, move, delete). The Task Agent only performs a Write Action after a
Confirmation; the Workspace itself just does what it is told.
"""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Literal, Protocol

Status = Literal["connected", "not connected", "needs reconnecting"]
# Declared when connecting: a test account holds only fake data; a real one is yours.
# See perry.privacy for what Perry refuses to do with a real account.
AccountKind = Literal["test", "real"]


class ReconnectNeeded(Exception):
    """The Workspace can't be reached until you connect (or reconnect) Google.

    The message is written for the user. A Turn that hits this stops and shows it.
    """


def not_connected() -> ReconnectNeeded:
    return ReconnectNeeded(
        "Google isn't connected yet. Use the Connect Google button in the sidebar, then ask again."
    )


def expired() -> ReconnectNeeded:
    return ReconnectNeeded(
        "Google needs reconnecting: the sign-in expired or was revoked. "
        "Use the Reconnect Google button in the sidebar, then ask again."
    )


@dataclass(frozen=True)
class Event:
    title: str
    start: datetime
    end: datetime
    location: str = ""
    attendees: tuple[str, ...] = ()
    id: str = ""


@dataclass(frozen=True)
class Email:
    sender: str
    subject: str
    body: str
    sent_at: datetime
    to: tuple[str, ...] = ()
    id: str = ""
    # The email thread (Gmail's grouping of replies) this email belongs to.
    # Not to be confused with a Perry Thread.
    email_thread_id: str = ""


class Workspace(Protocol):
    def status(self) -> Status: ...

    def account_kind(self) -> AccountKind | None:
        """The kind of the connected account, None if nothing is connected."""
        ...

    def list_events(self, start: datetime, end: datetime) -> list[Event]:
        """Events overlapping [start, end), earliest first."""
        ...

    def search_email(self, query: str, limit: int = 10) -> list[Email]:
        """Emails matching a Gmail-style query (e.g. `from:sam budget`), newest first."""
        ...

    def read_email_thread(self, email_thread_id: str) -> list[Email]:
        """Every email in one email thread, oldest first, with full bodies."""
        ...

    def get_event(self, event_id: str) -> Event | None:
        """One event by id, None if there is no such event."""
        ...

    # Write Actions: only ever called after a Confirmation.

    def send_email(self, to: tuple[str, ...], subject: str, body: str) -> Email:
        """Send a new email; returns it as sent."""
        ...

    def reply_email(self, email_thread_id: str, body: str) -> Email:
        """Reply to the latest email in an email thread; returns the reply as sent."""
        ...

    def create_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        location: str = "",
        attendees: tuple[str, ...] = (),
    ) -> Event:
        """Add an event to the calendar (attendees are invited); returns it."""
        ...

    def move_event(self, event_id: str, start: datetime, end: datetime) -> Event:
        """Give an event a new time; returns it as moved."""
        ...

    def delete_event(self, event_id: str) -> None:
        """Remove an event from the calendar."""
        ...


class NotFound(LookupError):
    """A Write Action named an event or email thread that doesn't exist."""


@dataclass
class InMemoryWorkspace:
    """A Workspace held in memory, for tests and evals. Never touches Google.

    `search_email` understands `from:`, `to:` and plain words (all must match).
    `disconnect()` makes every call fail the way an expired sign-in does.
    Emails sent or replied by Write Actions land in `emails` and, in order, in `sent`.
    """

    events: list[Event] = field(default_factory=list)
    emails: list[Email] = field(default_factory=list)
    sent: list[Email] = field(default_factory=list)
    me: str = "me@example.com"
    account: AccountKind = "test"
    _status: Status = "connected"

    def disconnect(self, status: Status = "needs reconnecting") -> None:
        self._status = status

    def status(self) -> Status:
        return self._status

    def account_kind(self) -> AccountKind | None:
        return None if self._status == "not connected" else self.account

    def _check(self) -> None:
        if self._status == "not connected":
            raise not_connected()
        if self._status == "needs reconnecting":
            raise expired()

    def list_events(self, start: datetime, end: datetime) -> list[Event]:
        self._check()
        hits = [e for e in self.events if e.start < end and e.end > start]
        return sorted(hits, key=lambda e: e.start)

    def search_email(self, query: str, limit: int = 10) -> list[Email]:
        self._check()
        hits = [e for e in self.emails if _matches(e, query)]
        return sorted(hits, key=lambda e: e.sent_at, reverse=True)[:limit]

    def read_email_thread(self, email_thread_id: str) -> list[Email]:
        self._check()
        thread = [e for e in self.emails if (e.email_thread_id or e.id) == email_thread_id]
        return sorted(thread, key=lambda e: e.sent_at)

    def get_event(self, event_id: str) -> Event | None:
        self._check()
        return next((e for e in self.events if e.id == event_id), None)

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}{len(self.events) + len(self.emails) + 1}"

    def _send(self, email: Email) -> Email:
        self.emails.append(email)
        self.sent.append(email)
        return email

    def send_email(self, to: tuple[str, ...], subject: str, body: str) -> Email:
        self._check()
        email_id = self._new_id("sent-")
        return self._send(
            Email(self.me, subject, body, datetime.now().astimezone(), tuple(to), email_id, email_id)
        )

    def reply_email(self, email_thread_id: str, body: str) -> Email:
        self._check()
        thread = self.read_email_thread(email_thread_id)
        if not thread:
            raise NotFound(f"No email thread {email_thread_id!r}.")
        last = thread[-1]
        subject = last.subject if last.subject.lower().startswith("re:") else f"Re: {last.subject}"
        to = last.to if self.me in last.sender else (last.sender,)
        return self._send(
            Email(
                self.me,
                subject,
                body,
                datetime.now().astimezone(),
                to,
                self._new_id("sent-"),
                email_thread_id,
            )
        )

    def create_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        location: str = "",
        attendees: tuple[str, ...] = (),
    ) -> Event:
        self._check()
        event = Event(title, start, end, location, tuple(attendees), self._new_id("event-"))
        self.events.append(event)
        return event

    def move_event(self, event_id: str, start: datetime, end: datetime) -> Event:
        event = self._event(event_id)
        moved = replace(event, start=start, end=end)
        self.events[self.events.index(event)] = moved
        return moved

    def delete_event(self, event_id: str) -> None:
        self.events.remove(self._event(event_id))

    def _event(self, event_id: str) -> Event:
        event = self.get_event(event_id)
        if event is None:
            raise NotFound(f"No event {event_id!r}.")
        return event


def _matches(email: Email, query: str) -> bool:
    for term in query.lower().split():
        if term.startswith("from:"):
            ok = term[5:] in email.sender.lower()
        elif term.startswith("to:"):
            ok = any(term[3:] in t.lower() for t in email.to)
        else:
            words = re.findall(r"\w+", f"{email.subject} {email.body}".lower())
            ok = term.strip("\"'") in words
        if not ok:
            return False
    return True


_override: Workspace | None = None


def current() -> Workspace:
    """The Workspace Turns use: the connected Google account, unless overridden."""
    if _override is not None:
        return _override
    from perry.google_workspace import GoogleWorkspace

    return GoogleWorkspace()


@contextmanager
def use(workspace: Workspace) -> Iterator[Workspace]:
    """Route every Workspace call to `workspace` until the block exits (fakes in tests)."""
    global _override
    previous, _override = _override, workspace
    try:
        yield workspace
    finally:
        _override = previous
