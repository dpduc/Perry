"""The Workspace: your calendar and email, as the Task Agent sees them.

The Task Agent depends only on the `Workspace` interface below. `google_workspace`
implements it on Google Calendar and Gmail; `InMemoryWorkspace` is the fake used by
tests and evals. Every method here is a Read Action. Write Actions (send, reply,
create, move, delete) will be added beside them, each behind a Confirmation.
"""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol

Status = Literal["connected", "not connected", "needs reconnecting"]


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

    def list_events(self, start: datetime, end: datetime) -> list[Event]:
        """Events overlapping [start, end), earliest first."""
        ...

    def search_email(self, query: str, limit: int = 10) -> list[Email]:
        """Emails matching a Gmail-style query (e.g. `from:sam budget`), newest first."""
        ...

    def read_email_thread(self, email_thread_id: str) -> list[Email]:
        """Every email in one email thread, oldest first, with full bodies."""
        ...


@dataclass
class InMemoryWorkspace:
    """A Workspace held in memory, for tests and evals. Never touches Google.

    `search_email` understands `from:`, `to:` and plain words (all must match).
    `disconnect()` makes every call fail the way an expired sign-in does.
    """

    events: list[Event] = field(default_factory=list)
    emails: list[Email] = field(default_factory=list)
    _status: Status = "connected"

    def disconnect(self, status: Status = "needs reconnecting") -> None:
        self._status = status

    def status(self) -> Status:
        return self._status

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
