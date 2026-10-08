"""The Workspace on Google Calendar and Gmail, for the account connected in google_auth."""

import base64
import html
import re
from collections.abc import Callable
from datetime import date, datetime, time
from email.utils import getaddresses
from typing import TypeVar

from google.auth.exceptions import RefreshError
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from perry import google_auth, workspace
from perry.workspace import Email, Event, Status

T = TypeVar("T")

MAX_EVENTS = 100


class GoogleWorkspace:
    def status(self) -> Status:
        return google_auth.status()

    def _service(self, name: str, version: str):
        return build(name, version, credentials=google_auth.credentials(), cache_discovery=False)

    def list_events(self, start: datetime, end: datetime) -> list[Event]:
        def run():
            response = (
                self._service("calendar", "v3")
                .events()
                .list(
                    calendarId="primary",
                    timeMin=start.isoformat(),
                    timeMax=end.isoformat(),
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=MAX_EVENTS,
                )
                .execute()
            )
            return [_event(item) for item in response.get("items", [])]

        return _signed_in(run)

    def search_email(self, query: str, limit: int = 10) -> list[Email]:
        def run():
            messages = self._service("gmail", "v1").users().messages()
            found = messages.list(userId="me", q=query, maxResults=limit).execute()
            return [
                _email(
                    messages.get(
                        userId="me",
                        id=ref["id"],
                        format="metadata",
                        metadataHeaders=["From", "To", "Subject"],
                    ).execute(),
                    full=False,
                )
                for ref in found.get("messages", [])
            ]

        return _signed_in(run)

    def read_email_thread(self, email_thread_id: str) -> list[Email]:
        def run():
            threads = self._service("gmail", "v1").users().threads()
            try:
                thread = threads.get(userId="me", id=email_thread_id, format="full").execute()
            except HttpError as error:
                if error.status_code in (400, 404):
                    return []
                raise
            return [_email(m, full=True) for m in thread.get("messages", [])]

        return _signed_in(run)


def _signed_in(call: Callable[[], T]) -> T:
    """Run a Google call, turning a refused sign-in into workspace.ReconnectNeeded."""
    try:
        return call()
    except RefreshError as error:
        google_auth.mark_needs_reconnecting()
        raise workspace.expired() from error
    except HttpError as error:
        if error.status_code == 401:
            google_auth.mark_needs_reconnecting()
            raise workspace.expired() from error
        raise


def _when(value: dict) -> datetime:
    if "dateTime" in value:
        return datetime.fromisoformat(value["dateTime"]).astimezone()
    # All-day events have only a date.
    return datetime.combine(date.fromisoformat(value["date"]), time()).astimezone()


def _event(item: dict) -> Event:
    return Event(
        title=item.get("summary", "(no title)"),
        start=_when(item["start"]),
        end=_when(item["end"]),
        location=item.get("location", ""),
        attendees=tuple(a["email"] for a in item.get("attendees", []) if not a.get("self")),
        id=item["id"],
    )


def _email(message: dict, *, full: bool) -> Email:
    headers = {h["name"].lower(): h["value"] for h in message["payload"].get("headers", [])}
    body = _body(message["payload"]) if full else ""
    return Email(
        sender=headers.get("from", ""),
        subject=headers.get("subject", "(no subject)"),
        body=body or html.unescape(message.get("snippet", "")),
        sent_at=datetime.fromtimestamp(int(message["internalDate"]) / 1000).astimezone(),
        to=tuple(addr for _, addr in getaddresses([headers.get("to", "")]) if addr),
        id=message["id"],
        email_thread_id=message["threadId"],
    )


def _body(payload: dict) -> str:
    """The plain-text body, falling back to HTML with the tags stripped."""
    plain = _part(payload, "text/plain")
    if plain:
        return plain
    rich = _part(payload, "text/html")
    return html.unescape(re.sub(r"<[^>]+>", " ", rich)) if rich else ""


def _part(payload: dict, mime_type: str) -> str:
    if payload.get("mimeType") == mime_type and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", "replace")
    for part in payload.get("parts", []):
        if text := _part(part, mime_type):
            return text
    return ""
