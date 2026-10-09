"""The Workspace on Google Calendar and Gmail, for the account connected in google_auth."""

import base64
import html
import re
from collections.abc import Callable
from datetime import date, datetime, time
from email.mime.text import MIMEText
from email.utils import getaddresses
from typing import TypeVar

from google.auth.exceptions import RefreshError
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from perry import google_auth, workspace
from perry.workspace import AccountKind, Email, Event, Status

T = TypeVar("T")

MAX_EVENTS = 100


class GoogleWorkspace:
    def status(self) -> Status:
        return google_auth.status()

    def account_kind(self) -> AccountKind | None:
        return google_auth.account_kind()

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

    def get_event(self, event_id: str) -> Event | None:
        def run():
            try:
                item = self._events().get(calendarId="primary", eventId=event_id).execute()
            except HttpError as error:
                if error.status_code in (400, 404, 410):
                    return None
                raise
            return None if item.get("status") == "cancelled" else _event(item)

        return _signed_in(run)

    # Write Actions. Attendees are notified of calendar changes (sendUpdates="all").

    def _events(self):
        return self._service("calendar", "v3").events()

    def send_email(self, to: tuple[str, ...], subject: str, body: str) -> Email:
        message = MIMEText(body)
        message["To"] = ", ".join(to)
        message["Subject"] = subject
        return self._send(message, subject, body, to)

    def reply_email(self, email_thread_id: str, body: str) -> Email:
        def latest():
            thread = (
                self._service("gmail", "v1")
                .users()
                .threads()
                .get(
                    userId="me",
                    id=email_thread_id,
                    format="metadata",
                    metadataHeaders=["From", "Reply-To", "Subject", "Message-ID", "References"],
                )
                .execute()
            )
            last = thread["messages"][-1]
            return {h["name"].lower(): h["value"] for h in last["payload"].get("headers", [])}

        headers = _signed_in(latest)
        subject = headers.get("subject", "")
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"
        to = headers.get("reply-to") or headers.get("from", "")
        message = MIMEText(body)
        message["To"] = to
        message["Subject"] = subject
        if message_id := headers.get("message-id"):
            message["In-Reply-To"] = message_id
            message["References"] = f"{headers.get('references', '')} {message_id}".strip()
        recipients = tuple(addr for _, addr in getaddresses([to]) if addr)
        return self._send(message, subject, body, recipients, email_thread_id)

    def _send(
        self, message: MIMEText, subject: str, body: str, to: tuple[str, ...], thread_id: str = ""
    ) -> Email:
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        payload = {"raw": raw, **({"threadId": thread_id} if thread_id else {})}

        def run():
            return (
                self._service("gmail", "v1").users().messages().send(userId="me", body=payload).execute()
            )

        sent = _signed_in(run)
        return Email(
            "me",
            subject,
            body,
            datetime.now().astimezone(),
            to,
            sent["id"],
            sent.get("threadId", thread_id),
        )

    def create_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        location: str = "",
        attendees: tuple[str, ...] = (),
    ) -> Event:
        body = {
            "summary": title,
            "start": {"dateTime": start.isoformat()},
            "end": {"dateTime": end.isoformat()},
            "location": location,
            "attendees": [{"email": a} for a in attendees],
        }
        return _signed_in(
            lambda: _event(
                self._events()
                .insert(calendarId="primary", body=body, sendUpdates="all")
                .execute()
            )
        )

    def move_event(self, event_id: str, start: datetime, end: datetime) -> Event:
        body = {"start": {"dateTime": start.isoformat()}, "end": {"dateTime": end.isoformat()}}
        return _signed_in(
            lambda: _event(
                self._events()
                .patch(calendarId="primary", eventId=event_id, body=body, sendUpdates="all")
                .execute()
            )
        )

    def delete_event(self, event_id: str) -> None:
        _signed_in(
            lambda: self._events()
            .delete(calendarId="primary", eventId=event_id, sendUpdates="all")
            .execute()
        )


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
