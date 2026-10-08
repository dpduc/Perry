"""Connecting a Google account: OAuth consent and where the sign-in is kept.

The sign-in (refresh token) lives in one file under your user app-data folder,
never in the repo, Postgres or the sandbox. Nothing else in Perry reads it.
"""

import json
import os
import sys
from pathlib import Path

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from perry import workspace
from perry.workspace import Status

# The narrowest scopes covering every agreed Workspace operation, Read and Write,
# so there is a single consent screen:
# - calendar.events: list, create, move and delete events (not calendar settings or sharing).
# - gmail.readonly: search and read email.
# - gmail.send: send and reply (cannot read, change or delete mail).
SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]

REPO_ROOT = Path(__file__).resolve().parent.parent


def app_data_dir() -> Path:
    """Perry's folder in the user's app-data (%APPDATA%\\Perry on Windows)."""
    if override := os.getenv("PERRY_APP_DATA_DIR"):
        return Path(override)
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / "Perry"
    base = os.getenv("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / "perry"


def client_file() -> Path:
    """The OAuth client (Desktop app) downloaded from Google Cloud Console."""
    return Path(os.getenv("PERRY_GOOGLE_CLIENT_FILE") or app_data_dir() / "google_client.json")


def token_file() -> Path:
    path = (app_data_dir() / "google_token.json").resolve()
    if path.is_relative_to(REPO_ROOT):
        raise RuntimeError(f"Refusing to keep the Google sign-in inside the repo: {path}")
    return path


def connect() -> None:
    """Run the consent flow in the browser (blocks until done) and keep the sign-in."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not client_file().exists():
        raise FileNotFoundError(
            f"No Google OAuth client file at {client_file()}. See README: Connecting Google."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(client_file()), SCOPES)
    # prompt=consent makes Google return a refresh token even on a repeat connect.
    creds = flow.run_local_server(
        host="127.0.0.1", port=0, open_browser=True, prompt="consent", access_type="offline"
    )
    save(creds)


def save(creds: Credentials) -> None:
    _write({"credentials": json.loads(creds.to_json()), "needs_reconnecting": False})


def _read() -> dict | None:
    try:
        return json.loads(token_file().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _write(stored: dict) -> None:
    path = token_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(path)


def status() -> Status:
    stored = _read()
    if stored is None:
        return "not connected"
    return "needs reconnecting" if stored.get("needs_reconnecting") else "connected"


def mark_needs_reconnecting() -> None:
    """Google refused the sign-in (expired or revoked): keep asking until reconnected."""
    if stored := _read():
        _write({**stored, "needs_reconnecting": True})


def credentials() -> Credentials:
    """Valid credentials for the connected account, refreshed if needed.

    Raises workspace.ReconnectNeeded instead of any auth error.
    """
    stored = _read()
    if stored is None:
        raise workspace.not_connected()
    if stored.get("needs_reconnecting"):
        raise workspace.expired()
    creds = Credentials.from_authorized_user_info(stored["credentials"], SCOPES)
    if not creds.valid:
        try:
            creds.refresh(Request())
        except RefreshError as error:
            mark_needs_reconnecting()
            raise workspace.expired() from error
        save(creds)
    return creds
