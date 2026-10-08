"""Where the Google sign-in is kept, and what an expired one turns into. No network."""

from datetime import UTC, datetime, timedelta

import pytest
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials

from perry import google_auth
from perry.google_workspace import GoogleWorkspace
from perry.workspace import ReconnectNeeded


@pytest.fixture
def app_data(tmp_path, monkeypatch):
    monkeypatch.setenv("PERRY_APP_DATA_DIR", str(tmp_path))
    return tmp_path


def expired_sign_in() -> Credentials:
    return Credentials(
        token="old-access-token",
        refresh_token="revoked-refresh-token",
        token_uri="https://oauth2.googleapis.com/token",
        client_id="client-id",
        client_secret="client-secret",
        scopes=google_auth.SCOPES,
        expiry=datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1),  # naive UTC
    )


def test_sign_in_is_kept_in_the_users_app_data_folder_outside_the_repo(monkeypatch):
    monkeypatch.delenv("PERRY_APP_DATA_DIR", raising=False)

    path = google_auth.token_file()

    assert not path.is_relative_to(google_auth.REPO_ROOT)
    assert path.parent.name.lower() == "perry"


def test_refuses_to_keep_the_sign_in_inside_the_repo(monkeypatch):
    monkeypatch.setenv("PERRY_APP_DATA_DIR", str(google_auth.REPO_ROOT / ".perry"))

    with pytest.raises(RuntimeError, match="inside the repo"):
        google_auth.token_file()


def test_without_a_sign_in_google_is_not_connected(app_data):
    google = GoogleWorkspace()

    assert google.status() == "not connected"
    with pytest.raises(ReconnectNeeded, match="isn't connected"):
        google.search_email("from:sam")


def test_a_refused_refresh_asks_for_reconnecting_and_keeps_asking(app_data, monkeypatch):
    google_auth.save(expired_sign_in())
    assert GoogleWorkspace().status() == "connected"

    def refused(self, request):
        raise RefreshError("invalid_grant: Token has been expired or revoked.")

    now = datetime.now().astimezone()
    with monkeypatch.context() as patch:
        patch.setattr(Credentials, "refresh", refused)
        with pytest.raises(ReconnectNeeded, match="Google needs reconnecting"):
            GoogleWorkspace().list_events(now, now + timedelta(days=1))
    assert GoogleWorkspace().status() == "needs reconnecting"

    # Even if a refresh would now work, the user must reconnect first.
    with pytest.raises(ReconnectNeeded, match="Google needs reconnecting"):
        google_auth.credentials()
