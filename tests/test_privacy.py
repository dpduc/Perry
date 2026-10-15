"""Privacy switches: a real Google account only with local text generation, never traced.

Covers every combination of test/real account and local/cloud text provider, at
connect time and at startup. No network: Google consent is faked.
"""

from datetime import UTC, datetime, timedelta

import pytest
from fakes import ScriptedChatModel, fake_provider
from google.oauth2.credentials import Credentials

from perry import google_auth, models, privacy


@pytest.fixture
def app_data(tmp_path, monkeypatch):
    monkeypatch.setenv("PERRY_APP_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def cloud():
    with models.use(fake_provider(ScriptedChatModel())) as provider:
        yield provider


@pytest.fixture
def local():
    with models.use(fake_provider(ScriptedChatModel(), local=True)) as provider:
        yield provider


def a_sign_in() -> Credentials:
    return Credentials(
        token="access-token",
        refresh_token="refresh-token",
        token_uri="https://oauth2.googleapis.com/token",
        client_id="client-id",
        client_secret="client-secret",
        scopes=google_auth.SCOPES,
        expiry=datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1),  # naive UTC
    )


def consent_given():
    """Stands in for the browser consent flow, and records that it ran."""
    consent_given.ran = True
    return a_sign_in()


@pytest.fixture(autouse=True)
def no_consent_yet():
    consent_given.ran = False


@pytest.fixture(autouse=True)
def tracing_restored(monkeypatch):
    """Undo any tracing switch a test flips, so it can't leak into other tests."""
    import langsmith

    for prefix in ("LANGSMITH", "LANGCHAIN"):
        monkeypatch.setenv(f"{prefix}_TRACING_V2", "false")
        monkeypatch.setenv(f"{prefix}_TRACING", "false")
    yield
    langsmith.configure(enabled=None)  # back to whatever the environment says


def test_connecting_a_real_account_with_a_cloud_text_provider_is_refused(app_data, cloud):
    with pytest.raises(privacy.Refused, match="local"):
        google_auth.connect("real", consent=consent_given)

    assert not consent_given.ran  # refused before Google is even asked
    assert google_auth.status() == "not connected"


def test_connecting_a_real_account_with_a_local_text_provider_keeps_it_as_real(app_data, local):
    google_auth.connect("real", consent=consent_given)

    assert google_auth.status() == "connected"
    assert google_auth.account_kind() == "real"


@pytest.mark.parametrize("provider", ["cloud", "local"])
def test_a_test_account_connects_whatever_the_text_provider(app_data, provider, request):
    request.getfixturevalue(provider)

    google_auth.connect("test", consent=consent_given)

    assert google_auth.status() == "connected"
    assert google_auth.account_kind() == "test"


def test_the_account_kind_survives_a_refreshed_sign_in(app_data, cloud):
    google_auth.connect("test", consent=consent_given)

    google_auth.save(a_sign_in())  # what a refresh does

    assert google_auth.account_kind() == "test"


def test_a_sign_in_kept_before_the_kind_was_asked_counts_as_real(app_data):
    google_auth.token_file().write_text(
        '{"credentials": {}, "needs_reconnecting": false}', encoding="utf-8"
    )

    assert google_auth.account_kind() == "real"


# --- Starting Perry with an account already connected ---------------------------


@pytest.fixture
def mailbox():
    """The Workspace the Turns see: one email from Sam, on an account of either kind."""
    from perry import workspace
    from perry.workspace import Email, InMemoryWorkspace

    fake = InMemoryWorkspace(
        emails=[
            Email(
                "Sam Lee <sam@example.com>",
                "Q3 budget",
                "The Q3 budget is approved at 40k.",
                datetime(2026, 9, 28, 10).astimezone(),
                id="m1",
            )
        ]
    )
    with workspace.use(fake):
        yield fake


def ask_about_sam(perry):
    from fakes import call, last_message

    perry.script(
        call("ask_task_agent", request="What did Sam say about the budget?"),  # Supervisor
        call("search_email", query="from:sam budget"),  # Task Agent
        last_message,  # Task Agent reports
        last_message,  # Supervisor passes it on
    )
    return perry.send("What did Sam say about the budget?")


@pytest.mark.parametrize(
    ("account", "local_text_generation"), [("test", False), ("test", True), ("real", True)]
)
def test_the_task_agent_reads_mail_on_allowed_combinations(
    perry, mailbox, account, local_text_generation
):
    mailbox.account = account

    with models.use(fake_provider(perry.model, local=local_text_generation)):
        assert privacy.startup() is None
        reply = ask_about_sam(perry)

    assert "approved at 40k" in reply


def test_a_real_account_with_a_cloud_text_provider_disables_the_task_agent(perry, mailbox):
    mailbox.account = "real"

    notice = privacy.startup()
    reply = ask_about_sam(perry)

    assert "Task Agent is off" in notice and "local" in notice
    assert "Task Agent is off" in reply
    assert "40k" not in reply
    # Only the Supervisor's call reached the model: no email was read or sent to it.
    assert len(perry.model.calls) == 1
    assert perry.history() == []


# --- Tracing --------------------------------------------------------------------


@pytest.fixture
def tracing_on(monkeypatch):
    """Tracing switched on, as .env does with LANGCHAIN_TRACING_V2=true."""
    import langsmith
    from langsmith.utils import tracing_is_enabled

    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    langsmith.configure(enabled=True)
    assert tracing_is_enabled()


@pytest.mark.parametrize("provider", ["cloud", "local"])
def test_with_a_test_account_tracing_follows_env(mailbox, tracing_on, provider, request):
    from langsmith.utils import tracing_is_enabled

    request.getfixturevalue(provider)

    privacy.startup()

    assert tracing_is_enabled()


@pytest.mark.parametrize("provider", ["cloud", "local"])
def test_with_a_real_account_connected_tracing_is_off_whatever_env_says(
    mailbox, tracing_on, provider, request
):
    import os

    from langsmith.utils import tracing_is_enabled

    request.getfixturevalue(provider)
    mailbox.account = "real"

    privacy.startup()

    assert not tracing_is_enabled()
    assert os.environ["LANGCHAIN_TRACING_V2"] == os.environ["LANGSMITH_TRACING"] == "false"


def test_connecting_a_real_account_turns_tracing_off_straight_away(app_data, local, tracing_on):
    from langsmith.utils import tracing_is_enabled

    google_auth.connect("real", consent=consent_given)

    assert not tracing_is_enabled()


def test_the_connect_button_explains_a_refusal(app_data, cloud, owner):
    from logins import request_as

    from perry import app

    _, message = app.connect_google(request_as(owner), "real")

    assert "won't connect a real Google account" in message
    assert google_auth.status() == "not connected"
