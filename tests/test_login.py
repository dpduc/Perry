"""Logging in and out of the served app: nothing in Perry is reachable without a session."""

import pytest
from logins import OWNER_PASSWORD, OWNER_USERNAME, log_in

# The Gradio API, as the browser's chat page calls it.
QUEUE_JOIN = "/chat/gradio_api/queue/join"


def is_logged_in(client) -> bool:
    return client.get("/chat/", follow_redirects=False).status_code == 200


def test_the_owner_logs_in_and_reaches_the_chat_ui(client):
    response = log_in(client, OWNER_USERNAME, OWNER_PASSWORD)

    assert response.status_code == 303
    assert response.headers["location"] == "/chat/"
    assert client.get("/chat/").status_code == 200


def test_the_username_ignores_case(client):
    log_in(client, OWNER_USERNAME.upper(), OWNER_PASSWORD)

    assert is_logged_in(client)


@pytest.mark.parametrize(
    "username, password",
    [(OWNER_USERNAME, "not-the-password"), ("nobody", OWNER_PASSWORD)],
    ids=["wrong password", "unknown username"],
)
def test_wrong_credentials_get_the_same_generic_message(client, username, password):
    response = log_in(client, username, password)

    assert "Wrong username or password." in response.text
    assert not is_logged_in(client)


def test_without_a_session_the_chat_ui_redirects_to_login(client):
    for page in ("/", "/chat", "/chat/"):
        response = client.get(page, follow_redirects=False)

        assert response.status_code == 303
        assert response.headers["location"] == "/login"


def test_without_a_session_the_gradio_api_is_refused(client):
    assert client.get("/chat/config").status_code == 401
    assert client.get("/chat/gradio_api/info").status_code == 401
    joined = client.post(QUEUE_JOIN, json={"data": [], "fn_index": 0, "session_hash": "x"})
    assert joined.status_code == 401


def test_the_session_lasts_across_requests(owner_client):
    assert is_logged_in(owner_client)
    assert owner_client.get("/chat/config").status_code == 200
    assert owner_client.get("/", follow_redirects=False).headers["location"] == "/chat/"


def test_logging_out_ends_the_session(owner_client):
    response = owner_client.get("/logout", follow_redirects=False)

    assert response.headers["location"] == "/login"
    assert not is_logged_in(owner_client)
    assert owner_client.get("/chat/config").status_code == 401


def test_the_session_cookie_is_http_only_and_lasts_14_days(client):
    cookie = log_in(client, OWNER_USERNAME, OWNER_PASSWORD).headers["set-cookie"].lower()

    assert "httponly" in cookie
    assert f"max-age={14 * 24 * 60 * 60}" in cookie


def test_a_forged_session_cookie_is_refused(client):
    client.cookies.set("perry_session", "eyJ1c2VyIjoxLCJ2ZXJzaW9uIjowfQ.forged.signature")

    assert not is_logged_in(client)


def test_a_session_signed_with_another_secret_is_refused(client, monkeypatch):
    log_in(client, OWNER_USERNAME, OWNER_PASSWORD)

    monkeypatch.setenv("PERRY_SESSION_SECRET", "a-different-secret")

    assert not is_logged_in(client)


def test_the_app_refuses_to_start_without_an_owner(database):
    from perry import users, web

    with pytest.raises(users.Refused, match="python -m perry.users create-owner"):
        web.create_app()


def test_the_app_refuses_to_start_without_a_session_secret(owner, monkeypatch):
    from perry import users, web

    monkeypatch.delenv("PERRY_SESSION_SECRET")

    with pytest.raises(users.Refused, match="PERRY_SESSION_SECRET"):
        web.create_app()
