"""Invites and registering, through the served app: the Owner creates Invites in the
Owner panel, and an invited person registers with one and is logged in straight away."""

import datetime
import zoneinfo

import pytest
from logins import (
    OWNER_PASSWORD,
    OWNER_USERNAME,
    create_invite,
    invites_listed,
    log_in,
    log_out,
    register,
)

OWNERS_TIME_ZONE = zoneinfo.ZoneInfo("Europe/London")
ASK_FOR_A_NEW_ONE = "Ask the Owner for a new one."


@pytest.fixture
def invite(owner_client):
    """An Invite the Owner created, with the browser then logged out, ready for the
    invitee to register on it."""
    code = create_invite(owner_client)
    log_out(owner_client)
    return code


def is_logged_in(client) -> bool:
    return client.get("/chat/", follow_redirects=False).status_code == 200


def shown_as_logged_in(client) -> str:
    """What the chat UI's sidebar says about who is logged in."""
    started = client.post("/chat/gradio_api/call/logged_in_as", json={"data": []})
    return client.get(f"/chat/gradio_api/call/logged_in_as/{started.json()['event_id']}").text


def as_owner(client):
    log_in(client, OWNER_USERNAME, OWNER_PASSWORD)
    return client


# The Owner panel


def test_the_owner_creates_an_invite_and_sees_it_listed_with_its_expiry(owner_client):
    code = create_invite(owner_client)

    assert invites_listed(owner_client) == [code]
    in_a_week = datetime.datetime.now(OWNERS_TIME_ZONE) + datetime.timedelta(days=7)
    assert in_a_week.date().isoformat() in owner_client.get("/owner").text


def test_each_invite_is_its_own_code(owner_client):
    first, second = create_invite(owner_client), create_invite(owner_client)

    assert first != second
    assert sorted(invites_listed(owner_client)) == sorted([first, second])


def test_the_owner_withdraws_an_unused_invite(owner_client):
    kept, withdrawn = create_invite(owner_client), create_invite(owner_client)

    owner_client.post(f"/owner/invites/{withdrawn}/withdraw")

    assert invites_listed(owner_client) == [kept]


def test_without_a_session_the_owner_panel_sends_you_to_log_in(client):
    for response in (
        client.get("/owner", follow_redirects=False),
        client.post("/owner/invites", follow_redirects=False),
    ):
        assert response.status_code == 303
        assert response.headers["location"] == "/login"


def test_other_users_cant_reach_the_owner_panel(client, invite):
    register(client, invite, "sam")

    assert client.get("/owner").status_code == 403
    assert client.post("/owner/invites").status_code == 403
    assert "Only the Owner" in client.get("/owner").text
    log_out(client)
    assert invites_listed(as_owner(client)) == []  # Sam's attempt created no Invite


def test_only_the_owner_is_offered_the_owner_panel_in_the_chat_ui(client, invite):
    assert "/owner" in shown_as_logged_in(as_owner(client))
    log_out(client)

    register(client, invite, "sam")

    assert "/owner" not in shown_as_logged_in(client)


# Registering


def test_an_invited_person_registers_and_is_logged_in_straight_away(client, invite):
    response = register(client, invite, "Sam", display_name="Sam Smith")

    assert response.status_code == 303
    assert response.headers["location"] == "/chat/"
    assert is_logged_in(client)
    assert "Sam Smith" in shown_as_logged_in(client)


def test_a_registered_user_logs_in_again_later(client, invite):
    register(client, invite, "Sam", password="sams-password")
    log_out(client)

    log_in(client, "sam", "sams-password")

    assert "New User" in shown_as_logged_in(client)


def test_the_register_page_says_where_messages_go_and_prefills_the_time_zone(client):
    page = client.get("/register").text

    assert "Google Gemini" in page and "OpenAI" in page  # the paid fallback too
    assert "may be logged" in page
    assert "resolvedOptions().timeZone" in page


def test_the_owners_register_link_fills_in_the_invite(owner_client):
    code = create_invite(owner_client)
    log_out(owner_client)

    assert f'value="{code}"' in owner_client.get(f"/register?invite={code}").text


def test_an_invite_works_only_once(client, invite):
    register(client, invite, "sam")
    log_out(client)

    response = register(client, invite, "alex")

    assert "That Invite has already been used." in response.text
    assert ASK_FOR_A_NEW_ONE in response.text
    assert not is_logged_in(client)
    assert invites_listed(as_owner(client)) == []


def test_a_wrong_invite_is_refused(client, invite):
    response = register(client, invite + "x", "sam")

    assert "That Invite code isn't right." in response.text
    assert not is_logged_in(client)


def test_an_invite_expires_after_7_days(client, invite, monkeypatch):
    from perry import users

    in_8_days = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=8)
    monkeypatch.setattr(users, "_now", lambda: in_8_days)

    response = register(client, invite, "sam")

    assert "That Invite has expired." in response.text
    assert ASK_FOR_A_NEW_ONE in response.text
    assert not is_logged_in(client)
    assert invites_listed(as_owner(client)) == []


def test_an_invite_still_works_on_day_6(client, invite, monkeypatch):
    from perry import users

    in_6_days = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=6)
    monkeypatch.setattr(users, "_now", lambda: in_6_days)

    register(client, invite, "sam")

    assert is_logged_in(client)


def test_a_withdrawn_invite_is_refused(client, invite):
    as_owner(client).post(f"/owner/invites/{invite}/withdraw")
    log_out(client)

    response = register(client, invite, "sam")

    assert "That Invite was withdrawn by the Owner." in response.text
    assert not is_logged_in(client)


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"username": "OWNER"}, "The username 'OWNER' is taken."),
        ({"password": "too-short"}, "at least 10 characters"),
        ({"time_zone": "Mars/Olympus"}, "isn't a known time zone"),
    ],
    ids=["taken username, ignoring case", "short password", "unknown time zone"],
)
def test_a_failed_registration_says_why_and_keeps_the_invite(client, invite, fields, message):
    response = register(client, invite, **({"username": "sam"} | fields))

    assert response.status_code == 400
    assert message in response.text
    assert not is_logged_in(client)
    # The invitee fixes the mistake and registers with the same Invite.
    register(client, invite, "sam")
    assert is_logged_in(client)


def test_a_failed_registration_keeps_what_was_typed_except_the_password(client, invite):
    response = register(client, invite, "OWNER", password="secret-pass-1", display_name="Sam")

    assert 'value="OWNER"' in response.text
    assert 'value="Sam"' in response.text
    assert f'value="{invite}"' in response.text
    assert "secret-pass-1" not in response.text
