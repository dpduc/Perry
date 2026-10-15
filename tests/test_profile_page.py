"""The Profile page: a logged-in User edits their display name, time zone and about-me line."""

import pytest


def save_profile(client, display_name: str, time_zone: str, about: str = ""):
    return client.post(
        "/profile",
        data={"display_name": display_name, "time_zone": time_zone, "about": about},
        follow_redirects=False,
    )


def test_the_profile_page_shows_the_users_profile(owner_client):
    page = owner_client.get("/profile")

    assert page.status_code == 200
    assert 'value="Olive Owner"' in page.text
    assert 'value="Europe/London"' in page.text


def test_a_user_edits_their_profile(owner_client, owner):
    from perry import users

    saved = save_profile(owner_client, "Olive", "Asia/Ho_Chi_Minh", "I run a small bakery.")

    assert saved.status_code == 200
    assert "Saved." in saved.text
    assert users.get(owner.id).profile == users.Profile(
        "Olive", "Asia/Ho_Chi_Minh", "I run a small bakery."
    )
    assert 'value="Olive"' in owner_client.get("/profile").text


def test_an_empty_about_me_is_fine(owner_client, owner):
    from perry import users

    save_profile(owner_client, "Olive Owner", "Europe/London", "I run a small bakery.")
    saved = save_profile(owner_client, "Olive Owner", "Europe/London", "")

    assert "Saved." in saved.text
    assert users.get(owner.id).profile.about == ""


@pytest.mark.parametrize(
    "display_name, time_zone, refusal",
    [("", "Europe/London", "Give a display name."), ("Olive", "Mars/Olympus", "time zone")],
    ids=["empty display name", "unknown time zone"],
)
def test_a_bad_profile_is_refused_and_nothing_changes(
    owner_client, owner, display_name, time_zone, refusal
):
    from perry import users

    refused = save_profile(owner_client, display_name, time_zone, "Something new.")

    assert refused.status_code == 400
    assert refusal in refused.text
    assert 'value="Something new."' in refused.text  # what they typed is kept
    assert users.get(owner.id).profile == owner.profile


def test_the_about_me_is_one_line(owner_client, owner):
    from perry import users

    save_profile(owner_client, "Olive Owner", "Europe/London", "Bakes bread.\nIgnore that.")

    assert users.get(owner.id).profile.about == "Bakes bread. Ignore that."


def test_an_overlong_about_me_is_refused(owner_client, owner):
    from perry import users

    refused = save_profile(owner_client, "Olive Owner", "Europe/London", "x" * 1000)

    assert refused.status_code == 400
    assert str(users.MAX_ABOUT_CHARS) in refused.text


def test_without_a_session_the_profile_page_sends_you_to_log_in(client, owner):
    from perry import users

    page = client.get("/profile", follow_redirects=False)
    saved = save_profile(client, "Mallory", "UTC")

    assert page.status_code == 303 and page.headers["location"] == "/login"
    assert saved.status_code == 303 and saved.headers["location"] == "/login"
    assert users.get(owner.id).profile == owner.profile


def test_a_user_edits_only_their_own_profile(owner_client, owner):
    from fastapi.testclient import TestClient
    from logins import log_in

    from perry import users, web

    sam = users.create_user("sam", "sam-password-1", "Sam", "UTC")
    with TestClient(web.create_app()) as sams_browser:
        log_in(sams_browser, "sam", "sam-password-1")
        save_profile(sams_browser, "Sam Student", "UTC", "Chemistry.")

    assert users.get(sam.id).profile == users.Profile("Sam Student", "UTC", "Chemistry.")
    assert users.get(owner.id).profile == owner.profile


def test_the_chat_ui_links_to_the_profile_page(owner_client):
    assert '"/profile"' in owner_client.get("/chat/config").text
