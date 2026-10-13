"""Password reset and change, through the served app: the Owner issues a Reset Code from
the Owner panel for a User who forgot their password, the User chooses a new one on the
reset page, and any logged-in User changes their own password."""

import datetime
import re

from logins import (
    OWNER_PASSWORD,
    OWNER_USERNAME,
    SAMS_PASSWORD,
    is_logged_in,
    log_in,
    user_id_of,
)

NEW_PASSWORD = "sams-new-password-1"


def reset_codes_listed(owner_client) -> dict[str, str]:
    """The unused Reset Codes the Owner panel lists, by the username they are for."""
    page = owner_client.get("/owner").text
    found = re.findall(r'<tr data-reset="([^"]+)">\s*<td>([^<]*)</td>', page)
    return {username: code for code, username in found}


def issue_reset_code(owner_client, username: str) -> str:
    """Issue a Reset Code from the Owner panel, as the Owner clicking its button."""
    owner_client.post(f"/owner/users/{user_id_of(owner_client, username)}/reset")
    return reset_codes_listed(owner_client)[username]


def hours_from_now(monkeypatch, hours: int) -> None:
    from perry import users

    later = datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=hours)
    monkeypatch.setattr(users, "_now", lambda: later)


def reset(client, code: str, password: str = NEW_PASSWORD):
    """Submit the reset page; returns the response (a redirect into Perry on success)."""
    return client.post(
        "/reset", data={"code": code, "password": password}, follow_redirects=False
    )


def test_a_user_resets_their_password_with_a_reset_code_from_the_owner(sam, owner_browser):
    code = issue_reset_code(owner_browser, "sam")
    sam.get("/logout")

    response = reset(sam, code)

    assert response.status_code == 303
    assert is_logged_in(sam)
    sam.get("/logout")
    assert "Wrong username or password." in log_in(sam, "sam", SAMS_PASSWORD).text
    log_in(sam, "sam", NEW_PASSWORD)
    assert is_logged_in(sam)


def test_a_reset_code_works_once(sam, owner_browser):
    code = issue_reset_code(owner_browser, "sam")
    reset(sam, code)
    sam.get("/logout")

    response = reset(sam, code, "yet-another-password-1")

    assert response.status_code == 400
    assert "That Reset Code has already been used." in response.text
    assert not is_logged_in(sam)
    assert "sam" not in reset_codes_listed(owner_browser)


def test_a_reset_code_expires_after_a_day(sam, owner_browser, monkeypatch):
    code = issue_reset_code(owner_browser, "sam")

    hours_from_now(monkeypatch, 25)
    response = reset(sam, code)

    assert response.status_code == 400
    assert "That Reset Code has expired." in response.text
    sam.get("/logout")
    assert "Wrong username or password." in log_in(sam, "sam", NEW_PASSWORD).text
    assert "sam" not in reset_codes_listed(owner_browser)


def test_a_wrong_reset_code_is_refused(sam):
    response = reset(sam, "not-a-code")

    assert response.status_code == 400
    assert "That Reset Code isn't right." in response.text


def test_the_new_password_must_be_at_least_10_characters(sam, owner_browser):
    code = issue_reset_code(owner_browser, "sam")
    sam.get("/logout")

    response = reset(sam, code, "short")

    assert response.status_code == 400
    assert "at least 10 characters" in response.text
    assert not is_logged_in(sam)
    assert reset_codes_listed(owner_browser)["sam"] == code  # still usable
    log_in(sam, "sam", SAMS_PASSWORD)
    assert is_logged_in(sam)


def test_a_reset_ends_the_users_other_sessions(sam, owner_browser):
    from fastapi.testclient import TestClient

    from perry import web

    code = issue_reset_code(owner_browser, "sam")
    with TestClient(web.create_app()) as other_browser:
        reset(other_browser, code)

        assert is_logged_in(other_browser)
        assert not is_logged_in(sam)


def test_a_reset_clears_the_failed_logins_for_that_username(sam, owner_browser):
    sam.get("/logout")
    for _ in range(5):
        log_in(sam, "sam", "not-the-password")
    assert log_in(sam, "sam", SAMS_PASSWORD).status_code == 429

    reset(sam, issue_reset_code(owner_browser, "sam"))
    sam.get("/logout")
    log_in(sam, "sam", NEW_PASSWORD)

    assert is_logged_in(sam)


def test_the_reset_page_fills_in_the_code_from_the_link(sam, owner_browser):
    code = issue_reset_code(owner_browser, "sam")

    page = sam.get(f"/reset?code={code}").text

    assert f'value="{code}"' in page


def test_only_the_owner_issues_reset_codes(sam, owner_browser):
    owners_id = user_id_of(owner_browser, OWNER_USERNAME)

    assert sam.post(f"/owner/users/{owners_id}/reset").status_code == 403
    assert reset_codes_listed(owner_browser) == {}


# Changing your own password


def change_password(client, current: str, new: str = NEW_PASSWORD):
    return client.post("/password", data={"current_password": current, "new_password": new})


def test_a_logged_in_user_changes_their_password_by_entering_the_current_one(sam):
    response = change_password(sam, SAMS_PASSWORD)

    assert response.status_code == 200
    assert "Your password is changed." in response.text
    assert is_logged_in(sam)  # this session carries on
    sam.get("/logout")
    assert "Wrong username or password." in log_in(sam, "sam", SAMS_PASSWORD).text
    log_in(sam, "sam", NEW_PASSWORD)
    assert is_logged_in(sam)


def test_changing_the_password_needs_the_right_current_one(sam):
    response = change_password(sam, "not-the-password")

    assert response.status_code == 400
    assert "That isn't your current password." in response.text
    sam.get("/logout")
    log_in(sam, "sam", SAMS_PASSWORD)
    assert is_logged_in(sam)


def test_a_changed_password_must_be_at_least_10_characters(sam):
    response = change_password(sam, SAMS_PASSWORD, "short")

    assert response.status_code == 400
    assert "at least 10 characters" in response.text
    sam.get("/logout")
    log_in(sam, "sam", SAMS_PASSWORD)
    assert is_logged_in(sam)


def test_changing_the_password_ends_the_users_other_sessions(sam):
    from fastapi.testclient import TestClient

    from perry import web

    with TestClient(web.create_app()) as other_browser:
        log_in(other_browser, "sam", SAMS_PASSWORD)

        change_password(sam, SAMS_PASSWORD)

        assert not is_logged_in(other_browser)
        assert is_logged_in(sam)


def test_changing_the_password_needs_a_session(client):
    response = client.get("/password", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"
    assert change_password(client, OWNER_PASSWORD).url.path == "/login"


def test_the_profile_page_links_to_changing_the_password(sam):
    assert 'href="/password"' in sam.get("/profile").text
