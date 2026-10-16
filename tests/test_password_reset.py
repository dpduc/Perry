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
    log_out,
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
    log_out(sam)

    response = reset(sam, code)

    assert response.status_code == 303
    assert is_logged_in(sam)
    log_out(sam)
    assert "Wrong username or password." in log_in(sam, "sam", SAMS_PASSWORD).text
    log_in(sam, "sam", NEW_PASSWORD)
    assert is_logged_in(sam)


def test_a_reset_code_works_once(sam, owner_browser):
    code = issue_reset_code(owner_browser, "sam")
    reset(sam, code)
    log_out(sam)

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
    log_out(sam)
    assert "Wrong username or password." in log_in(sam, "sam", NEW_PASSWORD).text
    assert "sam" not in reset_codes_listed(owner_browser)


def test_a_wrong_reset_code_is_refused(sam):
    response = reset(sam, "not-a-code")

    assert response.status_code == 400
    assert "That Reset Code isn't right." in response.text


def test_the_new_password_must_be_at_least_10_characters(sam, owner_browser):
    code = issue_reset_code(owner_browser, "sam")
    log_out(sam)

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
    log_out(sam)
    for _ in range(5):
        log_in(sam, "sam", "not-the-password")
    assert log_in(sam, "sam", SAMS_PASSWORD).status_code == 429

    reset(sam, issue_reset_code(owner_browser, "sam"))
    log_out(sam)
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


def disable(owner_client, username: str) -> None:
    owner_client.post(f"/owner/users/{user_id_of(owner_client, username)}/disable")


def test_a_disabled_user_cant_redeem_a_reset_code(sam, owner_browser):
    code = issue_reset_code(owner_browser, "sam")
    disable(owner_browser, "sam")

    response = reset(sam, code)

    assert response.status_code == 400
    assert "disabled" in response.text
    assert not is_logged_in(sam)
    owner_browser.post(f"/owner/users/{user_id_of(owner_browser, 'sam')}/enable")
    assert "Wrong username or password." in log_in(sam, "sam", NEW_PASSWORD).text
    log_in(sam, "sam", SAMS_PASSWORD)
    assert is_logged_in(sam)


def test_the_owner_isnt_offered_a_reset_code_for_a_disabled_user(sam, owner_browser):
    sams_id = user_id_of(owner_browser, "sam")
    disable(owner_browser, "sam")

    assert f"/owner/users/{sams_id}/reset" not in owner_browser.get("/owner").text
    response = owner_browser.post(f"/owner/users/{sams_id}/reset")
    assert "disabled" in response.text
    assert reset_codes_listed(owner_browser) == {}


def test_a_new_reset_code_is_the_one_just_created(owner, monkeypatch):
    """Not just the open one that expires last."""
    from perry import users

    sam = users.create_user("sam", "sam-password-1", "Sam", "UTC")
    alex = users.create_user("alex", "alex-password-1", "Alex", "UTC")
    hours_from_now(monkeypatch, 1)
    users.create_reset_code(alex.id)
    monkeypatch.undo()

    assert users.create_reset_code(sam.id).username == "sam"


# Resetting a password on this machine (python -m perry.users reset-password)


def reset_on_this_machine(username: str, *typed, prompts: list[str] | None = None) -> int:
    """Run `python -m perry.users reset-password <username>`, typing `typed` at its
    password prompts (the new password, then again). An answer that is an exception
    (EOFError, KeyboardInterrupt) is raised at its prompt instead of typed; `prompts`
    collects the prompts shown. Fails if Perry asks for more than is typed."""
    from perry import users

    answers = iter(typed)

    def prompt(shown: str) -> str:
        if prompts is not None:
            prompts.append(shown)
        answer = next(answers, None)
        assert answer is not None, f"Perry asked more than expected: {shown!r}"
        if isinstance(answer, BaseException):
            raise answer
        return answer

    return users.main(["reset-password", username], ask=prompt, ask_secret=prompt)


def test_a_locked_out_owner_resets_their_password_on_this_machine(owner_client, capsys):
    from fastapi.testclient import TestClient

    from perry import web

    with TestClient(web.create_app()) as other_browser:
        for _ in range(5):
            log_in(other_browser, OWNER_USERNAME, "not-the-password")

        assert reset_on_this_machine(OWNER_USERNAME, NEW_PASSWORD, NEW_PASSWORD) == 0

        assert not is_logged_in(owner_client)  # every session ended
        log_in(other_browser, OWNER_USERNAME, NEW_PASSWORD)  # the failed logins cleared
        assert is_logged_in(other_browser)
    assert "reset" in capsys.readouterr().out.lower()


def test_resetting_an_unknown_user_on_this_machine_fails_clearly(owner, capsys):
    assert reset_on_this_machine("nobody", NEW_PASSWORD, NEW_PASSWORD) == 1
    assert "nobody" in capsys.readouterr().err


def test_mismatched_passwords_on_this_machine_ask_both_again(owner_client, capsys):
    prompts = []
    assert reset_on_this_machine(
        OWNER_USERNAME, NEW_PASSWORD, "something-else-1", "third-password-1", "third-password-1",
        prompts=prompts,
    ) == 0

    assert "don't match" in capsys.readouterr().err
    assert prompts.count("New password again: ") == 2
    assert not is_logged_in(owner_client)  # the password was changed


def test_a_short_password_on_this_machine_is_asked_again_at_once(owner_client, capsys):
    prompts = []
    assert reset_on_this_machine(
        OWNER_USERNAME, "short", NEW_PASSWORD, NEW_PASSWORD, prompts=prompts
    ) == 0

    assert "at least 10 characters" in capsys.readouterr().err
    assert prompts.count("New password again: ") == 1


def test_ctrl_c_while_resetting_on_this_machine_changes_nothing(owner_client, capsys):
    assert reset_on_this_machine(OWNER_USERNAME, NEW_PASSWORD, KeyboardInterrupt()) == 1

    assert "Nothing was changed." in capsys.readouterr().err
    assert is_logged_in(owner_client)


def test_end_of_input_while_resetting_on_this_machine_changes_nothing(owner_client, capsys):
    assert reset_on_this_machine(OWNER_USERNAME, EOFError()) == 1

    assert "Nothing was changed." in capsys.readouterr().err
    assert is_logged_in(owner_client)


# Changing your own password


def change_password(client, current: str, new: str = NEW_PASSWORD):
    return client.post("/password", data={"current_password": current, "new_password": new})


def test_a_logged_in_user_changes_their_password_by_entering_the_current_one(sam):
    response = change_password(sam, SAMS_PASSWORD)

    assert response.status_code == 200
    assert "Your password is changed." in response.text
    assert is_logged_in(sam)  # this session carries on
    log_out(sam)
    assert "Wrong username or password." in log_in(sam, "sam", SAMS_PASSWORD).text
    log_in(sam, "sam", NEW_PASSWORD)
    assert is_logged_in(sam)


def test_changing_the_password_needs_the_right_current_one(sam):
    response = change_password(sam, "not-the-password")

    assert response.status_code == 400
    assert "That isn't your current password." in response.text
    log_out(sam)
    log_in(sam, "sam", SAMS_PASSWORD)
    assert is_logged_in(sam)


def test_a_changed_password_must_be_at_least_10_characters(sam):
    response = change_password(sam, SAMS_PASSWORD, "short")

    assert response.status_code == 400
    assert "at least 10 characters" in response.text
    log_out(sam)
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
