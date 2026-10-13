"""Login limits and disabling Users, through the served app: failed logins are limited per
username, and the Owner lists Users in the Owner panel and disables or re-enables them."""

import datetime
import re
import zoneinfo

from fakes import with_usage
from logins import (
    OWNER_PASSWORD,
    OWNER_USERNAME,
    SAMS_PASSWORD,
    create_invite,
    is_logged_in,
    log_in,
    log_out,
    register,
    user_id_of,
)

TRY_AGAIN_LATER = "Too many failed logins. Try again later."


def minutes_from_now(monkeypatch, minutes: int) -> None:
    from perry import users

    later = datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=minutes)
    monkeypatch.setattr(users, "_now", lambda: later)


def fail_to_log_in(client, username: str, times: int) -> None:
    for _ in range(times):
        log_in(client, username, "not-the-password")


# Login limits


def test_after_5_failed_logins_even_the_right_password_is_refused(client):
    fail_to_log_in(client, OWNER_USERNAME, 5)

    response = log_in(client, OWNER_USERNAME, OWNER_PASSWORD)

    assert response.status_code == 429
    assert TRY_AGAIN_LATER in response.text
    assert not is_logged_in(client)


def test_4_failed_logins_still_let_the_right_password_in(client):
    fail_to_log_in(client, OWNER_USERNAME, 4)

    log_in(client, OWNER_USERNAME, OWNER_PASSWORD)

    assert is_logged_in(client)


def test_the_limit_is_per_username_ignoring_case(client):
    from perry import users

    users.create_user("sam", "sam-password-1", "Sam", "UTC")
    fail_to_log_in(client, "SAM", 5)

    assert TRY_AGAIN_LATER in log_in(client, "sam", "sam-password-1").text
    log_in(client, OWNER_USERNAME, OWNER_PASSWORD)
    assert is_logged_in(client)


def test_the_limit_lifts_after_15_minutes(client, monkeypatch):
    fail_to_log_in(client, OWNER_USERNAME, 5)

    minutes_from_now(monkeypatch, 16)
    log_in(client, OWNER_USERNAME, OWNER_PASSWORD)

    assert is_logged_in(client)


def test_failed_logins_for_unknown_usernames_are_limited_too(client):
    fail_to_log_in(client, "nobody", 5)

    assert TRY_AGAIN_LATER in log_in(client, "nobody", "anything-at-all").text


def test_logins_at_the_same_moment_dont_get_past_the_limit(owner):
    """Many wrong passwords at once: still only 5 are checked, the rest are refused."""
    from concurrent.futures import ThreadPoolExecutor

    from perry import users

    def attempt(_) -> str:
        try:
            users.authenticate(OWNER_USERNAME, "not-the-password")
        except users.TooManyLogins:
            return "limited"
        except users.Refused:
            return "wrong"
        return "in"

    with ThreadPoolExecutor(max_workers=10) as pool:
        outcomes = list(pool.map(attempt, range(10)))

    assert sorted(outcomes) == ["limited"] * 5 + ["wrong"] * 5


def test_failed_logins_older_than_the_window_are_deleted(client, monkeypatch, database):
    fail_to_log_in(client, OWNER_USERNAME, 3)

    minutes_from_now(monkeypatch, 16)
    fail_to_log_in(client, "someone-else", 1)

    with database.pool().connection() as conn:
        kept = conn.execute("SELECT username FROM failed_logins").fetchall()
    assert [row["username"] for row in kept] == ["someone-else"]


# The Users list


def users_listed(owner_client) -> list[dict]:
    """The rows of the Owner panel's Users table: username, joined, last login, status."""
    page = owner_client.get("/owner").text
    rows = re.findall(r'<tr data-user="\d+">(.*?)</tr>', page, flags=re.DOTALL)
    keys = ("username", "joined", "last_login", "status")
    return [dict(zip(keys, cells(row), strict=False)) for row in rows]


def cells(row: str) -> list[str]:
    found = re.findall(r"<td>(.*?)</td>", row, flags=re.DOTALL)
    return [re.sub(r"<[^>]+>", "", cell).strip() for cell in found]


def owners_today() -> str:
    return datetime.datetime.now(zoneinfo.ZoneInfo("Europe/London")).date().isoformat()


def test_the_owner_panel_lists_users_with_when_they_joined_and_last_logged_in(
    sam, owner_browser
):
    listed = {row["username"]: row for row in users_listed(owner_browser)}

    assert set(listed) == {OWNER_USERNAME, "sam"}
    assert listed["sam"]["joined"].startswith(owners_today())
    assert listed["sam"]["last_login"] == "never"  # registered, but not logged in since
    assert listed[OWNER_USERNAME]["last_login"].startswith(owners_today())
    assert listed["sam"]["status"] == "active"


def test_the_users_list_shows_a_later_login(sam, owner_browser):
    log_out(sam)
    log_in(sam, "sam", SAMS_PASSWORD)

    listed = {row["username"]: row for row in users_listed(owner_browser)}

    assert listed["sam"]["last_login"].startswith(owners_today())


def test_the_users_list_escapes_what_users_typed(client, owner_browser):
    register(client, create_invite(owner_browser), "<b>sam</b>")

    page = owner_browser.get("/owner").text

    assert "&lt;b&gt;sam&lt;/b&gt;" in page
    assert "<b>sam</b>" not in page


# Disabling and re-enabling


def disable(owner_client, username: str):
    return owner_client.post(f"/owner/users/{user_id_of(owner_client, username)}/disable")


def enable(owner_client, username: str):
    return owner_client.post(f"/owner/users/{user_id_of(owner_client, username)}/enable")


def status_of(owner_client, username: str) -> str:
    listed = {row["username"]: row for row in users_listed(owner_client)}
    return listed[username]["status"]


def test_disabling_a_user_ends_their_open_session_at_once(sam, owner_browser):
    assert is_logged_in(sam)

    disable(owner_browser, "sam")

    assert not is_logged_in(sam)
    assert sam.get("/chat/config").status_code == 401
    assert status_of(owner_browser, "sam") == "disabled"


def test_a_disabled_user_cant_log_in(sam, owner_browser):
    disable(owner_browser, "sam")
    log_out(sam)

    response = log_in(sam, "sam", SAMS_PASSWORD)

    assert "Wrong username or password." in response.text
    assert not is_logged_in(sam)


def test_a_re_enabled_user_logs_in_again(sam, owner_browser):
    disable(owner_browser, "sam")
    enable(owner_browser, "sam")

    assert not is_logged_in(sam)  # the session disabling ended stays ended
    log_in(sam, "sam", SAMS_PASSWORD)
    assert is_logged_in(sam)
    assert status_of(owner_browser, "sam") == "active"


def test_the_owner_cant_disable_themselves(owner_client):
    response = disable(owner_client, OWNER_USERNAME)

    assert "The Owner can't be disabled." in response.text
    assert is_logged_in(owner_client)
    assert status_of(owner_client, OWNER_USERNAME) == "active"


def test_the_panel_shows_no_message_made_up_in_its_link(owner_client):
    """A link to the panel can't make it say anything (e.g. a phishing message)."""
    page = owner_client.get("/owner?error=Perry+was+hacked:+log+in+at+evil.example").text

    assert "hacked" not in page
    assert 'role="alert"' not in page


def test_the_panel_offers_no_disable_button_for_the_owner(sam, owner_browser):
    page = owner_browser.get("/owner").text

    assert f"/owner/users/{user_id_of(owner_browser, 'sam')}/disable" in page
    assert f"/owner/users/{user_id_of(owner_browser, OWNER_USERNAME)}/disable" not in page


def test_other_users_cant_disable_anyone(sam, owner_browser):
    owners_id = user_id_of(owner_browser, OWNER_USERNAME)

    assert sam.post(f"/owner/users/{owners_id}/disable").status_code == 403
    assert is_logged_in(owner_browser)


def test_a_disabled_users_threads_and_usage_are_kept(sam, owner_browser, perry, prices):
    from perry import usage, users

    prices({"scripted": {"input": 1.0, "output": 1.0}})
    perry.user = users.authenticate("sam", SAMS_PASSWORD)
    perry.script(with_usage("Hello Sam!", 1000, 0))
    perry.send("Hello")
    spent = usage.today()

    disable(owner_browser, "sam")

    assert [message["content"] for message in perry.history()] == ["Hello", "Hello Sam!"]
    assert usage.today() == spent
