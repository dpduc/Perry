"""The create-owner command: Perry gets its one Owner, who keeps the Threads from before."""

from logins import log_in


def create_owner(*answers: str) -> int:
    """Run `python -m perry.users create-owner`, typing `answers` at its prompts in order:
    username, password, password again, display name, time zone."""
    from perry import users

    typed = iter(answers)
    return users.main(["create-owner"], ask=lambda _: next(typed), ask_secret=lambda _: next(typed))


OWNER = ("duke", "a-long-password", "a-long-password", "Duke", "Asia/Ho_Chi_Minh")


def test_create_owner_makes_an_owner_who_can_log_in(database):
    from fastapi.testclient import TestClient

    from perry import web

    assert create_owner(*OWNER) == 0

    with TestClient(web.create_app()) as client:
        assert log_in(client, "duke", "a-long-password").headers["location"] == "/chat/"


def test_a_second_run_fails_clearly_and_keeps_the_first_owner(database, capsys):
    from perry import users

    create_owner(*OWNER)

    assert create_owner("other", "another-password", "another-password", "O", "UTC") == 1
    assert "already has an Owner" in capsys.readouterr().err
    assert users.owner().username == "duke"


def test_existing_threads_are_given_to_the_owner(database):
    from perry import threads, users

    before = [threads.create("From before there were Users"), threads.create("Another one")]

    create_owner(*OWNER)

    assert all(threads.belongs_to(thread, users.owner()) for thread in before)


def test_a_short_password_is_refused_with_the_reason(database, capsys):
    from perry import users

    assert create_owner("duke", "short", "short", "Duke", "UTC") == 1
    assert "at least 10 characters" in capsys.readouterr().err
    assert users.owner() is None


def test_mismatched_passwords_create_nothing(database, capsys):
    from perry import users

    assert create_owner("duke", "a-long-password", "a-different-one", "Duke", "UTC") == 1
    assert "don't match" in capsys.readouterr().err
    assert users.owner() is None


def test_an_unknown_time_zone_is_refused(database, capsys):
    from perry import users

    assert create_owner("duke", "a-long-password", "a-long-password", "Duke", "Mars/Olympus") == 1
    assert "time zone" in capsys.readouterr().err
    assert users.owner() is None
