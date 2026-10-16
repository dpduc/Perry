"""The create-owner command: Perry gets its one Owner, who keeps the Threads from before."""

from logins import log_in

PASSWORD = "a-long-password"


def create_owner(*answers, prompts: list[str] | None = None) -> int:
    """Run `python -m perry.users create-owner`, typing `answers` at its prompts in order:
    username, password, password again, display name, time zone. An answer that is an
    exception (EOFError, KeyboardInterrupt) is raised at its prompt instead of typed;
    `prompts` collects the prompts shown. Fails if Perry asks for more than is typed."""
    from perry import users

    typed = iter(answers)

    def prompt(shown: str) -> str:
        if prompts is not None:
            prompts.append(shown)
        answer = next(typed, None)
        assert answer is not None, f"Perry asked more than expected: {shown!r}"
        if isinstance(answer, BaseException):
            raise answer
        return answer

    return users.main(["create-owner"], ask=prompt, ask_secret=prompt)


def machine_time_zone(monkeypatch, name: str):
    """Make `name` what this machine says its time zone is."""
    import tzlocal

    monkeypatch.setattr(tzlocal, "get_localzone_name", lambda: name)


OWNER = ("duke", PASSWORD, PASSWORD, "Duke", "Asia/Ho_Chi_Minh")


def test_create_owner_makes_an_owner_who_can_log_in(database):
    from fastapi.testclient import TestClient

    from perry import web

    assert create_owner(*OWNER) == 0

    with TestClient(web.create_app()) as client:
        assert log_in(client, "duke", PASSWORD).headers["location"] == "/chat/"


def test_a_second_run_fails_clearly_and_keeps_the_first_owner(database, capsys):
    from perry import users

    create_owner(*OWNER)

    assert create_owner("other", "another-password", "another-password", "O", "UTC") == 1
    assert "already has an Owner" in capsys.readouterr().err
    assert users.owner().username == "duke"


def test_existing_threads_are_given_to_the_owner(database):
    from perry import threads, users

    # Threads from before there were Users belong to no one.
    before = ["old-thread-1", "old-thread-2"]
    with database.pool().connection() as conn:
        for thread in before:
            conn.execute("INSERT INTO threads (id, title) VALUES (%s, 'Old')", (thread,))

    create_owner(*OWNER)

    assert all(threads.belongs_to(thread, users.owner()) for thread in before)


def test_a_short_password_is_asked_again_at_once(database, capsys):
    from fastapi.testclient import TestClient

    from perry import web

    prompts = []
    # "short" is refused before "Password again", and the username is not asked twice.
    assert create_owner("duke", "short", PASSWORD, PASSWORD, "Duke", "UTC", prompts=prompts) == 0

    assert "at least 10 characters" in capsys.readouterr().err
    assert prompts.count("Username: ") == 1
    assert prompts.count("Password again: ") == 1
    with TestClient(web.create_app()) as client:
        assert log_in(client, "duke", PASSWORD).headers["location"] == "/chat/"


def test_mismatched_passwords_ask_both_passwords_again_but_not_the_username(database, capsys):
    from fastapi.testclient import TestClient

    from perry import web

    prompts = []
    assert create_owner(
        "duke", PASSWORD, "a-different-one", "second-password", "second-password", "Duke", "UTC",
        prompts=prompts,
    ) == 0

    assert "don't match" in capsys.readouterr().err
    assert prompts.count("Username: ") == 1
    assert prompts.count("Password again: ") == 2
    with TestClient(web.create_app()) as client:
        assert log_in(client, "duke", "second-password").headers["location"] == "/chat/"


def test_a_blank_username_is_asked_again(database, capsys):
    from perry import users

    prompts = []
    assert create_owner("  ", "duke", PASSWORD, PASSWORD, "Duke", "UTC", prompts=prompts) == 0

    assert "username" in capsys.readouterr().err.lower()
    assert prompts.count("Username: ") == 2
    assert users.owner().username == "duke"


def test_enter_accepts_the_username_as_the_display_name(database):
    from perry import users

    prompts = []
    create_owner("batman", PASSWORD, PASSWORD, "", "UTC", prompts=prompts)

    assert "Display name [batman]: " in prompts
    assert users.owner().profile.display_name == "batman"


def test_enter_accepts_this_machines_time_zone(database, monkeypatch):
    from perry import users

    machine_time_zone(monkeypatch, "Asia/Ho_Chi_Minh")
    prompts = []
    create_owner("duke", PASSWORD, PASSWORD, "Duke", "", prompts=prompts)

    assert "Time zone [Asia/Ho_Chi_Minh]: " in prompts
    assert users.owner().profile.time_zone == "Asia/Ho_Chi_Minh"


def test_without_an_iana_name_for_this_machine_there_is_no_default(database, monkeypatch, capsys):
    from perry import users

    machine_time_zone(monkeypatch, "SE Asia Standard Time")  # what Windows says
    prompts = []
    assert create_owner("duke", PASSWORD, PASSWORD, "Duke", "", "Europe/London",
                        prompts=prompts) == 0

    assert not any("[" in p for p in prompts if p.startswith("Time zone"))
    assert "Give a time zone" in capsys.readouterr().err
    assert users.owner().profile.time_zone == "Europe/London"


def test_an_unknown_time_zone_is_asked_again_with_close_matches(database, monkeypatch, capsys):
    from perry import users

    machine_time_zone(monkeypatch, "Asia/Bangkok")
    assert create_owner("duke", PASSWORD, PASSWORD, "Duke", "Asia/Ho_Chi_Mihn", "") == 0

    err = capsys.readouterr().err
    assert "Did you mean Asia/Ho_Chi_Minh" in err
    assert "Region/City" in err
    assert users.owner().profile.time_zone == "Asia/Bangkok"  # Enter took the default


def test_hanoi_is_not_a_time_zone_and_the_form_is_explained(database, capsys):
    assert create_owner("duke", PASSWORD, PASSWORD, "Duke", "Hanoi", "Asia/Ho_Chi_Minh") == 0

    err = capsys.readouterr().err
    assert "Hanoi" in err
    assert "Region/City" in err and "Asia/Ho_Chi_Minh" in err


def test_an_unknown_time_zone_offers_this_machines_as_the_way_out(database, monkeypatch, capsys):
    from perry import users

    machine_time_zone(monkeypatch, "Asia/Bangkok")
    assert create_owner("duke", PASSWORD, PASSWORD, "Duke", "Hanoi", "") == 0

    assert "Press Enter for Asia/Bangkok, this machine's time zone." in capsys.readouterr().err
    assert users.owner().profile.time_zone == "Asia/Bangkok"


def test_at_most_three_close_matches_are_suggested(database, capsys):
    create_owner("duke", PASSWORD, PASSWORD, "Duke", "America/Argentin", "UTC")

    (suggestion,) = [line for line in capsys.readouterr().err.splitlines() if "Did you mean" in line]
    suggested = suggestion.split("Did you mean ")[1].split("?")[0].split(" or ")
    assert 1 <= len(suggested) <= 3


def test_a_time_zone_is_matched_ignoring_case(database):
    from perry import users

    create_owner("duke", PASSWORD, PASSWORD, "Duke", "asia/ho_chi_minh")

    assert users.owner().profile.time_zone == "Asia/Ho_Chi_Minh"


def test_ctrl_c_at_a_prompt_ends_cleanly_and_creates_nothing(database, capsys):
    from perry import users

    assert create_owner("duke", PASSWORD, PASSWORD, "Duke", KeyboardInterrupt()) == 1

    assert "Nothing was created." in capsys.readouterr().err
    assert users.owner() is None


def test_end_of_input_ends_cleanly_and_creates_nothing(database, capsys):
    from perry import users

    assert create_owner(EOFError()) == 1

    assert "Nothing was created." in capsys.readouterr().err
    assert users.owner() is None
