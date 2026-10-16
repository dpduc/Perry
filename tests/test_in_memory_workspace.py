"""The in-memory Workspace's email search understands the Gmail syntax the Task Agent is
told to use, so tests and evals see the misses Gmail would give, and no others."""

from datetime import UTC, datetime, timedelta

from perry.workspace import Email, InMemoryWorkspace

NOW = datetime.now(UTC)


def inbox(*emails: Email) -> InMemoryWorkspace:
    workspace = InMemoryWorkspace()
    workspace.emails = list(emails)
    return workspace


LUNCH = Email("Priya Shah <priya@example.com>", "Lunch tomorrow?", "Noodles?", NOW, id="new")
OLD = Email("Priya Shah <priya@example.com>", "Lunch", "Pizza?", NOW - timedelta(days=60), id="old")


def found(workspace: InMemoryWorkspace, query: str) -> list[str]:
    return [email.id for email in workspace.search_email(query)]


def test_newer_than_keeps_only_recent_email():
    workspace = inbox(LUNCH, OLD)

    assert found(workspace, "from:priya lunch newer_than:30d") == ["new"]
    assert found(workspace, "from:priya older_than:30d") == ["old"]


def test_subject_matches_a_word_of_the_subject_only():
    workspace = inbox(LUNCH, OLD)

    assert found(workspace, "subject:tomorrow") == ["new"]
    assert found(workspace, "subject:noodles") == []


def test_an_operator_it_does_not_know_is_ignored_not_matched_as_a_word():
    assert found(inbox(LUNCH), "lunch in:inbox") == ["new"]


def test_a_plain_word_also_matches_the_sender_as_in_gmail():
    assert found(inbox(LUNCH), "priya lunch") == ["new"]


def test_or_and_parentheses_match_either_term():
    workspace = inbox(LUNCH, OLD)

    assert found(workspace, "from:priya (noodles OR pizza)") == ["new", "old"]
    assert found(workspace, "noodles OR sushi newer_than:30d") == ["new"]
