"""A Thread as one User sees it: opening it, sending to it, and what it keeps."""

import pytest
from fakes import call, last_message, spoken, with_usage


def test_a_first_message_refused_by_the_daily_cap_leaves_no_thread(perry, prices, monkeypatch):
    from perry import threads

    prices({"scripted": {"input": 1.0, "output": 1.0}})
    monkeypatch.setenv("PERRY_DAILY_CAP_USD", "0.01")
    perry.script(with_usage("Hi! How can I help?", 10_000))  # $0.01: the whole cap
    threads.open(perry.user, None).send("Hello")
    before = threads.list_recent(perry.user)

    thread = threads.open(perry.user, None)
    reply = thread.send("How big is my Q3 marketing budget?")

    assert reply.startswith("Perry has used today's budget of $0.01")
    assert thread.id is None  # the next message starts a Thread afresh
    assert threads.list_recent(perry.user) == before


def test_a_first_turn_that_fails_leaves_no_thread(perry):
    from perry import threads

    thread = threads.open(perry.user, None)
    with pytest.raises(AssertionError):  # the model has nothing scripted: the call fails
        thread.send("Hello")

    assert thread.id is None
    assert threads.list_recent(perry.user) == []


def test_a_voice_request_is_heard_answered_and_read_aloud_as_one_turn(perry, monkeypatch):
    from perry import threads

    # 50 tokens to transcribe and 960 here pass the ceiling only if they are one Turn.
    monkeypatch.setenv("PERRY_TURN_TOKEN_CEILING", "1000")
    perry.script(with_usage("Hi! How can I help?", 960))
    thread = threads.open(perry.user, None)

    heard = thread.hear(spoken("Hello there"))
    answer = thread.send(heard)
    speech = thread.read_aloud(heard, answer)

    assert heard.transcript == "Hello there"
    assert answer == "Hi! How can I help?"
    assert speech.audio is None  # the read-aloud call is over the Turn's ceiling
    assert "limit of 1,000 tokens" in speech.note
    assert thread.history() == [
        {"role": "user", "content": "Hello there"},
        {"role": "assistant", "content": "Hi! How can I help?"},
    ]


def propose_email_to_sam(perry):
    perry.script(
        call("ask_task_agent", request="Email Sam that I'm running late."),
        call("send_email", to="sam@example.com", subject="Late", body="Running late."),
    )


def test_a_reopened_thread_shows_its_pending_confirmation_and_approving_performs_it(
    perry, workspace
):
    from perry import threads

    propose_email_to_sam(perry)
    first = threads.open(perry.user, None)
    first.send("Tell Sam I'm running late")

    thread = threads.open(perry.user, first.id)
    pending = thread.pending_confirmation()
    perry.script(last_message, last_message)  # Task Agent reports, Supervisor passes it on
    reply = thread.approve({"subject": "Late, sorry"})

    assert pending.fields["to"] == "sam@example.com"
    assert [e.subject for e in workspace.sent] == ["Late, sorry"]
    assert "Sent" in reply
    assert thread.pending_confirmation() is None


def test_rejecting_a_confirmation_performs_nothing_and_perry_revises(perry, workspace):
    from perry import threads

    propose_email_to_sam(perry)
    thread = threads.open(perry.user, None)
    thread.send("Tell Sam I'm running late")
    perry.script(last_message, "Okay, I won't send it.")

    reply = thread.reject("Don't, I'll call him")

    assert workspace.sent == []
    assert reply == "Okay, I won't send it."


def test_another_users_thread_opens_as_if_it_did_not_exist(perry):
    from perry import threads, users

    sam = users.create_user("sam", "sam-password-1", "Sam", "UTC")
    perry.script("Hi Olive!")
    owners = threads.open(perry.user, None)
    owners.send("Hello")

    for thread_id in (owners.id, "no-such-thread"):
        with pytest.raises(threads.ThreadNotFound, match="doesn't exist"):
            threads.open(sam, thread_id)


def ledger_rows(database, thread_id: str) -> int:
    with database.pool().connection() as conn:
        row = conn.execute(
            "SELECT count(*) AS n FROM usage_ledger WHERE thread_id = %s", (thread_id,)
        ).fetchone()
    return row["n"]


def test_deleting_a_thread_removes_it_and_its_conversation_but_keeps_its_usage(
    database, perry
):
    from perry import supervisor, threads

    perry.script("Hi Olive!")
    thread = threads.open(perry.user, None)
    thread.send("Hello")
    spent = ledger_rows(database, thread.id)

    threads.delete(perry.user, thread.id)

    assert threads.list_recent(perry.user) == []
    with pytest.raises(threads.ThreadNotFound):
        threads.open(perry.user, thread.id)
    assert supervisor.thread_history(thread.id) == []
    assert spent > 0 and ledger_rows(database, thread.id) == spent


def test_another_users_thread_cant_be_deleted(perry):
    from perry import threads, users

    sam = users.create_user("sam", "sam-password-1", "Sam", "UTC")
    perry.script("Hi Olive!")
    owners = threads.open(perry.user, None)
    owners.send("Hello")

    for thread_id in (owners.id, "no-such-thread"):
        with pytest.raises(threads.ThreadNotFound):
            threads.delete(sam, thread_id)

    assert threads.list_recent(perry.user) == [("Hello", owners.id)]
    assert threads.open(perry.user, owners.id).history()[-1]["content"] == "Hi Olive!"


def test_a_thread_waiting_on_a_confirmation_is_deleted_and_its_write_action_never_runs(
    perry, workspace
):
    from perry import supervisor, threads

    propose_email_to_sam(perry)
    thread = threads.open(perry.user, None)
    thread.send("Tell Sam I'm running late")

    threads.delete(perry.user, thread.id)

    assert supervisor.pending_confirmation(thread.id) is None
    with pytest.raises(threads.ThreadNotFound):
        thread.approve()
    assert workspace.sent == []
