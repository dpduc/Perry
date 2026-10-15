"""Threads are private to their User (ADR-0004): each User sees and acts on only their own."""

import gradio as gr
import pytest
from fakes import call, last_message, spoken
from logins import request_as

from perry.supervisor import Approve

MESSAGE = {"text": "Hello there", "files": []}


def propose_email_to_ana(perry):
    perry.script(
        call("ask_task_agent", request="Email Ana that I'm late."),
        call("send_email", to="ana@example.com", subject="Late", body="Running late."),
    )


@pytest.fixture
def sam(database):
    """A User other than the Owner."""
    from perry import users

    return users.create_user("sam", "sam-password-1", "Sam", "UTC")


def test_the_thread_picker_lists_only_the_users_own_threads(perry, owner, sam):
    from perry import app

    perry.script("Hi Olive!", "Hi Sam!")
    app.answer(request_as(owner), MESSAGE, [], None)
    _, sams_thread, *_ = app.answer(request_as(sam), MESSAGE, [], None)

    listed = [thread_id for _title, thread_id in app.thread_choices(request_as(sam))["choices"]]

    assert listed == [sams_thread]


def test_opening_another_users_thread_behaves_as_if_it_does_not_exist(perry, owner, sam):
    from perry import app

    perry.script("Hi Olive!")
    _, owners_thread, *_ = app.answer(request_as(owner), MESSAGE, [], None)

    as_sam = app.open_thread(request_as(sam), owners_thread)
    nothing = app.open_thread(request_as(sam), "no-such-thread")

    history, thread_id, waiting, *_ = as_sam
    assert (history, thread_id, waiting) == ([], None, None)
    assert as_sam[:3] == nothing[:3]


def test_the_owner_opens_their_own_thread(perry, owner):
    from perry import app

    perry.script("Hi Olive!")
    _, thread, *_ = app.answer(request_as(owner), MESSAGE, [], None)

    history, thread_id, *_ = app.open_thread(request_as(owner), thread)

    assert thread_id == thread
    assert history[-1] == {"role": "assistant", "content": "Hi Olive!"}


def test_a_turn_on_another_users_thread_is_refused_and_leaves_it_untouched(perry, sam):
    from perry import threads

    perry.script("Hi Olive!")
    perry.send("Hello there")
    before = perry.history()
    owner, perry.user = perry.user, sam

    with pytest.raises(threads.ThreadNotFound):
        perry.send("What did Olive ask?")

    perry.user = owner
    assert perry.history() == before
    assert len(perry.model.calls) == 1  # only the Owner's Turn called the model


def test_a_decision_on_another_users_confirmation_is_refused_and_does_nothing(
    perry, workspace, sam
):
    from perry import threads
    from perry.supervisor import Confirmation

    propose_email_to_ana(perry)
    assert isinstance(perry.send("Email Ana that I'm late"), Confirmation)
    perry.user = sam
    perry.script(last_message, last_message)

    with pytest.raises(threads.ThreadNotFound):
        perry.resume(Approve())

    assert workspace.sent == []


def test_sending_on_another_users_thread_is_refused_in_the_ui(perry, owner, sam):
    from perry import app

    perry.script("Hi Olive!")
    _, owners_thread, *_ = app.answer(request_as(owner), MESSAGE, [], None)

    with pytest.raises(gr.Error, match="doesn't exist"):
        app.answer(request_as(sam), MESSAGE, [], owners_thread)

    assert len(perry.model.calls) == 1


@pytest.mark.parametrize("decide", ["approve", "reject"])
def test_deciding_on_another_users_confirmation_is_refused_in_the_ui(
    perry, workspace, sam, decide
):
    from perry import app

    propose_email_to_ana(perry)
    waiting = perry.send("Email Ana that I'm late")
    perry.script(last_message, last_message)
    as_sam = request_as(sam)

    with pytest.raises(gr.Error, match="doesn't exist"):
        if decide == "approve":
            app.approve(as_sam, perry.thread_id, waiting, [], *[""] * len(app.CARD_FIELDS))
        else:
            app.reject(as_sam, perry.thread_id, waiting, [], "Say it nicer")

    assert workspace.sent == []
    perry.resume(Approve())  # the Confirmation still waits for the Owner
    assert len(workspace.sent) == 1


def test_voice_on_another_users_thread_is_refused_before_transcribing(perry, sam, tmp_path):
    from perry import app, usage

    perry.script("Hi Olive!")
    perry.send("Hello there")
    used = usage.today().tokens
    recording = tmp_path / "request.wav"
    recording.write_bytes(spoken("What did Olive ask?"))

    with pytest.raises(gr.Error, match="doesn't exist"):
        app.hear(request_as(sam), str(recording), [], perry.thread_id)

    assert usage.today().tokens == used  # no speech-to-text call was made


def test_reading_aloud_on_another_users_thread_is_refused(perry, sam):
    from perry import app, threads, usage

    perry.script("Hi Olive!")
    perry.send("Hello there")
    used = usage.today().tokens
    answered = [{"role": "assistant", "content": "Hi Olive!"}]

    with pytest.raises(gr.Error, match="doesn't exist"):
        app.read_aloud(request_as(sam), threads.Heard("Hi", "t1"), answered, perry.thread_id)

    assert usage.today().tokens == used  # no text-to-speech call was made


def ledger_users(database) -> list[tuple[str, int | None]]:
    """(kind, user id) of every model call in the Usage Ledger, oldest first."""
    with database.pool().connection() as conn:
        rows = conn.execute("SELECT kind, user_id FROM usage_ledger ORDER BY id").fetchall()
    return [(row["kind"], row["user_id"]) for row in rows]


def test_the_usage_ledger_records_whose_turn_each_model_call_was(database, perry, owner, sam):
    perry.script("Hi Olive!", "Hi Sam!")
    perry.send("Hello there")
    perry.user, perry.thread_id = sam, None
    perry.send("Hello there")

    assert ledger_users(database) == [("chat", owner.id), ("chat", sam.id)]


def test_a_voice_turns_speech_calls_are_recorded_as_the_users(database, perry, sam):
    perry.user = sam
    perry.script("Hi Sam!")

    perry.say(spoken("Hello there"))

    assert ledger_users(database) == [
        ("speech_to_text", sam.id),
        ("chat", sam.id),
        ("text_to_speech", sam.id),
    ]


def test_model_calls_outside_a_turn_belong_to_no_user(database, perry, tmp_path):
    from perry import ingest

    notes = tmp_path / "notes.md"
    notes.write_text("# Q3\nThe Q3 marketing budget is 40,000 EUR.", encoding="utf-8")

    ingest.ingest_files([notes])

    assert ("embedding", None) in ledger_users(database)
    assert {user for _kind, user in ledger_users(database)} == {None}
