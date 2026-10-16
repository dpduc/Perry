"""Switching Threads in the UI: the picker lists the User's Threads and can't switch mid-Turn."""


def test_threads_created_after_the_ui_was_built_are_listed_on_page_load(perry, owner):
    from logins import request_as

    from perry import app

    app.build()
    perry.script("Hi!")
    perry.send("Hello there")

    listed = app.thread_choices(request_as(owner))["choices"]

    assert perry.thread_id in [thread_id for _title, thread_id in listed]


def test_switching_threads_is_locked_while_a_turn_runs_and_unlocked_after():
    from perry import app

    locked = app.lock_threads()
    unlocked = app.unlock_threads("a-thread")

    # New thread, Thread picker, Delete thread; Delete stays off with no Thread open.
    assert [update["interactive"] for update in locked] == [False, False, False]
    assert [update["interactive"] for update in unlocked] == [True, True, True]
    assert app.unlock_threads(None)[2]["interactive"] is False


def test_every_turn_in_the_ui_locks_and_unlocks_thread_switching():
    from perry import app

    demo = app.build()
    names = [getattr(fn.fn, "__name__", "") for fn in demo.fns.values()]

    # Each way of running a Turn (typed, voice, approve, reject) locks, then unlocks.
    assert names.count("lock_threads") == 4
    assert names.count("unlock_threads") == 4
    assert "thread_choices" in names  # refreshed on every page load


def started_thread(perry, owner) -> str:
    from logins import request_as

    from perry import app

    perry.script("Hi Olive!")
    _, thread_id, *_ = app.answer(request_as(owner), {"text": "Hello", "files": []}, [], None)
    return thread_id


def test_the_delete_popup_names_the_thread_and_cancel_deletes_nothing(perry, owner):
    from logins import request_as

    from perry import app, threads

    thread_id = started_thread(perry, owner)

    popup, question, doomed = app.ask_delete(request_as(owner), thread_id)
    closed, kept = app.cancel_delete()

    assert popup["visible"] is True
    assert question == "Delete \"Hello\"? Its messages can't be brought back."
    assert doomed == thread_id
    assert closed["visible"] is False and kept is None
    assert threads.list_recent(owner) == [("Hello", thread_id)]


def test_delete_in_the_popup_removes_the_thread_and_opens_an_empty_one(perry, owner):
    from logins import request_as

    from perry import app, threads

    thread_id = started_thread(perry, owner)

    popup, doomed, chat, shown, picker, waiting, *_, button = app.delete_thread(
        request_as(owner), thread_id
    )

    assert threads.list_recent(owner) == []
    assert popup["visible"] is False and doomed is None
    assert (chat, shown, waiting) == ([], None, None)
    assert picker["choices"] == [] and picker["value"] is None
    assert button["interactive"] is False  # no Thread open to delete


def test_delete_thread_is_off_until_a_thread_is_open(perry, owner):
    from logins import request_as

    from perry import app

    thread_id = started_thread(perry, owner)

    *_, opened = app.open_thread(request_as(owner), thread_id)
    *_, fresh = app.new_thread()

    assert opened["interactive"] is True
    assert fresh["interactive"] is False
