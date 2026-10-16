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
    unlocked_on_thread = app.unlock_threads("t1")
    unlocked_on_new = app.unlock_threads(None)

    assert [update["interactive"] for update in locked] == [False, False, False]
    assert [update["interactive"] for update in unlocked_on_thread] == [True, True, True]
    assert [update["interactive"] for update in unlocked_on_new] == [True, True, False]


def test_every_turn_in_the_ui_locks_and_unlocks_thread_switching():
    from perry import app

    demo = app.build()
    names = [getattr(fn.fn, "__name__", "") for fn in demo.fns.values()]

    # Each way of running a Turn (typed, voice, approve, reject) locks, then unlocks.
    assert names.count("lock_threads") == 4
    assert names.count("unlock_threads") == 4
    assert "thread_choices" in names  # refreshed on every page load


def test_cancelling_delete_keeps_the_thread_in_the_picker(perry, owner):
    from logins import request_as

    from perry import app

    perry.script("Hi!")
    _, thread_id, *_ = app.answer(request_as(owner), {"text": "Hello there", "files": []}, [], None)
    popup, note = app.delete_prompt(request_as(owner), thread_id)
    hidden, _ = app.cancel_delete_thread()
    listed = [thread for _title, thread in app.thread_choices(request_as(owner))["choices"]]

    assert popup["visible"] and "Delete" in note
    assert not hidden["visible"]
    assert thread_id in listed


def test_deleting_removes_the_thread_and_opens_a_new_empty_one(perry, owner):
    from logins import request_as

    from perry import app

    perry.script("Hi!")
    _, thread_id, *_ = app.answer(request_as(owner), {"text": "Hello there", "files": []}, [], None)
    history, opened, picker, *_ = app.delete_current_thread(request_as(owner), thread_id)

    assert history == []
    assert opened is None
    assert picker["value"] is None
    assert thread_id not in [thread for _title, thread in picker["choices"]]
