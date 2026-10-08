"""Switching Threads in the UI: the picker lists every Thread and can't switch mid-Turn."""


def test_threads_created_after_the_ui_was_built_are_listed_on_page_load(perry):
    from perry import app

    app.build()
    perry.script("Hi!")
    perry.send("Hello there")

    listed = app.thread_choices()["choices"]

    assert perry.thread_id in [thread_id for _title, thread_id in listed]


def test_switching_threads_is_locked_while_a_turn_runs_and_unlocked_after():
    from perry import app

    locked = app.lock_threads()
    unlocked = app.unlock_threads()

    assert [update["interactive"] for update in locked] == [False, False]
    assert [update["interactive"] for update in unlocked] == [True, True]


def test_every_turn_in_the_ui_locks_and_unlocks_thread_switching():
    from perry import app

    demo = app.build()
    names = [getattr(fn.fn, "__name__", "") for fn in demo.fns.values()]

    # Each way of running a Turn (typed, voice, approve, reject) locks, then unlocks.
    assert names.count("lock_threads") == 4
    assert names.count("unlock_threads") == 4
    assert "thread_choices" in names  # refreshed on every page load
