"""The chat UI's handlers learn the User from Gradio's request, never from the browser."""

import gradio as gr
import pytest
from logins import request_as

MESSAGE = {"text": "Hello there", "files": []}


def test_a_logged_in_user_sends_a_message_and_gets_the_answer(perry, owner):
    from perry import app

    perry.script("Hi Olive!")

    history, thread_id, *_ = app.answer(request_as(owner), MESSAGE, [], None)

    assert history[-1] == {"role": "assistant", "content": "Hi Olive!"}
    assert thread_id is not None


@pytest.mark.parametrize(
    "request_",
    [gr.Request(), gr.Request(username="999999"), gr.Request(username="not a user")],
    ids=["no user", "unknown user", "garbage"],
)
def test_a_message_without_a_logged_in_user_is_refused_and_runs_no_turn(perry, request_):
    from perry import app

    with pytest.raises(gr.Error, match="log in"):
        app.answer(request_, MESSAGE, [], None)

    assert perry.model.calls == []


def test_the_sidebar_says_who_is_logged_in(owner):
    from perry import app

    assert "Olive Owner" in app.logged_in_as(request_as(owner))


def test_the_ui_shows_the_user_behind_the_session(owner_client):
    """End to end over HTTP: Gradio hands the handler the User the session cookie names."""
    started = owner_client.post("/chat/gradio_api/call/logged_in_as", json={"data": []})
    event_id = started.json()["event_id"]

    stream = owner_client.get(f"/chat/gradio_api/call/logged_in_as/{event_id}").text

    assert "event: complete" in stream
    assert "Olive Owner" in stream
