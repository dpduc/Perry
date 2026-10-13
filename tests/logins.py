"""Logging in, for tests: through the login page, or as a Gradio request for a handler."""

import gradio as gr

OWNER_USERNAME = "owner"
OWNER_PASSWORD = "owner-password-1"


def log_in(client, username: str, password: str):
    """Submit the login page; returns the response (a redirect into Perry on success)."""
    return client.post(
        "/login", data={"username": username, "password": password}, follow_redirects=False
    )


def request_as(user) -> gr.Request:
    """The request Gradio hands a handler when `user` is logged in."""
    return gr.Request(username=str(user.id))
