"""Logging in, for tests: through the login page, or as a Gradio request for a handler.
Also the Owner panel's Invites and the register page, as a browser uses them."""

import html
import re

import gradio as gr

OWNER_USERNAME = "owner"
OWNER_PASSWORD = "owner-password-1"
SAMS_PASSWORD = "sams-password-1"


def log_in(client, username: str, password: str):
    """Submit the login page; returns the response (a redirect into Perry on success)."""
    return client.post(
        "/login", data={"username": username, "password": password}, follow_redirects=False
    )


def log_out(client):
    """Click the sidebar's Log out button; returns the response (a redirect to log in)."""
    return client.post("/logout", follow_redirects=False)


def request_as(user) -> gr.Request:
    """The request Gradio hands a handler when `user` is logged in."""
    return gr.Request(username=str(user.id))


def invites_listed(client) -> list[str]:
    """The codes of the unused Invites the Owner panel lists."""
    return re.findall(r'data-invite="([^"]+)"', client.get("/owner").text)


def create_invite(client) -> str:
    """Create an Invite from the Owner panel, as the Owner clicking its button; returns
    its code."""
    before = set(invites_listed(client))
    client.post("/owner/invites")
    (code,) = set(invites_listed(client)) - before
    return code


def register(client, invite: str, username: str, password: str = "new-password-1", **fields):
    """Submit the register page; returns the response (a redirect into Perry on success)."""
    form = {
        "invite": invite,
        "username": username,
        "password": password,
        "display_name": "New User",
        "time_zone": "Europe/Paris",
    } | fields
    return client.post("/register", data=form, follow_redirects=False)


def is_logged_in(client) -> bool:
    return client.get("/chat/", follow_redirects=False).status_code == 200


def user_id_of(owner_client, username: str) -> str:
    """The id of the User the Owner panel lists with this username."""
    page = owner_client.get("/owner").text
    name = re.escape(html.escape(username))
    (found,) = re.findall(rf'<tr data-user="(\d+)" data-username="{name}">', page)
    return found
