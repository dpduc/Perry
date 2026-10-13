"""The Owner panel at /owner: the pages only the Owner can open.

The panel is a list of sections, each a function that takes the Owner and returns its
HTML, shown in the order they are defined here. To add one:

    @section
    def users_section(owner: users.User) -> str:
        return "<section><h2>Users</h2>...</section>"

    @router.post("/users/{user_id}/disable")
    def disable(user_id: int):
        try:
            users.disable(user_id)
        except users.Refused:
            return back_to_panel(OWNER_CANT_BE_DISABLED)
        return back_to_panel()

Every route on `router` is Owner-only: without a session it sends you to /login, and
any other User is refused. Forms post to a route here, which does the work and calls
`back_to_panel()`, or `back_to_panel(error)` with one of the keys in ERRORS to show
its message at the top. The panel never shows text from its URL: an unknown key shows
nothing.
"""

import html
import zoneinfo
from collections.abc import Callable
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from perry import pages, users

PATH = "/owner"
NOT_THE_OWNER = "Only the Owner can open the Owner panel."

Section = Callable[[users.User], str]
SECTIONS: list[Section] = []


def section(render: Section) -> Section:
    """Add `render` to the panel, after the sections defined before it."""
    SECTIONS.append(render)
    return render


class _NotLoggedIn(Exception):
    pass


class _NotOwner(Exception):
    pass


def owner_only(request: Request) -> users.User:
    """The Owner making the request; anyone else doesn't get past this."""
    user = pages.current_user(request)
    if user is None:
        raise _NotLoggedIn
    if not user.is_owner:
        raise _NotOwner
    return user


router = APIRouter(prefix=PATH, dependencies=[Depends(owner_only)])


def install(app: FastAPI) -> None:
    """Serve the Owner panel on `app`."""
    app.include_router(router)
    app.add_exception_handler(
        _NotLoggedIn, lambda request, exc: RedirectResponse(pages.LOGIN_PATH, status_code=303)
    )
    app.add_exception_handler(
        _NotOwner,
        lambda request, exc: pages.page(
            "Owner panel", pages.error_line(NOT_THE_OWNER), status_code=403
        ),
    )


# The messages the panel can show at the top, by the key in its URL.
OWNER_CANT_BE_DISABLED = "owner-cant-be-disabled"
NO_RESET_CODE_FOR_DISABLED = "no-reset-code-for-disabled"
ERRORS = {
    OWNER_CANT_BE_DISABLED: users.OWNER_CANT_BE_DISABLED,
    NO_RESET_CODE_FOR_DISABLED: users.NO_RESET_CODE_FOR_DISABLED,
}


def back_to_panel(error: str = "") -> RedirectResponse:
    """Back to the panel after a form, showing the message for `error` (a key in ERRORS)
    if there is one."""
    query = f"?error={quote(error)}" if error else ""
    return RedirectResponse(PATH + query, status_code=303)


@router.get("", response_class=HTMLResponse)
def panel(owner: Annotated[users.User, Depends(owner_only)], error: str = ""):
    sections = "\n".join(render(owner) for render in SECTIONS)
    body = f"""<h1>Owner panel</h1>
<p><a href="{pages.CHAT_PATH}/">Back to Perry</a></p>
{pages.error_line(ERRORS.get(error, ""))}
{sections}"""
    return pages.page("Owner panel", body, wide=True)


def local_time(moment, owner: users.User) -> str:
    """`moment` in the Owner's time zone, to the minute."""
    shown = moment.astimezone(zoneinfo.ZoneInfo(owner.profile.time_zone))
    return f"{shown:%Y-%m-%d %H:%M} ({html.escape(owner.profile.time_zone)})"


# Invites


@section
def invites_section(owner: users.User) -> str:
    rows = "\n".join(
        f"""<tr><td><code data-invite="{html.escape(invite.code)}">{html.escape(invite.code)}</code></td>
  <td><a href="/register?invite={quote(invite.code)}">register link</a></td>
  <td>{local_time(invite.expires_at, owner)}</td>
  <td><form method="post" action="{PATH}/invites/{quote(invite.code)}/withdraw">
    <button class="small" type="submit">Withdraw</button></form></td></tr>"""
        for invite in users.open_invites()
    )
    table = (
        f"<table><tr><th>Invite</th><th></th><th>Expires</th><th></th></tr>{rows}</table>"
        if rows
        else "<p>No unused Invites.</p>"
    )
    return f"""<section><h2>Invites</h2>
<p>Each Invite lets one person register, and expires after {users.INVITE_DAYS} days.</p>
{table}
<form method="post" action="{PATH}/invites"><button type="submit">Create an Invite</button></form>
</section>"""


@router.post("/invites")
def create_invite():
    users.create_invite()
    return back_to_panel()


@router.post("/invites/{code}/withdraw")
def withdraw_invite(code: str):
    users.withdraw_invite(code)
    return back_to_panel()


# Users


@section
def users_section(owner: users.User) -> str:
    rows = "\n".join(_user_row(listing, owner) for listing in users.all_users())
    return f"""<section><h2>Users</h2>
<p>A disabled User can't log in, and is logged out at once. Their Threads and usage are kept.</p>
<table><tr><th>Username</th><th>Joined</th><th>Last login</th><th>Status</th><th></th></tr>
{rows}</table>
</section>"""


def _user_row(listing: users.UserListing, owner: users.User) -> str:
    user = listing.user
    last_login = local_time(listing.last_login_at, owner) if listing.last_login_at else "never"
    if user.is_owner:
        status, action = "active", "Owner"
    elif user.disabled:
        status, action = "disabled", _user_button(user, "enable", "Re-enable")
    else:
        status = "active"
        action = _user_button(user, "disable", "Disable") + _user_button(
            user, "reset", "Reset Code"
        )
    return f"""<tr data-user="{user.id}">
  <td>{html.escape(user.username)}</td>
  <td>{local_time(listing.joined_at, owner)}</td>
  <td>{last_login}</td>
  <td>{status}</td>
  <td>{action}</td></tr>"""


def _user_button(user: users.User, action: str, label: str) -> str:
    return f"""<form method="post" action="{PATH}/users/{user.id}/{action}">
    <button class="small" type="submit">{label}</button></form>"""


@router.post("/users/{user_id}/disable")
def disable_user(user_id: int):
    try:
        users.disable(user_id)
    except users.Refused:  # the only refusal: the Owner
        return back_to_panel(OWNER_CANT_BE_DISABLED)
    return back_to_panel()


@router.post("/users/{user_id}/enable")
def enable_user(user_id: int):
    users.enable(user_id)
    return back_to_panel()


@router.post("/users/{user_id}/reset")
def issue_reset_code(user_id: int):
    try:
        users.create_reset_code(user_id)
    except users.Refused:  # the only refusal: a disabled User
        return back_to_panel(NO_RESET_CODE_FOR_DISABLED)
    return back_to_panel()


# Reset Codes


@section
def reset_codes_section(owner: users.User) -> str:
    rows = "\n".join(
        f"""<tr data-reset="{html.escape(reset.code)}">
  <td>{html.escape(reset.username)}</td>
  <td><code>{html.escape(reset.code)}</code></td>
  <td><a href="/reset?code={quote(reset.code)}">reset link</a></td>
  <td>{local_time(reset.expires_at, owner)}</td></tr>"""
        for reset in users.open_reset_codes()
    )
    table = (
        f"<table><tr><th>User</th><th>Reset Code</th><th></th><th>Expires</th></tr>{rows}</table>"
        if rows
        else "<p>No unused Reset Codes.</p>"
    )
    return f"""<section><h2>Reset Codes</h2>
<p>A Reset Code lets a User who forgot their password choose a new one, and logs them out
everywhere. Give it to them yourself; it expires after {users.RESET_CODE_HOURS} hours.</p>
{table}
</section>"""
