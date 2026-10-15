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
    body = f"""<a class="back" href="{pages.CHAT_PATH}/">&larr; Back to Perry</a>
<h1>Owner panel</h1>
<p class="hint">Times are in your time zone, {html.escape(owner.profile.time_zone)}.</p>
{pages.error_line(ERRORS.get(error, ""))}
{sections}
<script>{SCRIPT}</script>"""
    return pages.page("Owner panel", body, wide=True)


# Copy buttons put the full link on the clipboard; a form with data-confirm asks first.
SCRIPT = """
document.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-copy]");
  if (!button) return;
  const link = new URL(button.dataset.copy, location.origin).href;
  try {
    await navigator.clipboard.writeText(link);
  } catch {
    prompt("Copy this link:", link);
    return;
  }
  const label = button.textContent;
  button.textContent = "Copied \\u2713";
  button.disabled = true;
  setTimeout(() => { button.textContent = label; button.disabled = false; }, 1600);
});
document.addEventListener("submit", (event) => {
  const question = event.target.dataset.confirm;
  if (question && !confirm(question)) event.preventDefault();
});
"""


def panel_section(title: str, hint: str, content: str, button: str = "") -> str:
    """A section: its heading (with an optional button beside it), a line saying what
    it is for, and its content."""
    return f"""<section><div class="section-head"><h2>{title}</h2>{button}</div>
<p class="hint">{hint}</p>
{content}
</section>"""


def table(headings: list[str], rows: str) -> str:
    """A table under these column headings; an empty heading is the column of buttons."""
    head = "".join(f'<th class="actions">{h}</th>' if not h else f"<th>{h}</th>" for h in headings)
    return f'<div class="table-wrap"><table><tr>{head}</tr>{rows}</table></div>'


def empty(text: str) -> str:
    """What a section shows when it has nothing to list."""
    return f'<p class="empty">{text}</p>'


def local_time(moment, owner: users.User) -> str:
    """`moment` in the Owner's time zone, to the minute (the panel names the zone once)."""
    shown = moment.astimezone(zoneinfo.ZoneInfo(owner.profile.time_zone))
    return f"{shown:%Y-%m-%d %H:%M}"


# Invites


@section
def invites_section(owner: users.User) -> str:
    rows = "\n".join(
        f"""<tr><td><code data-invite="{html.escape(invite.code)}">{html.escape(invite.code)}</code></td>
  <td class="when" data-label="Expires">{local_time(invite.expires_at, owner)}</td>
  <td class="actions"><button class="small secondary" type="button"
      data-copy="/register?invite={quote(invite.code)}">Copy link</button><form method="post"
      action="{PATH}/invites/{quote(invite.code)}/withdraw"><button class="small danger"
      type="submit">Withdraw</button></form></td></tr>"""
        for invite in users.open_invites()
    )
    content = (
        table(["Invite", "Expires", ""], rows)
        if rows
        else empty("No unused Invites. Create one, then send its link to the person you invite.")
    )
    create = f"""<form method="post" action="{PATH}/invites">
  <button class="small" type="submit">Create an Invite</button></form>"""
    hint = f"Each Invite lets one person register, and expires after {users.INVITE_DAYS} days."
    return panel_section("Invites", hint, content, button=create)


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
    hint = "A disabled User can't log in, and is logged out at once. Their Threads and usage are kept."
    return panel_section("Users", hint, table(["User", "Joined", "Last login", "Status", ""], rows))


def _user_row(listing: users.UserListing, owner: users.User) -> str:
    user = listing.user
    name = html.escape(user.username)
    last_login = local_time(listing.last_login_at, owner) if listing.last_login_at else "never"
    status = "disabled" if user.disabled else "active"
    if user.is_owner:
        buttons = ""
    elif user.disabled:
        buttons = _user_button(user, "enable", "Re-enable", "secondary")
    else:
        disable_question = (
            f"Disable {user.username}? They are logged out at once, "
            "and can't log in until you re-enable them."
        )
        buttons = _user_button(user, "reset", "Reset Code", "secondary") + _user_button(
            user, "disable", "Disable", "danger", confirm=disable_question
        )
    badge = '<span class="badge">Owner</span>' if user.is_owner else ""
    return f"""<tr data-user="{user.id}" data-username="{name}">
  <td>{name}{badge}</td>
  <td class="when" data-label="Joined">{local_time(listing.joined_at, owner)}</td>
  <td class="when" data-label="Last login">{last_login}</td>
  <td><span class="status {status}">{status}</span></td>
  <td class="actions">{buttons}</td></tr>"""


def _user_button(user: users.User, action: str, label: str, kind: str, confirm: str = "") -> str:
    asks = f' data-confirm="{html.escape(confirm)}"' if confirm else ""
    return f"""<form method="post" action="{PATH}/users/{user.id}/{action}"{asks}><button
    class="small {kind}" type="submit">{label}</button></form>"""


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
  <td class="when" data-label="Expires">{local_time(reset.expires_at, owner)}</td>
  <td class="actions"><button class="small secondary" type="button"
      data-copy="/reset?code={quote(reset.code)}">Copy link</button></td></tr>"""
        for reset in users.open_reset_codes()
    )
    content = (
        table(["User", "Reset Code", "Expires", ""], rows)
        if rows
        else empty("No unused Reset Codes. Issue one from the User's row above.")
    )
    hint = (
        "A Reset Code lets a User who forgot their password choose a new one, and logs them "
        "out everywhere. Give it to them yourself; it expires after "
        f"{users.RESET_CODE_HOURS} hours."
    )
    return panel_section("Reset Codes", hint, content)
