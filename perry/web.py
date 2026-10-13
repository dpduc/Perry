"""The served app: Perry's login and logout pages, with the chat UI (perry.app) mounted
behind them at /chat. Run with: uv run python -m perry.web

Every request under /chat, page or Gradio API, needs a session: pages without one are
sent to the login page, API calls are refused. Handlers learn the User from Gradio's
request (see `perry.app.user_of`), never from anything the browser holds.
"""

import html
from dataclasses import dataclass

import gradio as gr
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from perry import app as chat_ui
from perry import charts, owner_panel, pages, users
from perry.pages import PAGE, SESSION_COOKIE

CHAT_PATH = "/chat"
HOST = "127.0.0.1"
PORT = 7860


def create_app(blocks: gr.Blocks | None = None, *, secure_cookies: bool = False) -> FastAPI:
    """The served app, with `blocks` (by default the chat UI) mounted at /chat.

    Refuses to start if there is no Owner or no session secret. `secure_cookies` marks
    the session cookie Secure, for serving over HTTPS."""
    users.require_owner()
    users.session_secret()
    served = FastAPI()

    @served.middleware("http")
    async def require_session(request: Request, call_next):
        path = request.url.path
        if (path == CHAT_PATH or path.startswith(CHAT_PATH + "/")) and _user(request) is None:
            if path.rstrip("/") == CHAT_PATH:
                return RedirectResponse("/login", status_code=303)
            return JSONResponse({"error": "Not logged in"}, status_code=401)
        return await call_next(request)

    @served.get("/")
    def home(request: Request):
        return RedirectResponse(f"{CHAT_PATH}/" if _user(request) else "/login", status_code=303)

    @served.get("/login", response_class=HTMLResponse)
    def login_page(request: Request):
        if _user(request) is not None:
            return RedirectResponse(f"{CHAT_PATH}/", status_code=303)
        return _login_page()

    @served.post("/login")
    def log_in(username: str = Form(""), password: str = Form("")):
        try:
            user = users.authenticate(username, password)
        except users.Refused as refused:
            return _login_page(str(refused), username, status_code=401)
        return logged_in(user)

    def logged_in(user: users.User) -> RedirectResponse:
        """Into Perry, with a new session for `user`."""
        response = RedirectResponse(f"{CHAT_PATH}/", status_code=303)
        response.set_cookie(
            SESSION_COOKIE,
            users.start_session(user),
            max_age=users.SESSION_SECONDS,
            httponly=True,
            secure=secure_cookies,
            samesite="lax",
        )
        return response

    @served.get("/register", response_class=HTMLResponse)
    def register_page(invite: str = ""):
        return _register_page(RegisterForm(invite=invite))

    @served.post("/register")
    def register(
        invite: str = Form(""),
        username: str = Form(""),
        password: str = Form(""),
        display_name: str = Form(""),
        time_zone: str = Form(""),
    ):
        form = RegisterForm(invite, username, display_name, time_zone)
        try:
            user = users.register(invite, username, password, display_name, time_zone.strip())
        except users.Refused as refused:
            return _register_page(form, str(refused), status_code=400)
        return logged_in(user)

    @served.api_route("/logout", methods=["GET", "POST"])
    def log_out():
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(SESSION_COOKIE, httponly=True, secure=secure_cookies, samesite="lax")
        return response

    owner_panel.install(served)

    @served.get("/profile", response_class=HTMLResponse)
    def profile_page(request: Request):
        user = _user(request)
        if user is None:
            return RedirectResponse("/login", status_code=303)
        return _profile_page(user.profile)

    @served.post("/profile", response_class=HTMLResponse)
    def save_profile(
        request: Request,
        display_name: str = Form(""),
        time_zone: str = Form(""),
        about: str = Form(""),
    ):
        user = _user(request)
        if user is None:
            return RedirectResponse("/login", status_code=303)
        try:
            saved = users.update_profile(user, display_name, time_zone, about)
        except users.Refused as refused:
            typed = users.Profile(display_name, time_zone, about)
            return _profile_page(typed, error=str(refused), status_code=400)
        return _profile_page(saved.profile, saved=True)

    return gr.mount_gradio_app(
        served,
        blocks if blocks is not None else chat_ui.build(),
        path=CHAT_PATH,
        # Behind ngrok (which sends X-Forwarded-Host), Gradio would otherwise drop /chat
        # from the links it builds, e.g. to a Chart in an answer.
        root_path=CHAT_PATH,
        auth_dependency=_gradio_user,
        allowed_paths=[str(charts.directory())],
        footer_links=[],
    )


def _user(request: Request) -> users.User | None:
    return pages.current_user(request)


def _gradio_user(request: Request) -> str | None:
    """What Gradio knows the User by (its request's `username`): the User's id."""
    user = _user(request)
    return str(user.id) if user else None


def _login_page(error: str = "", username: str = "", status_code: int = 200) -> HTMLResponse:
    shown_error = f'<p class="error" role="alert">{html.escape(error)}</p>' if error else ""
    body = f"""<h1>Log in to Perry</h1>
{shown_error}
<form method="post" action="/login">
  <label>Username <input name="username" autocomplete="username" required autofocus
         value="{html.escape(username)}"></label>
  <label>Password <input name="password" type="password" autocomplete="current-password"
         required></label>
  <button type="submit">Log in</button>
</form>
<p class="notice">Have an Invite? <a href="/register">Register</a></p>"""
    return HTMLResponse(PAGE.format(title="Log in · Perry", body=body), status_code=status_code)


@dataclass(frozen=True)
class RegisterForm:
    """What the register page shows filled in (never the password)."""

    invite: str = ""
    username: str = ""
    display_name: str = ""
    time_zone: str = ""


REGISTER_NOTICE = (
    "Perry answers with Google Gemini: your messages are sent to Google and may be logged."
)


def _register_page(form: RegisterForm, error: str = "", status_code: int = 200) -> HTMLResponse:
    def value(text: str) -> str:
        return f'value="{html.escape(text)}"'

    body = f"""<h1>Register with Perry</h1>
<p>You need an Invite from Perry's Owner.</p>
{pages.error_line(error)}
<form method="post" action="/register">
  <label>Invite <input name="invite" required autocomplete="off" {value(form.invite)}></label>
  <label>Username <input name="username" autocomplete="username" required
         {value(form.username)}></label>
  <label>Password (at least {users.MIN_PASSWORD_CHARS} characters)
    <input name="password" type="password" autocomplete="new-password" required
           minlength="{users.MIN_PASSWORD_CHARS}"></label>
  <label>Display name (what Perry calls you)
    <input name="display_name" autocomplete="name" required {value(form.display_name)}></label>
  <label>Time zone <input name="time_zone" id="time_zone" required
         placeholder="e.g. Europe/London" {value(form.time_zone)}></label>
  <p class="notice">{html.escape(REGISTER_NOTICE)}</p>
  <button type="submit">Register</button>
</form>
<p class="notice">Already registered? <a href="/login">Log in</a></p>
<script>
  // Pre-fill the time zone from the browser.
  const zone = document.getElementById("time_zone");
  if (!zone.value) zone.value = Intl.DateTimeFormat().resolvedOptions().timeZone || "";
</script>"""
    return pages.page("Register", body, status_code=status_code)


def _profile_page(
    profile: users.Profile, error: str = "", saved: bool = False, status_code: int = 200
) -> HTMLResponse:
    if error:
        note = f'<p class="error" role="alert">{html.escape(error)}</p>'
    else:
        note = '<p class="saved" role="status">Saved.</p>' if saved else ""
    body = f"""<h1>Your Profile</h1>
<p>Perry reads this when answering you.</p>
{note}
<form method="post" action="/profile">
  <label>Display name <input name="display_name" required autocomplete="name"
         value="{html.escape(profile.display_name)}"></label>
  <label>Time zone <input name="time_zone" required placeholder="Europe/London"
         value="{html.escape(profile.time_zone)}"></label>
  <label>About you (optional, one line)
    <input name="about" maxlength="{users.MAX_ABOUT_CHARS}"
           placeholder="e.g. a second-year chemistry student"
           value="{html.escape(profile.about)}"></label>
  <button type="submit">Save</button>
</form>
<p><a href="{CHAT_PATH}/">Back to Perry</a></p>"""
    return pages.page("Profile", body, status_code=status_code)


HTTPS_FLAG = "--https"


def main(argv: list[str] | None = None) -> None:
    """Serve Perry on HOST:PORT. With --https (as scripts/serve.py runs it, behind ngrok's
    HTTPS endpoint), the session cookie is marked Secure."""
    import sys

    import uvicorn

    from perry import db, ingest, privacy, summaries

    # Before anything runs: switches tracing off if a real account is connected.
    notice = privacy.startup() or ""
    if notice:
        print(notice)
    argv = sys.argv[1:] if argv is None else argv
    unknown = [arg for arg in argv if arg != HTTPS_FLAG]
    if unknown:
        raise SystemExit(f"Usage: uv run python -m perry.web [{HTTPS_FLAG}]")
    db.init_schema()
    try:
        served = create_app(chat_ui.build(notice), secure_cookies=HTTPS_FLAG in argv)
    except users.Refused as refused:
        raise SystemExit(str(refused)) from None
    # e.g. a changed embedding model, Documents stored before ColBERT, or after a crash.
    ingest.ensure_indexes_current()
    # Retry Summaries that failed (e.g. the local model wasn't running) and finish pending ones.
    summaries.resume()
    print(f"Perry: http://{HOST}:{PORT}/")
    uvicorn.run(served, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
