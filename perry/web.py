"""The served app: Perry's login and logout pages, with the chat UI (perry.app) mounted
behind them at /chat. Run with: uv run python -m perry.web

Every request under /chat, page or Gradio API, needs a session: pages without one are
sent to the login page, API calls are refused. Handlers learn the User from Gradio's
request (see `perry.app.user_of`), never from anything the browser holds.
"""

import html

import gradio as gr
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from perry import app as chat_ui
from perry import charts, users

CHAT_PATH = "/chat"
SESSION_COOKIE = "perry_session"
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

    @served.api_route("/logout", methods=["GET", "POST"])
    def log_out():
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(SESSION_COOKIE, httponly=True, secure=secure_cookies, samesite="lax")
        return response

    return gr.mount_gradio_app(
        served,
        blocks if blocks is not None else chat_ui.build(),
        path=CHAT_PATH,
        auth_dependency=_gradio_user,
        allowed_paths=[str(charts.directory())],
        footer_links=[],
    )


def _user(request: Request) -> users.User | None:
    return users.user_for_session(request.cookies.get(SESSION_COOKIE))


def _gradio_user(request: Request) -> str | None:
    """What Gradio knows the User by (its request's `username`): the User's id."""
    user = _user(request)
    return str(user.id) if user else None


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #f6f6f4; color: #1d1d1b;
         display: flex; justify-content: center; padding: 10vh 16px; margin: 0; }}
  main {{ background: #fff; padding: 2rem; border-radius: 12px; width: 100%;
         max-width: 22rem; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  label {{ display: block; margin-top: 1rem; font-size: .9rem; }}
  input {{ width: 100%; box-sizing: border-box; padding: .55rem; margin-top: .3rem;
          font-size: 1rem; border: 1px solid #c8c8c4; border-radius: 6px; }}
  button {{ margin-top: 1.5rem; width: 100%; padding: .65rem; font-size: 1rem;
           border: 0; border-radius: 6px; background: #2f5d50; color: #fff; cursor: pointer; }}
  .error {{ color: #a3261b; margin-top: 1rem; }}
</style></head>
<body><main>{body}</main></body></html>"""


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
</form>"""
    return HTMLResponse(PAGE.format(title="Log in · Perry", body=body), status_code=status_code)


def main() -> None:
    import uvicorn

    from perry import db, ingest, privacy, summaries

    # Before anything runs: switches tracing off if a real account is connected.
    notice = privacy.startup() or ""
    if notice:
        print(notice)
    db.init_schema()
    try:
        served = create_app(chat_ui.build(notice))
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
