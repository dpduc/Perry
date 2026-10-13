"""What the served app's small server-rendered pages share: the page shell, the session
cookie, and the User behind a request. Used by perry.web and perry.owner_panel."""

import html

from fastapi import Request
from fastapi.responses import HTMLResponse

from perry import users

SESSION_COOKIE = "perry_session"

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #f6f6f4; color: #1d1d1b;
         display: flex; justify-content: center; padding: 10vh 16px; margin: 0; }}
  main {{ background: #fff; padding: 2rem; border-radius: 12px; width: 100%;
         max-width: 22rem; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  main:has(> .wide) {{ max-width: 44rem; }}
  label {{ display: block; margin-top: 1rem; font-size: .9rem; }}
  input {{ width: 100%; box-sizing: border-box; padding: .55rem; margin-top: .3rem;
          font-size: 1rem; border: 1px solid #c8c8c4; border-radius: 6px; }}
  button {{ margin-top: 1.5rem; width: 100%; padding: .65rem; font-size: 1rem;
           border: 0; border-radius: 6px; background: #2f5d50; color: #fff; cursor: pointer; }}
  button.small {{ margin: 0; width: auto; padding: .3rem .7rem; font-size: .85rem; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: .5rem; font-size: .9rem; }}
  th, td {{ text-align: left; padding: .4rem .3rem; border-bottom: 1px solid #e4e4e0; }}
  section {{ margin-top: 2rem; }}
  .error {{ color: #a3261b; margin-top: 1rem; }}
  .saved {{ color: #2f5d50; margin-top: 1rem; }}
  .notice {{ color: #5b5b57; font-size: .85rem; margin-top: 1rem; }}
</style></head>
<body><main>{body}</main></body></html>"""


def page(title: str, body: str, *, status_code: int = 200, wide: bool = False) -> HTMLResponse:
    """A page with Perry's look; `body` is HTML (escape anything from a person)."""
    if wide:
        body = f'<div class="wide">{body}</div>'
    return HTMLResponse(
        PAGE.format(title=f"{html.escape(title)} · Perry", body=body), status_code=status_code
    )


def error_line(error: str) -> str:
    """An error message for the top of a form; empty if there is none."""
    return f'<p class="error" role="alert">{html.escape(error, quote=False)}</p>' if error else ""


def current_user(request: Request) -> users.User | None:
    """The User whose session the request carries, if any."""
    return users.user_for_session(request.cookies.get(SESSION_COOKIE))
