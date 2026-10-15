"""What the served app's small server-rendered pages share: the page shell, the session
cookie, and the User behind a request. Used by perry.web and perry.owner_panel."""

import html

from fastapi import Request
from fastapi.responses import HTMLResponse

from perry import users

SESSION_COOKIE = "perry_session"
CHAT_PATH = "/chat"
LOGIN_PATH = "/login"
LOGOUT_PATH = "/logout"  # POST only

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #f6f6f4; color: #1d1d1b;
         display: flex; justify-content: center; padding: 10vh 16px; margin: 0; }}
  main {{ background: #fff; padding: 2rem; border-radius: 12px; width: 100%;
         max-width: 26rem; box-shadow: 0 1px 4px rgba(0,0,0,.08); box-sizing: border-box;
         min-width: 0; }}
  @media (max-width: 600px) {{
    body {{ padding: 16px; }}
    main {{ padding: 1.25rem; }}
  }}
  main:has(> .wide) {{ max-width: 56rem; }}
  label {{ display: block; margin-top: 1rem; font-size: .9rem; }}
  input {{ width: 100%; box-sizing: border-box; padding: .55rem; margin-top: .3rem;
          font-size: 1rem; border: 1px solid #c8c8c4; border-radius: 6px; }}
  button {{ font-family: inherit; margin-top: 1.5rem; width: 100%; padding: .65rem; font-size: 1rem;
           border: 0; border-radius: 6px; background: #2f5d50; color: #fff; cursor: pointer; }}
  button:hover {{ background: #264d42; }}
  button.small {{ margin: 0; width: auto; padding: .35rem .75rem; font-size: .85rem;
                 line-height: 1.2; border: 1px solid #2f5d50; white-space: nowrap; }}
  button.secondary {{ background: #fff; color: #1d1d1b; border-color: #c8c8c4; }}
  button.secondary:hover {{ border-color: #2f5d50; color: #2f5d50; }}
  button.danger {{ background: #fff; color: #a3261b; border-color: #e3bdb8; }}
  button.danger:hover {{ background: #fbefed; border-color: #a3261b; }}
  button:disabled {{ cursor: default; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: .5rem; font-size: .9rem; }}
  th, td {{ text-align: left; padding: .4rem .3rem; border-bottom: 1px solid #e4e4e0; }}
  section {{ margin-top: 2rem; }}
  code {{ font-family: ui-monospace, Consolas, monospace; font-size: .85rem;
         background: #f1f1ee; padding: .15rem .4rem; border-radius: 4px; }}
  .back {{ color: #5b5b57; text-decoration: none; font-size: .9rem; }}
  .back:hover {{ color: #2f5d50; }}
  .wide h1 {{ margin: .5rem 0 .25rem; font-size: 1.6rem; }}
  .hint {{ color: #5b5b57; font-size: .9rem; margin: .3rem 0 0; }}
  .section-head {{ display: flex; flex-wrap: wrap; align-items: center;
                  justify-content: space-between; gap: .5rem 1rem; }}
  .section-head h2 {{ margin: 0; font-size: 1.2rem; }}
  .table-wrap {{ overflow-x: auto; margin-top: .75rem; }}
  .wide table {{ margin-top: 0; }}
  .wide th {{ padding: .5rem; font-size: .75rem; font-weight: 600; color: #6b6b66;
             text-transform: uppercase; letter-spacing: .04em; }}
  .wide td {{ padding: .6rem .5rem; vertical-align: middle; }}
  .wide tr:last-child td {{ border-bottom: 0; }}
  td.when {{ white-space: nowrap; font-variant-numeric: tabular-nums; color: #3d3d3a; }}
  .actions {{ text-align: right; white-space: nowrap; }}
  .actions form {{ display: inline; margin: 0; }}
  .actions button {{ margin-left: .4rem; }}
  button[data-copy] {{ min-width: 6.5rem; }}
  .badge {{ display: inline-block; vertical-align: 1px; margin-left: .5rem; padding: .1rem .5rem; border-radius: 999px;
           background: #e6efec; color: #2f5d50; font-size: .75rem; font-weight: 600; }}
  .status {{ display: inline-flex; align-items: center; gap: .4rem;
            text-transform: capitalize; }}
  .status::before {{ content: ""; width: .5rem; height: .5rem; border-radius: 50%;
                    background: #3c9a6b; }}
  .status.disabled {{ color: #6b6b66; }}
  .status.disabled::before {{ background: #b0b0aa; }}
  @media (max-width: 600px) {{
    .wide table, .wide tbody, .wide tr, .wide td {{ display: block; }}
    .wide tr:first-child {{ display: none; }}
    .wide tr {{ padding: .75rem 0; border-bottom: 1px solid #e4e4e0; }}
    .wide tr:last-child {{ border-bottom: 0; }}
    .wide td {{ padding: .15rem 0; border: 0; }}
    .wide td[data-label]::before {{ content: attr(data-label); margin-right: .4rem;
                                    color: #6b6b66; font-size: .8rem; }}
    .wide .actions {{ text-align: left; padding-top: .5rem; }}
    .wide .actions button {{ margin: 0 .4rem 0 0; }}
  }}
  .empty {{ margin: .75rem 0 0; padding: 1rem; border: 1px dashed #d6d6d1; border-radius: 8px;
           color: #5b5b57; font-size: .9rem; text-align: center; }}
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
