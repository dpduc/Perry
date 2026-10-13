"""Serve Perry over ngrok: starts Postgres, checks Ollama, launches Perry on 127.0.0.1 and
opens the ngrok tunnel on the Owner's static domain (PERRY_NGROK_DOMAIN).

Run from the repo root: uv run python -m scripts.serve   (Ctrl+C stops Perry and ngrok)
"""

import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

import psycopg

from perry import db, models, users, web  # importing perry loads .env

DOMAIN_VARIABLE = "PERRY_NGROK_DOMAIN"
NO_DOMAIN = (
    f"{DOMAIN_VARIABLE} is not set. Claim your free static domain at "
    "https://dashboard.ngrok.com/domains and add it to .env, e.g. "
    f"{DOMAIN_VARIABLE}=your-name.ngrok-free.app"
)
NO_NGROK = (
    "ngrok isn't installed (or isn't on PATH). Install it (winget install ngrok.ngrok), "
    "then connect your account once: ngrok config add-authtoken <your token>"
)
NO_POSTGRES = "Postgres didn't start with `docker compose up -d`: is Docker Desktop running?"
PERRY_URL = f"http://{web.HOST}:{web.PORT}"
STARTUP_SECONDS = 600  # re-indexing after a model change can take a while


def ngrok_domain() -> str:
    """The static domain, as host name only (the .env value may be a full URL)."""
    domain = os.getenv(DOMAIN_VARIABLE, "").strip()
    domain = domain.removeprefix("https://").removeprefix("http://").rstrip("/")
    if not domain:
        raise SystemExit(NO_DOMAIN)
    return domain


def require_settings() -> str:
    """Every .env setting serving needs, all refusals at once; returns the domain."""
    problems = []
    try:
        domain = ngrok_domain()
    except SystemExit as refused:
        problems.append(str(refused))
    try:
        users.session_secret()
    except users.Refused as refused:
        problems.append(str(refused))
    if problems:
        raise SystemExit("\n".join(problems))
    return domain


def require_ngrok() -> None:
    if shutil.which("ngrok") is None:
        raise SystemExit(NO_NGROK)


def postgres_is_up() -> bool:
    try:
        with psycopg.connect(db.DATABASE_URL, connect_timeout=3):
            return True
    except psycopg.OperationalError:
        return False


def start_postgres() -> None:
    if postgres_is_up():
        return
    print("Starting Postgres (docker compose up -d) ...")
    try:
        subprocess.run(["docker", "compose", "up", "-d", "--wait"], check=True)
    except (OSError, subprocess.CalledProcessError):
        raise SystemExit(NO_POSTGRES) from None
    if not postgres_is_up():
        raise SystemExit(NO_POSTGRES)


def require_ollama() -> None:
    problem = models.local_models_problem()
    if problem:
        raise SystemExit(problem)


def require_owner() -> None:
    db.init_schema()
    try:
        users.require_owner()
    except users.Refused as refused:
        raise SystemExit(str(refused)) from None


def answers(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2):
            return True
    except (urllib.error.URLError, OSError):
        return False


def launch_perry() -> subprocess.Popen:
    """Perry on 127.0.0.1 only, its session cookie Secure (ngrok serves it over HTTPS)."""
    if answers(f"{PERRY_URL}/login"):
        raise SystemExit(f"Something is already serving {PERRY_URL}: stop it first.")
    perry = subprocess.Popen([sys.executable, "-m", "perry.web", web.HTTPS_FLAG])
    deadline = time.monotonic() + STARTUP_SECONDS
    while not answers(f"{PERRY_URL}/login"):
        if perry.poll() is not None:
            raise SystemExit(f"Perry stopped while starting (exit code {perry.returncode}).")
        if time.monotonic() > deadline:
            perry.terminate()
            raise SystemExit(f"Perry didn't answer at {PERRY_URL} within {STARTUP_SECONDS}s.")
        time.sleep(1)
    return perry


def ngrok_command(domain: str) -> list[str]:
    # Logs to the console (warnings only) instead of ngrok's full-screen view, so the
    # public URL printed above stays visible.
    target = f"{web.HOST}:{web.PORT}"
    return ["ngrok", "http", f"--url={domain}", "--log=stdout", "--log-level=warn", target]


def main() -> None:
    domain = require_settings()
    require_ngrok()
    start_postgres()
    require_ollama()
    require_owner()
    perry = launch_perry()
    tunnel = None
    try:
        tunnel = subprocess.Popen(ngrok_command(domain))
        print(f"\nPerry is public at: https://{domain}/\nCtrl+C stops Perry and ngrok.\n")
        code = tunnel.wait()
        if code:
            print(f"ngrok stopped (exit code {code}); see its message above.")
    except KeyboardInterrupt:
        pass
    finally:
        for process in (tunnel, perry):
            if process is not None and process.poll() is None:
                process.terminate()


if __name__ == "__main__":
    main()
