import base64
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import psycopg
import pymupdf
import pytest
from psycopg import sql

# Tests use their own database so they can never touch real Documents.
ADMIN_URL = "postgresql://perry:perry@127.0.0.1:5433/perry"
# Each checkout (e.g. a git worktree) gets its own database, so parallel runs never
# collide. PERRY_TEST_DB overrides the name.
_CHECKOUT = hashlib.sha1(str(Path(__file__).resolve().parent.parent).lower().encode())
TEST_DB = os.environ.get("PERRY_TEST_DB") or f"perry_test_{_CHECKOUT.hexdigest()[:8]}"
TEST_URL = f"postgresql://perry:perry@127.0.0.1:5433/{TEST_DB}"
os.environ["DATABASE_URL"] = TEST_URL
# Set before perry loads .env (which never overrides), so tests never trace to LangSmith.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["PERRY_RETRY_DELAY_SECONDS"] = "0"
# Turns script which technique tool the Knowledge Agent calls, so by default its model
# picks (logically). Tests of semantic and cascade picking set the mode themselves.
os.environ["PERRY_PICKING_MODE"] = "logical"
# No real ColBERT model (torch, and an index in the real app-data folder) unless a test
# asks for one; tests that need ColBERT use the fake encoder.
os.environ["PERRY_COLBERT_MODEL"] = ""
# Placeholder keys, so no test spends real quota and the suite runs where there is no
# .env (CI, a fresh worktree). Tests that need the paid fallback set its key themselves.
os.environ["GOOGLE_API_KEY"] = "test-google-key"
os.environ["OPENAI_API_KEY"] = ""
# Sessions are signed with this; the served app refuses to start without one.
os.environ["PERRY_SESSION_SECRET"] = "test-session-secret"

from fakes import ScriptedChatModel, fake_embed, fake_provider
from logins import (
    OWNER_PASSWORD,
    OWNER_USERNAME,
    SAMS_PASSWORD,
    create_invite,
    log_in,
    register,
)


@pytest.fixture
def database():
    try:
        with psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=3) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)
            ).fetchone()
            if not exists:
                conn.execute(f'CREATE DATABASE "{TEST_DB}"')
    except psycopg.OperationalError:
        if os.environ.get("CI"):  # in CI a skipped DB test would pass silently
            raise
        pytest.skip("Postgres not running: start it with `docker compose up -d`")

    from perry import db

    db.init_schema()
    with db.pool().connection() as conn:
        conn.execute("TRUNCATE documents, chunks, threads, users, invites, failed_logins, reset_codes RESTART IDENTITY CASCADE")
        conn.execute("TRUNCATE datasets")
        tables = "SELECT tablename FROM pg_tables WHERE schemaname = 'analytics'"
        for row in conn.execute(tables).fetchall():
            table = sql.Identifier("analytics", row["tablename"])
            conn.execute(sql.SQL("DROP TABLE {}").format(table))
        conn.execute("TRUNCATE usage_ledger, store_facts")
    return db


@pytest.fixture
def owner(database):
    """The Owner, created as the create-owner command does."""
    from perry import users

    return users.create_owner(OWNER_USERNAME, OWNER_PASSWORD, "Olive Owner", "Europe/London")


@pytest.fixture
def client(owner):
    """A browser on the served app (login pages with the Gradio UI mounted), not logged in."""
    from fastapi.testclient import TestClient

    from perry import web

    with TestClient(web.create_app()) as browser:
        yield browser


@pytest.fixture
def owner_client(client):
    """A browser logged in as the Owner."""
    log_in(client, OWNER_USERNAME, OWNER_PASSWORD)
    return client


@pytest.fixture
def owner_browser(owner):
    """The Owner, logged in on a second browser beside `client`."""
    from fastapi.testclient import TestClient

    from perry import web

    with TestClient(web.create_app()) as browser:
        log_in(browser, OWNER_USERNAME, OWNER_PASSWORD)
        yield browser


@pytest.fixture
def sam(client, owner_browser):
    """Sam, a User who registered with an Invite and is logged in on `client`."""
    register(client, create_invite(owner_browser), "sam", password=SAMS_PASSWORD)
    return client


@pytest.fixture(scope="session")
def sandbox_image():
    """The Sandbox image, built if missing. Skips when Docker isn't running."""
    from perry import sandbox

    if not sandbox.docker_running():
        pytest.skip("Docker not running: start Docker Desktop")
    sandbox.ensure_image()


@dataclass(frozen=True)
class VoiceReply:
    transcript: str
    text: str
    audio: bytes | None
    note: str = ""


class Perry:
    """Drives Turns on one Thread the way the chat UI does, with fake models.

    Script the model's replies with `script()`, then `send()` a message and look at
    what the user would see: the reply, the Thread's `history()`, and which Agents
    were `consulted()`.
    """

    def __init__(self, model: ScriptedChatModel, documents_dir, user):
        self.model = model
        self.user = user  # the User sending the messages (the Owner, by default)
        self.thread_id: str | None = None
        self._documents_dir = documents_dir

    def script(self, *steps) -> None:
        """Queue the model's next replies, in call order across the Supervisor and Agents."""
        self.model.steps.extend(steps)

    def add_document(self, name: str, text: str) -> None:
        from perry.ingest import ingest_file

        path = self._documents_dir / name
        path.write_text(text, encoding="utf-8")
        ingest_file(path, embed=fake_embed)

    def thread(self):
        """The Thread being driven, opened as the chat UI opens it on each request."""
        from perry import threads

        return threads.open(self.user, self.thread_id)

    def send(self, message: str, images=(), technique: str | None = None) -> str:
        """Run one Turn and return the reply the user sees. `images` are image files
        attached to the message, as when dropped into the chat; `technique` forces a
        Retrieval Technique, as the sidebar dropdown does."""
        thread = self.thread()
        try:
            return thread.send(message, images, technique)
        finally:
            self.thread_id = thread.id

    def resume(self, decision):
        """Answer the pending Confirmation (approve, approve with edits, or reject with a
        note) and return what the Turn shows next: a reply or another Confirmation."""
        from perry.supervisor import Reject

        if isinstance(decision, Reject):
            return self.thread().reject(decision.note)
        return self.thread().approve(decision.edits)

    def say(self, audio: bytes) -> "VoiceReply":
        """Run one Turn from recorded speech, as push-to-talk does: the transcription is
        sent as the message, and the reply comes back as text and (unless a limit stops
        it, which the note says) audio."""
        thread = self.thread()
        heard = thread.hear(audio)
        try:
            text = thread.send(heard)
        finally:
            self.thread_id = thread.id
        speech = thread.read_aloud(heard, text)
        return VoiceReply(heard.transcript, text, speech.audio, speech.note)

    def history(self) -> list[dict]:
        return self.thread().history()

    def consulted(self) -> list[str]:
        """The Agent tools the Supervisor called during the latest Turn, in order."""
        from perry import supervisor

        return supervisor.agents_consulted(self.thread_id) if self.thread_id else []


@pytest.fixture
def chat_model() -> ScriptedChatModel:
    return ScriptedChatModel()


@pytest.fixture
def perry(owner, chat_model, tmp_path):
    """A Perry, used by the Owner, whose every model call goes to `chat_model` and the
    fake embeddings."""
    from perry import models

    with models.use(fake_provider(chat_model)):
        yield Perry(chat_model, tmp_path, owner)


@pytest.fixture
def workspace():
    """An in-memory Workspace that every Turn in the test reads from and acts on."""
    from perry import workspace

    fake = workspace.InMemoryWorkspace()
    with workspace.use(fake):
        yield fake


@pytest.fixture
def prices(tmp_path, monkeypatch):
    """Write a price table (USD per million tokens) and point Perry at it."""
    path = tmp_path / "prices.toml"
    monkeypatch.setenv("PERRY_PRICES_FILE", str(path))

    def write(table: dict[str, dict[str, float]]) -> None:
        lines = []
        for model, rates in table.items():
            lines.append(f'[models."{model}"]')
            lines += [f"{kind}_per_million = {rate}" for kind, rate in rates.items()]
        path.write_text("\n".join(lines), encoding="utf-8")

    return write


# A 1x1 PNG, standing in for a scanned page.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def half_scanned_pdf(tmp_path):
    """report.pdf: page 1 has text (46 characters), page 2 is only an image, as if scanned."""
    path = tmp_path / "report.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "The marketing budget for Q3 is 40,000 dollars.")
        pdf.new_page().insert_image(pymupdf.Rect(72, 72, 300, 300), stream=PNG)
        pdf.save(path)
    return path
