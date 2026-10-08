import os

import psycopg
import pytest
from psycopg import sql

# Tests use their own database so they can never touch real Documents.
ADMIN_URL = "postgresql://perry:perry@127.0.0.1:5433/perry"
# PERRY_TEST_DB lets parallel checkouts each use their own database.
TEST_DB = os.environ.get("PERRY_TEST_DB", "perry_test")
TEST_URL = f"postgresql://perry:perry@127.0.0.1:5433/{TEST_DB}"
os.environ["DATABASE_URL"] = TEST_URL
# Set before perry loads .env (which never overrides), so tests never trace to LangSmith.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from fakes import ScriptedChatModel, fake_embed, fake_provider


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
        pytest.skip("Postgres not running: start it with `docker compose up -d`")

    from perry import db

    db.init_schema()
    with db.pool().connection() as conn:
        conn.execute("TRUNCATE documents, chunks, threads RESTART IDENTITY CASCADE")
        conn.execute("TRUNCATE datasets")
        tables = "SELECT tablename FROM pg_tables WHERE schemaname = 'analytics'"
        for row in conn.execute(tables).fetchall():
            table = sql.Identifier("analytics", row["tablename"])
            conn.execute(sql.SQL("DROP TABLE {}").format(table))
        conn.execute("TRUNCATE usage_ledger")
    return db


class Perry:
    """Drives Turns on one Thread the way the chat UI does, with fake models.

    Script the model's replies with `script()`, then `send()` a message and look at
    what the user would see: the reply, the Thread's `history()`, and which Agents
    were `consulted()`.
    """

    def __init__(self, model: ScriptedChatModel, documents_dir):
        self.model = model
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

    def send(self, message: str) -> str:
        """Run one Turn and return the reply the user sees."""
        from perry import supervisor, threads

        if self.thread_id is None:
            self.thread_id = threads.create(message)
        return supervisor.run_turn(self.thread_id, message)

    def history(self) -> list[dict]:
        from perry import supervisor

        return supervisor.thread_history(self.thread_id) if self.thread_id else []

    def consulted(self) -> list[str]:
        """The Agent tools the Supervisor called during the latest Turn, in order."""
        from perry import supervisor

        if self.thread_id is None:
            return []
        state = supervisor.supervisor().get_state({"configurable": {"thread_id": self.thread_id}})
        messages = state.values.get("messages", [])
        last_user = max((i for i, m in enumerate(messages) if m.type == "human"), default=-1)
        return [m.name for m in messages[last_user + 1 :] if m.type == "tool"]


@pytest.fixture
def chat_model() -> ScriptedChatModel:
    return ScriptedChatModel()


@pytest.fixture
def perry(database, chat_model, tmp_path):
    """A Perry whose every model call goes to `chat_model` and the fake embeddings."""
    from perry import models

    with models.use(fake_provider(chat_model)):
        yield Perry(chat_model, tmp_path)
