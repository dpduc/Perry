"""The one Postgres database: Documents, Chunks, Threads, Datasets and LangGraph checkpoints."""

import os
import time
from functools import cache

from pgvector.psycopg import register_vector
from psycopg import Connection, errors, sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from perry.models import EMBEDDING_DIMENSIONS

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://perry:perry@127.0.0.1:5433/perry")

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id           bigserial PRIMARY KEY,
    name         text NOT NULL UNIQUE,
    content_hash text NOT NULL,
    ingested_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    document_id bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page        int NOT NULL,
    text        text NOT NULL,
    embedding   vector({EMBEDDING_DIMENSIONS}) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
-- Each Chunk's ColBERT token vectors (a NumPy .npy array, one row per token): the
-- source the on-disk ColBERT index is built from (docs/adr/0002).
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS colbert bytea;
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);

-- Written in the background after upload (perry.summaries): the summarising status,
-- and the Document metadata extracted with the Summary.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary_status text NOT NULL DEFAULT 'pending'
    CHECK (summary_status IN ('pending', 'ready', 'failed'));
ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_date date;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS topic text;
CREATE INDEX IF NOT EXISTS documents_summary_status_idx ON documents (summary_status);
-- The status of the Document's RAPTOR tree (perry.raptor), built after its Summary.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS raptor_status text NOT NULL DEFAULT 'pending'
    CHECK (raptor_status IN ('pending', 'ready', 'failed'));

-- Summaries, used only to find Chunks: never cited. One per Document (kind 'document'),
-- or one per RAPTOR node (kind 'raptor'; level 1 and up; document_id NULL for the
-- cross-Document level). summary_chunks links each to every leaf Chunk beneath it, so a
-- match returns those Chunks with their page Citations.
CREATE TABLE IF NOT EXISTS summaries (
    id          bigserial PRIMARY KEY,
    document_id bigint REFERENCES documents(id) ON DELETE CASCADE,
    kind        text NOT NULL,
    level       int NOT NULL DEFAULT 1,
    text        text NOT NULL,
    embedding   vector({EMBEDDING_DIMENSIONS}) NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS summaries_document_idx ON summaries (document_id);
CREATE INDEX IF NOT EXISTS summaries_embedding_idx
    ON summaries USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS summary_chunks (
    summary_id bigint NOT NULL REFERENCES summaries(id) ON DELETE CASCADE,
    chunk_id   bigint NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    PRIMARY KEY (summary_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS summary_chunks_chunk_idx ON summary_chunks (chunk_id);

CREATE TABLE IF NOT EXISTS threads (
    id         text PRIMARY KEY,
    title      text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- Datasets are real tables in the analytics schema; this is their registry.
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE TABLE IF NOT EXISTS datasets (
    name        text PRIMARY KEY,
    origin      text NOT NULL,
    columns     jsonb NOT NULL,
    row_count   bigint NOT NULL,
    uploaded_at timestamptz NOT NULL DEFAULT now()
);

-- The usage ledger: one row per model call. Cost is priced when recorded.
CREATE TABLE IF NOT EXISTS usage_ledger (
    id            bigserial PRIMARY KEY,
    recorded_at   timestamptz NOT NULL DEFAULT now(),
    thread_id     text,
    turn_id       text,
    kind          text NOT NULL,
    model         text NOT NULL,
    input_tokens  int NOT NULL,
    output_tokens int NOT NULL,
    cost_usd      numeric NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_ledger_recorded_at_idx ON usage_ledger (recorded_at);
CREATE INDEX IF NOT EXISTS usage_ledger_turn_idx ON usage_ledger (turn_id);
"""

# The Analysis Agent's SQL runs as this role. Postgres, not the prompt, keeps it
# read-only and inside the analytics schema. Roles are cluster-wide (shared by the
# main and test databases); grants are per database, so both are (re)applied here.
ANALYST_ROLE = "perry_analyst"
ANALYST_PASSWORD = os.getenv("PERRY_ANALYST_PASSWORD", "perry_analyst")

ANALYST_ROLE_SQL = """
DO $$ BEGIN
    CREATE ROLE {role} LOGIN;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
ALTER ROLE {role} PASSWORD {password};
ALTER ROLE {role} SET default_transaction_read_only = on;
ALTER ROLE {role} SET search_path = analytics;
-- Temporary tables would be writable by anyone; nothing in Perry uses them.
DO $$ BEGIN
    EXECUTE format('REVOKE TEMPORARY ON DATABASE %I FROM PUBLIC', current_database());
END $$;
GRANT USAGE ON SCHEMA analytics TO {role};
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO {role};
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics GRANT SELECT ON TABLES TO {role};
"""


def _configure(conn: Connection) -> None:
    register_vector(conn)


@cache
def pool() -> ConnectionPool:
    """Shared pool. Autocommit + dict rows is what the LangGraph checkpointer requires."""
    _ensure_extension()
    return ConnectionPool(
        DATABASE_URL,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        configure=_configure,
        open=True,
    )


def _ensure_extension() -> None:
    # register_vector needs the extension to exist before any pooled connection opens.
    with Connection.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")


def init_schema() -> None:
    with pool().connection() as conn:
        # Several statements in one string can only run unprepared.
        conn.execute(SCHEMA, prepare=False)
        role_sql = sql.SQL(ANALYST_ROLE_SQL).format(
            role=sql.Identifier(ANALYST_ROLE), password=sql.Literal(ANALYST_PASSWORD)
        )
        # The role is cluster-wide, so another database's init_schema (e.g. a parallel
        # test run) can update it at the same moment; that loser just tries again.
        for attempt in range(5):
            try:
                conn.execute(role_sql, prepare=False)
                break
            except errors.InternalError_:
                if attempt == 4:
                    raise
                time.sleep(0.2 * (attempt + 1))


def analyst_conninfo(statement_timeout_ms: int) -> str:
    """Connection string for the read-only analyst role, with a statement timeout."""
    return make_conninfo(
        DATABASE_URL,
        user=ANALYST_ROLE,
        password=ANALYST_PASSWORD,
        options=f"-c statement_timeout={int(statement_timeout_ms)}",
    )
