"""The one Postgres database: Documents, Chunks, Threads and LangGraph checkpoints."""

import os
from functools import cache

from pgvector.psycopg import register_vector
from psycopg import Connection
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
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);

CREATE TABLE IF NOT EXISTS threads (
    id         text PRIMARY KEY,
    title      text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
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
