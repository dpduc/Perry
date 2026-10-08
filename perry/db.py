"""The one Postgres database: Documents, Chunks, Threads, Datasets and LangGraph checkpoints."""

import os
from functools import cache

from pgvector.psycopg import register_vector
from psycopg import Connection, sql
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
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);

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
        conn.execute(role_sql, prepare=False)


def analyst_conninfo(statement_timeout_ms: int) -> str:
    """Connection string for the read-only analyst role, with a statement timeout."""
    return make_conninfo(
        DATABASE_URL,
        user=ANALYST_ROLE,
        password=ANALYST_PASSWORD,
        options=f"-c statement_timeout={int(statement_timeout_ms)}",
    )
