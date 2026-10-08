import hashlib
import os
import re

import psycopg
import pytest

# Tests use their own database so they can never touch real Documents.
ADMIN_URL = "postgresql://perry:perry@127.0.0.1:5433/perry"
TEST_URL = "postgresql://perry:perry@127.0.0.1:5433/perry_test"
os.environ["DATABASE_URL"] = TEST_URL


def fake_embed_one(text: str) -> list[float]:
    """Deterministic bag-of-words embedding: texts sharing words point the same way."""
    from perry.models import EMBEDDING_DIMENSIONS

    vector = [0.0] * EMBEDDING_DIMENSIONS
    for word in re.findall(r"\w+", text.lower()):
        index = int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBEDDING_DIMENSIONS
        vector[index] += 1.0
    return vector


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [fake_embed_one(t) for t in texts]


@pytest.fixture
def database():
    try:
        with psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=3) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = 'perry_test'"
            ).fetchone()
            if not exists:
                conn.execute("CREATE DATABASE perry_test")
    except psycopg.OperationalError:
        pytest.skip("Postgres not running: start it with `docker compose up -d`")

    from perry import db

    db.init_schema()
    with db.pool().connection() as conn:
        conn.execute("TRUNCATE documents, chunks, threads RESTART IDENTITY CASCADE")
    return db
