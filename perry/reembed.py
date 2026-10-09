"""Keep every stored embedding made by the current embedding model (docs/adr/0003).

Embeddings from two models can't be compared, so when the embedding model changes, every
Chunk and Summary is embedded again. `db.init_schema` empties the embeddings whose size
changed; `ensure_current` empties the rest if another model made them, then embeds
everything left empty. Until then, searches skip what isn't embedded yet.
"""

from pgvector import Vector

from perry import db, models

BATCH = 32
_FACT = "embedding_model"


def ensure_current(report=print) -> int:
    """Re-embed every Chunk and Summary not embedded by the current model, and return how
    many were. Cheap when nothing changed: two queries."""
    model = models.embedding_model()
    with db.pool().connection() as conn:
        row = conn.execute("SELECT value FROM store_facts WHERE key = %s", (_FACT,)).fetchone()
        if row is not None and row["value"] != model:
            with conn.transaction():
                for table in db.EMBEDDED_TABLES:
                    conn.execute(f"UPDATE {table} SET embedding = NULL")
        conn.execute(
            "INSERT INTO store_facts (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (_FACT, model),
        )
    done = 0
    for table in db.EMBEDDED_TABLES:
        done += _embed_missing(table, report)
    return done


def _embed_missing(table: str, report) -> int:
    done = 0
    with db.pool().connection() as conn:
        missing = conn.execute(f"SELECT count(*) AS n FROM {table} WHERE embedding IS NULL")
        total = missing.fetchone()["n"]
    if total:
        report(f"Embedding {total} {table} with {models.embedding_model()}...")
    while True:
        with db.pool().connection() as conn:
            rows = conn.execute(
                f"SELECT id, text FROM {table} WHERE embedding IS NULL ORDER BY id LIMIT %s",
                (BATCH,),
            ).fetchall()
        if not rows:
            return done
        vectors = models.embed_documents([row["text"] for row in rows])
        with db.pool().connection() as conn, conn.transaction():
            for row, vector in zip(rows, vectors, strict=True):
                conn.execute(
                    f"UPDATE {table} SET embedding = %s WHERE id = %s",
                    (Vector(vector), row["id"]),
                )
        done += len(rows)
