"""Hybrid search over Chunks: vector similarity plus full-text, merged by reciprocal rank fusion."""

from dataclasses import dataclass

from pgvector import Vector

from perry import db, models
from perry.documents import citation

# How many candidates each method contributes before fusion.
CANDIDATES = 20
# Standard RRF constant: dampens the advantage of the very top ranks.
RRF_K = 60

HYBRID_SQL = """
WITH vector_hits AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> %(embedding)s) AS rank
    FROM chunks
    ORDER BY embedding <=> %(embedding)s
    LIMIT %(candidates)s
),
text_hits AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, query) DESC) AS rank
    FROM chunks, websearch_to_tsquery('english', %(query)s) AS query
    WHERE tsv @@ query
    ORDER BY ts_rank_cd(tsv, query) DESC
    LIMIT %(candidates)s
),
fused AS (
    SELECT id, sum(1.0 / (%(rrf_k)s + rank)) AS score
    FROM (SELECT * FROM vector_hits UNION ALL SELECT * FROM text_hits) AS hits
    GROUP BY id
)
SELECT d.name, c.page, c.text, f.score
FROM fused f
JOIN chunks c ON c.id = f.id
JOIN documents d ON d.id = c.document_id
ORDER BY f.score DESC
LIMIT %(limit)s
"""


@dataclass(frozen=True)
class SearchHit:
    document: str
    page: int
    text: str
    score: float

    @property
    def citation(self) -> str:
        return citation(self.document, self.page)


def hybrid_search(query: str, limit: int = 6, *, embed=models.embed_query) -> list[SearchHit]:
    params = {
        "embedding": Vector(embed(query)),
        "query": query,
        "candidates": CANDIDATES,
        "rrf_k": RRF_K,
        "limit": limit,
    }
    with db.pool().connection() as conn:
        rows = conn.execute(HYBRID_SQL, params).fetchall()
    return [SearchHit(r["name"], r["page"], r["text"], float(r["score"])) for r in rows]


# Summary search: how many Summaries may match, and how many of the best Chunks
# beneath each one come back.
SUMMARY_CANDIDATES = 3
CHUNKS_PER_SUMMARY = 3

SUMMARY_SQL = """
WITH matched AS (
    SELECT s.id, row_number() OVER (ORDER BY s.embedding <=> %(embedding)s) AS rank
    FROM summaries s
    LEFT JOIN documents d ON d.id = s.document_id
    WHERE s.kind = %(kind)s AND (s.document_id IS NULL OR d.{status} = 'ready')
    ORDER BY s.embedding <=> %(embedding)s
    LIMIT %(summaries)s
),
beneath AS (
    SELECT DISTINCT ON (c.id) c.id, m.rank, c.embedding <=> %(embedding)s AS distance
    FROM matched m
    JOIN summary_chunks sc ON sc.summary_id = m.id
    JOIN chunks c ON c.id = sc.chunk_id
    JOIN documents cd ON cd.id = c.document_id AND cd.{status} = 'ready'
    ORDER BY c.id, m.rank
),
ranked AS (
    SELECT *, row_number() OVER (PARTITION BY rank ORDER BY distance) AS place FROM beneath
)
SELECT d.name, c.page, c.text, 1 - r.distance AS score
FROM ranked r
JOIN chunks c ON c.id = r.id
JOIN documents d ON d.id = c.document_id
WHERE r.place <= %(per_summary)s
ORDER BY r.rank, r.distance
LIMIT %(limit)s
"""


# The status column that says whether a Document's Summaries of each kind are ready.
_STATUS = {"document": "summary_status", "raptor": "raptor_status"}


def summary_search(
    query: str, limit: int = 6, *, kind: str = "document", embed=models.embed_query
) -> list[SearchHit]:
    """The Chunks beneath the Summaries of `kind` that best match `query`: the best
    Chunks under the best-matching Summary first. A Summary is never returned itself,
    and the Summaries of Documents whose summarising isn't ready are skipped (for RAPTOR
    nodes, whose tree isn't ready; a cross-Document node returns only ready Documents'
    Chunks)."""
    params = {
        "embedding": Vector(embed(query)),
        "kind": kind,
        "summaries": SUMMARY_CANDIDATES,
        "per_summary": CHUNKS_PER_SUMMARY,
        "limit": limit,
    }
    with db.pool().connection() as conn:
        rows = conn.execute(SUMMARY_SQL.format(status=_STATUS[kind]), params).fetchall()
    return [SearchHit(r["name"], r["page"], r["text"], float(r["score"])) for r in rows]
