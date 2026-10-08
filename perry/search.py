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
