"""Hybrid search over Chunks: vector similarity plus full-text, merged by reciprocal rank fusion."""

from dataclasses import dataclass, fields
from datetime import date, datetime, time, timedelta

from pgvector import Vector

from perry import db, models
from perry.documents import citation

# How many candidates each method contributes before fusion.
CANDIDATES = 20
# Standard RRF constant: dampens the advantage of the very top ranks.
RRF_K = 60

HYBRID_SQL = """
WITH vector_hits AS (
    SELECT c.id, row_number() OVER (ORDER BY c.embedding <=> %(embedding)s) AS rank
    FROM chunks c{documents}
    {where}
    ORDER BY c.embedding <=> %(embedding)s
    LIMIT %(candidates)s
),
text_hits AS (
    SELECT c.id, row_number() OVER (ORDER BY ts_rank_cd(c.tsv, query) DESC) AS rank
    FROM chunks c{documents}, websearch_to_tsquery('english', %(query)s) AS query
    WHERE c.tsv @@ query{and_where}
    ORDER BY ts_rank_cd(c.tsv, query) DESC
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


@dataclass(frozen=True)
class DocumentFilter:
    """Which Chunks a search may return, by their Document's metadata. Every field set
    must match; an unset one matches everything.

    `name` and `topic` match when each of their words appears in the Document's name or
    topic (any case), so "2025 budget" matches "budget-2025.md". `file_type` is the
    name's suffix ("pdf"). The ranges are inclusive: `uploaded_*` on the local day the
    Document was ingested, `page_*` on the Chunk's page, and `dated_*` on the Document
    date. The Document date and topic come from the summarising job, so a Document not
    yet summarised never matches a filter on them.
    """

    name: str | None = None
    file_type: str | None = None
    uploaded_from: date | None = None
    uploaded_to: date | None = None
    page_from: int | None = None
    page_to: int | None = None
    dated_from: date | None = None
    dated_to: date | None = None
    topic: str | None = None

    def __bool__(self) -> bool:
        return any(getattr(self, f.name) is not None for f in fields(self))

    def sql(self) -> tuple[list[str], dict]:
        """The SQL conditions on `d` (documents) and `c` (chunks), and their parameters."""
        conditions: list[str] = []
        params: dict = {}

        def where(condition: str, **values) -> None:
            conditions.append(condition)
            params.update(values)

        for column, text in (("d.name", self.name), ("d.topic", self.topic)):
            for n, word in enumerate((text or "").split()):
                key = f"{column[2:]}_word_{n}"
                where(f"{column} ILIKE %({key})s", **{key: f"%{_like_escape(word)}%"})
        if self.file_type:
            suffix = "." + self.file_type.strip().lstrip(".")
            where("d.name ILIKE %(suffix)s", suffix=f"%{_like_escape(suffix)}")
        # Upload days are local days: from the first day's local midnight up to (not
        # including) the midnight after the last day.
        if self.uploaded_from is not None:
            where(
                "d.ingested_at >= %(uploaded_from)s",
                uploaded_from=_midnight(self.uploaded_from),
            )
        if self.uploaded_to is not None:
            after = _midnight(self.uploaded_to + timedelta(days=1))
            where("d.ingested_at < %(uploaded_before)s", uploaded_before=after)
        ranges = (
            ("c.page", "page", self.page_from, self.page_to),
            ("d.document_date", "dated", self.dated_from, self.dated_to),
        )
        for column, key, low, high in ranges:
            if low is not None:
                where(f"{column} >= %({key}_from)s", **{f"{key}_from": low})
            if high is not None:
                where(f"{column} <= %({key}_to)s", **{f"{key}_to": high})
        return conditions, params


def _midnight(day: date) -> datetime:
    """The local midnight starting `day`."""
    return datetime.combine(day, time.min).astimezone()


def _like_escape(text: str) -> str:
    """`text` matched literally inside a LIKE pattern."""
    return text.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")


def hybrid_search(
    query: str,
    limit: int = 6,
    *,
    embed=models.embed_query,
    where: DocumentFilter | None = None,
) -> list[SearchHit]:
    """The Chunks best matching `query` by meaning and by words, fused. With `where`,
    only Chunks of matching Documents (and pages) come back."""
    params = {
        "embedding": Vector(embed(query)),
        "query": query,
        "candidates": CANDIDATES,
        "rrf_k": RRF_K,
        "limit": limit,
    }
    conditions, filter_params = where.sql() if where else ([], {})
    params.update(filter_params)
    sql = HYBRID_SQL.format(
        documents=" JOIN documents d ON d.id = c.document_id" if conditions else "",
        where=("WHERE " + " AND ".join(conditions)) if conditions else "",
        and_where="".join(f" AND {condition}" for condition in conditions),
    )
    with db.pool().connection() as conn, conn.transaction():
        if conditions:
            # Keep scanning the vector index until enough Chunks pass the filter.
            conn.execute(
                "SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)"
            )
        rows = conn.execute(sql, params).fetchall()
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
    WHERE s.kind = %(kind)s AND (s.document_id IS NULL OR d.summary_status = 'ready')
    ORDER BY s.embedding <=> %(embedding)s
    LIMIT %(summaries)s
),
beneath AS (
    SELECT DISTINCT ON (c.id) c.id, m.rank, c.embedding <=> %(embedding)s AS distance
    FROM matched m
    JOIN summary_chunks sc ON sc.summary_id = m.id
    JOIN chunks c ON c.id = sc.chunk_id
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


def summary_search(
    query: str, limit: int = 6, *, kind: str = "document", embed=models.embed_query
) -> list[SearchHit]:
    """The Chunks beneath the Summaries of `kind` that best match `query`: the best
    Chunks under the best-matching Summary first. A Summary is never returned itself,
    and the Summaries of Documents whose summarising isn't ready are skipped."""
    params = {
        "embedding": Vector(embed(query)),
        "kind": kind,
        "summaries": SUMMARY_CANDIDATES,
        "per_summary": CHUNKS_PER_SUMMARY,
        "limit": limit,
    }
    with db.pool().connection() as conn:
        rows = conn.execute(SUMMARY_SQL, params).fetchall()
    return [SearchHit(r["name"], r["page"], r["text"], float(r["score"])) for r in rows]
