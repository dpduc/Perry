"""Summaries and Document metadata, written in the background by the local model.

An upload returns as soon as its Document is searchable by hybrid search, with its
summarising status `pending`. Then, one Document at a time, a background job asks the
local summary model (`models.summary_model()`) for one Summary of the Document plus a
Document date and topic, embeds the Summary, and marks the Document `ready`. A Document
whose summarising fails (e.g. Ollama isn't running) stays searchable, is marked
`failed`, and is retried on the next start (`resume()`).

A Summary is used only to find Chunks, never cited: each one is linked to the Chunks it
covers (`summary_chunks`), and techniques return those Chunks with their page Citations.
The same tables hold RAPTOR nodes (kind "raptor"): `save_summary` is the one writer.

Re-uploading a Document deletes it and its Summaries (ON DELETE CASCADE) and starts
again from `pending`, so a job still working on the old version just finds it gone.
"""

import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from pgvector import Vector
from psycopg import Connection

from perry import db, models

log = logging.getLogger(__name__)

PENDING, READY, FAILED = "pending", "ready", "failed"
DOCUMENT = "document"  # the kind of a Summary of a whole Document

# The local model reads ~9 tokens/s on a CPU, so a Summary reads only the start of a
# Document (about 2,500 tokens), which usually says what it is about and when.
INPUT_CHARS = 10_000
# What one Summary call is expected to cost, for the upload estimate (priced at zero
# for a local model): the instructions plus the Document's start in, a short reply out.
PROMPT_TOKENS = 200
OUTPUT_TOKENS = 250

SUMMARY_PROMPT = """You summarise one document so it can be found later by what it is about.
Reply with only a JSON object with these keys:
- "summary": 3 to 6 sentences on what the document covers: its subject, purpose and key \
names, figures and conclusions.
- "document_date": the date the document is about or was written, as YYYY-MM-DD \
(YYYY-MM-01 or YYYY-01-01 if only the month or year is given), or null if it has none.
- "topic": its topic in 2 to 5 words.
Write the summary in the document's language. Use only what the document says."""


@dataclass(frozen=True)
class DocumentSummary:
    """A Document's summarising status and, once ready, its Summary and metadata."""

    status: str
    summary: str | None = None
    document_date: date | None = None
    topic: str | None = None


def input_text(chunk_texts: list[str]) -> str:
    """What the summary model reads of a Document made of these Chunks."""
    return "\n\n".join(chunk_texts)[:INPUT_CHARS]


def document(name: str) -> DocumentSummary | None:
    """The Document called `name`'s summarising status, Summary and metadata."""
    with db.pool().connection() as conn:
        row = conn.execute(
            """SELECT d.summary_status, d.document_date, d.topic, s.text
               FROM documents d
               LEFT JOIN summaries s ON s.document_id = d.id AND s.kind = %s
               WHERE d.name = %s""",
            (DOCUMENT, name),
        ).fetchone()
    if row is None:
        return None
    return DocumentSummary(row["summary_status"], row["text"], row["document_date"], row["topic"])


def pending_count() -> int:
    """How many Documents are still waiting for their Summary or their RAPTOR tree."""
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT count(*) AS n FROM documents WHERE summary_status = %s OR raptor_status = %s",
            (PENDING, PENDING),
        ).fetchone()
    return int(row["n"])


def ready_documents() -> set[str]:
    """The names of the Documents whose Summaries are ready: the only ones that
    Summary-based Retrieval Techniques search."""
    with db.pool().connection() as conn:
        rows = conn.execute(
            "SELECT name FROM documents WHERE summary_status = %s", (READY,)
        ).fetchall()
    return {row["name"] for row in rows}


def summarise_pending() -> int:
    """Summarise every pending Document, one at a time, then build the pending RAPTOR
    trees (`raptor.build_pending`), and return how many Summaries were tried. Each ends
    `ready` or, if anything fails, `failed`. Slow with a real local model. Without a
    local summary model, Documents stay pending."""
    from perry import raptor  # raptor writes its nodes through this module
    if not models.summaries_supported():
        return 0
    tried = 0
    while (pending := _next_pending()) is not None:
        document_id, name = pending
        tried += 1
        try:
            _summarise(document_id)
        except Exception:
            log.warning("Couldn't summarise %s; will retry on the next start", name, exc_info=True)
            _set_status(document_id, FAILED)
    raptor.build_pending()
    return tried


def _next_pending() -> tuple[int, str] | None:
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT id, name FROM documents WHERE summary_status = %s ORDER BY id LIMIT 1",
            (PENDING,),
        ).fetchone()
    return (row["id"], row["name"]) if row else None


def _set_status(document_id: int, status: str) -> None:
    with db.pool().connection() as conn:
        conn.execute(
            "UPDATE documents SET summary_status = %s WHERE id = %s", (status, document_id)
        )


def _summarise(document_id: int) -> None:
    with db.pool().connection() as conn:
        name = conn.execute(
            "SELECT name FROM documents WHERE id = %s", (document_id,)
        ).fetchone()["name"]
        chunks = conn.execute(
            "SELECT id, text FROM chunks WHERE document_id = %s ORDER BY page, id",
            (document_id,),
        ).fetchall()
    # The slow model calls happen outside any transaction.
    reply = models.summary_model().invoke(
        [
            SystemMessage(SUMMARY_PROMPT),
            HumanMessage(f"Document: {name}\n\n{input_text([c['text'] for c in chunks])}"),
        ]
    )
    summary, document_date, topic = parse_reply(reply.text)
    embedding = models.embed_documents([summary])[0]

    with db.pool().connection() as conn, conn.transaction():
        still_there = conn.execute(
            "SELECT 1 FROM documents WHERE id = %s FOR UPDATE", (document_id,)
        ).fetchone()
        if still_there is None:  # replaced or deleted meanwhile: its new version is pending
            return
        conn.execute(
            "DELETE FROM summaries WHERE document_id = %s AND kind = %s", (document_id, DOCUMENT)
        )
        save_summary(
            conn,
            document_id=document_id,
            kind=DOCUMENT,
            level=1,
            text=summary,
            embedding=embedding,
            chunk_ids=[c["id"] for c in chunks],
        )
        conn.execute(
            """UPDATE documents SET summary_status = %s, document_date = %s, topic = %s
               WHERE id = %s""",
            (READY, document_date, topic, document_id),
        )


def save_summary(
    conn: Connection,
    *,
    document_id: int | None,
    kind: str,
    level: int,
    text: str,
    embedding: list[float],
    chunk_ids: list[int],
) -> int:
    """Store one Summary (a Document's, or a RAPTOR node's with `document_id` None for
    the cross-Document level) linked to every leaf Chunk beneath it, and return its id.
    Call inside the caller's transaction; embed before opening it."""
    summary_id = conn.execute(
        """INSERT INTO summaries (document_id, kind, level, text, embedding)
           VALUES (%s, %s, %s, %s, %s) RETURNING id""",
        (document_id, kind, level, text, Vector(embedding)),
    ).fetchone()["id"]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO summary_chunks (summary_id, chunk_id) VALUES (%s, %s)",
            [(summary_id, chunk_id) for chunk_id in dict.fromkeys(chunk_ids)],
        )
    return summary_id


def parse_reply(text: str) -> tuple[str, date | None, str | None]:
    """The Summary, Document date and topic in the summary model's JSON reply. A reply
    that isn't JSON is taken as the Summary itself, with no metadata."""
    match = re.search(r"\{.*\}", text, re.DOTALL)  # tolerates ```json fences and chatter
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or not str(data.get("summary") or "").strip():
        summary = text.strip()
        if not summary:
            raise ValueError("The summary model replied with nothing")
        return summary, None, None
    topic = str(data.get("topic") or "").strip() or None
    return str(data["summary"]).strip(), _date(data.get("document_date")), topic


def _date(value) -> date | None:
    match = re.match(r"(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?", str(value or ""))
    if not match:
        return None
    year, month, day = (int(part) if part else 1 for part in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


# --- Background job -------------------------------------------------------------

_lock = threading.Lock()
_wake = threading.Event()
_thread: threading.Thread | None = None


def start_background() -> None:
    """Summarise pending Documents on a background thread, without waiting. Called after
    an upload; if the job is already running, it makes another pass when it's done."""
    global _thread
    with _lock:
        _wake.set()
        if _thread is None:
            _thread = threading.Thread(target=_run, name="perry-summaries", daemon=True)
            _thread.start()


def _run() -> None:
    global _thread
    while True:
        with _lock:
            if not _wake.is_set():
                _thread = None
                return
            _wake.clear()
        try:
            summarise_pending()
        except Exception:
            log.exception("Background summarising stopped")


def resume() -> None:
    """On start: retry the Documents whose summarising failed, and finish pending ones,
    in the background."""
    retry_failed()
    start_background()


def retry_failed() -> None:
    """Make the Documents whose summarising (Summary or RAPTOR tree) failed pending again."""
    with db.pool().connection() as conn:
        for column in ("summary_status", "raptor_status"):
            conn.execute(
                f"UPDATE documents SET {column} = %s WHERE {column} = %s", (PENDING, FAILED)
            )
