"""Self-query: the model turns the question into a filter over Document metadata plus a
search query, and hybrid search looks only in matching Documents.

The filter covers what Perry knows about each Document: its name, file type, upload
date, pages, and the Document date and topic the summarising job extracts. A question
that names none of these (or a reply the model got wrong) has no usable filter, and
Self-query is then plain hybrid search for the question.
"""

import calendar
import json
import re
from datetime import date, datetime

from langchain_core.runnables import chain

from perry import db, models
from perry.search import DocumentFilter, NoHits, SearchHit, hybrid_search

# How many Documents the prompt lists, so the model can match the user's wording to them.
LISTED_DOCUMENTS = 100

PROMPT = """You turn a question about the user's Documents into a search query plus a \
filter on which Documents to search. Today is {today}.

The user's Documents (name; topic; Document date, if known):
{documents}

Reply with only a JSON object with these keys, using null for anything the question \
does not explicitly limit:
- "query": the question without the words that only say which Documents to search.
- "name": words of the Document name the question names, e.g. "2025 budget".
- "file_type": a file type the question names, e.g. "pdf" or "md".
- "uploaded_from", "uploaded_to": the period in which the Documents were uploaded, as \
YYYY-MM-DD (inclusive).
- "page_from", "page_to": the pages the question names, as numbers (inclusive).
- "dated_from", "dated_to": the period the Documents are dated in or are about, as \
YYYY-MM-DD (inclusive); e.g. "from 2025" is 2025-01-01 to 2025-12-31.
- "topic": a word or two for the subject the Documents must be about, only when the \
question names one of the topics listed above.

Question: {question}"""


@chain
def self_query(question: str) -> list[SearchHit]:
    """Hybrid search for the question within the Documents its filter allows; without a
    usable filter, plain hybrid search for the question."""
    prompt = PROMPT.format(
        today=datetime.now().astimezone().date().isoformat(),
        documents=_documents(),
        question=question,
    )
    reply = models.chat_model(kind="self_query").invoke(prompt).text
    query, where = parse_reply(reply, question)
    if not where:
        return hybrid_search(question)
    # Strict: a filter no Document matches finds nothing, and says so, rather than
    # quietly searching every Document.
    return hybrid_search(query, where=where) or NoHits(
        f"No Document matches the filter ({where.describe()})"
    )


def _documents() -> str:
    with db.pool().connection() as conn:
        rows = conn.execute(
            """SELECT name, topic, document_date FROM documents
               ORDER BY ingested_at DESC LIMIT %s""",
            (LISTED_DOCUMENTS,),
        ).fetchall()
    lines = [
        f"- {r['name']}; {r['topic'] or 'unknown'}; {r['document_date'] or 'unknown'}"
        for r in rows
    ]
    return "\n".join(lines) or "(none)"


def parse_reply(reply: str, question: str) -> tuple[str, DocumentFilter]:
    """The search query and filter in the model's JSON reply. Anything unusable is left
    out of the filter; a reply that isn't JSON gives no filter and the question itself."""
    match = re.search(
        r"\{.*\}", reply, re.DOTALL
    )  # tolerates ```json fences and chatter
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    query = _text(data.get("query")) or question
    where = DocumentFilter(
        name=_text(data.get("name")),
        file_type=_text(data.get("file_type")),
        uploaded_from=_date(data.get("uploaded_from"), end=False),
        uploaded_to=_date(data.get("uploaded_to"), end=True),
        page_from=_page(data.get("page_from")),
        page_to=_page(data.get("page_to")),
        dated_from=_date(data.get("dated_from"), end=False),
        dated_to=_date(data.get("dated_to"), end=True),
        topic=_text(data.get("topic")),
    )
    return query, where


def _text(value) -> str | None:
    return (value.strip() or None) if isinstance(value, str) else None


def _page(value) -> int | None:
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page if page >= 1 else None


def _date(value, *, end: bool) -> date | None:
    """A YYYY-MM-DD date; a bare year or year-month means its first day, or its last
    day when it ends a range."""
    match = re.fullmatch(r"(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?", _text(value) or "")
    if not match:
        return None
    year, month, day = (int(part) if part else None for part in match.groups())
    month = month or (12 if end else 1)
    try:
        if day is None:
            day = calendar.monthrange(year, month)[1] if end else 1
        return date(year, month, day)
    except ValueError:
        return None
