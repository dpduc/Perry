"""Thread titles. The conversation itself lives in the LangGraph checkpointer."""

import uuid
from typing import TYPE_CHECKING

from perry import db

if TYPE_CHECKING:
    from perry.users import User

TITLE_CHARS = 60


def create(first_message: str) -> str:
    thread_id = uuid.uuid4().hex
    title = " ".join(first_message.split())[:TITLE_CHARS] or "New thread"
    with db.pool().connection() as conn:
        conn.execute("INSERT INTO threads (id, title) VALUES (%s, %s)", (thread_id, title))
    return thread_id


def belongs_to(thread_id: str, user: "User") -> bool:
    """Whether the Thread exists and is `user`'s."""
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM threads WHERE id = %s AND user_id = %s", (thread_id, user.id)
        ).fetchone()
    return row is not None


def list_recent(limit: int = 50) -> list[tuple[str, str]]:
    """(title, id) pairs, newest first: the shape a Gradio dropdown wants."""
    with db.pool().connection() as conn:
        rows = conn.execute(
            "SELECT id, title FROM threads ORDER BY created_at DESC LIMIT %s", (limit,)
        ).fetchall()
    return [(r["title"], r["id"]) for r in rows]
