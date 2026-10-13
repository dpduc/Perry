"""Thread titles. The conversation itself lives in the LangGraph checkpointer."""

import uuid
from typing import TYPE_CHECKING

from perry import db

if TYPE_CHECKING:
    from perry.users import User

TITLE_CHARS = 60


class ThreadNotFound(LookupError):
    """The Thread doesn't exist, or isn't the User's: the two look the same to them."""

    def __init__(self):
        super().__init__("That Thread doesn't exist.")


def create(user: "User", first_message: str) -> str:
    """A new Thread belonging to `user`, titled from its first message."""
    thread_id = uuid.uuid4().hex
    title = " ".join(first_message.split())[:TITLE_CHARS] or "New thread"
    with db.pool().connection() as conn:
        conn.execute(
            "INSERT INTO threads (id, title, user_id) VALUES (%s, %s, %s)",
            (thread_id, title, user.id),
        )
    return thread_id


def belongs_to(thread_id: str, user: "User") -> bool:
    """Whether the Thread exists and is `user`'s."""
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM threads WHERE id = %s AND user_id = %s", (thread_id, user.id)
        ).fetchone()
    return row is not None


def require(thread_id: str, user: "User") -> None:
    """Raise ThreadNotFound unless the Thread exists and is `user`'s."""
    if not belongs_to(thread_id, user):
        raise ThreadNotFound()


def list_recent(user: "User", limit: int = 50) -> list[tuple[str, str]]:
    """`user`'s Threads as (title, id) pairs, newest first: the shape a Gradio dropdown wants."""
    with db.pool().connection() as conn:
        rows = conn.execute(
            "SELECT id, title FROM threads WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
            (user.id, limit),
        ).fetchall()
    return [(r["title"], r["id"]) for r in rows]
