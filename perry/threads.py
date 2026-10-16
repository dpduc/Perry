"""Threads, each belonging to one User: `open` one to send it Turns and read what it
keeps. Titles and owners live here; the conversation lives in the LangGraph checkpointer."""

import uuid
from dataclasses import dataclass
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


def open(user: "User", thread_id: str | None) -> "Thread":
    """`user`'s Thread `thread_id`, or a new one (None) that starts with its first Turn.
    Raises ThreadNotFound if the Thread isn't `user`'s."""
    if thread_id is not None:
        require(thread_id, user)
    return Thread(user, thread_id)


def title(user: "User", thread_id: str) -> str:
    """`user`'s Thread title. Raises ThreadNotFound if it isn't `user`'s."""
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT title FROM threads WHERE id = %s AND user_id = %s", (thread_id, user.id)
        ).fetchone()
    if row is None:
        raise ThreadNotFound()
    return row["title"]


def delete(user: "User", thread_id: str) -> None:
    """Delete `user`'s Thread and its saved conversation state."""
    require(thread_id, user)
    from perry import supervisor

    supervisor.delete_thread(thread_id)
    _delete(thread_id)


@dataclass(frozen=True)
class Heard:
    """A voice request, transcribed: send it to run the Turn it began, then read the
    answer aloud with it. The chat UI holds it between the two without looking inside."""

    transcript: str
    turn_id: str


class Thread:
    """A Thread as its User sees it: Turns are sent to it and it keeps what they said.
    Get one with `open`, which checks it is the User's."""

    def __init__(self, user: "User", thread_id: str | None):
        self.user = user
        self.id = thread_id

    def hear(self, audio: bytes) -> Heard:
        """Transcribe a voice request; its Turn starts here. Raises usage.DailyCapReached,
        before any model call, once today's usage has reached the Daily Cap."""
        from perry import voice

        turn_id = voice.new_turn()
        return Heard(voice.transcribe(self.user, audio, self.id, turn_id), turn_id)

    def send(self, message: "str | Heard", images=(), technique: str | None = None):
        """Run a Turn (the one a Heard voice request began) and return its answer, or the
        Confirmation it waits for. A new Thread exists only once a Turn is kept in it: a
        first message refused (e.g. by the Daily Cap) or forgotten after a failure
        leaves no Thread behind."""
        from perry import supervisor

        text, turn_id = (
            (message.transcript, message.turn_id)
            if isinstance(message, Heard)
            else (message, None)
        )
        new = self.id is None
        thread_id = self.id or create(self.user, text)
        try:
            outcome = supervisor.run_turn(
                self.user,
                thread_id,
                text,
                images=list(images),
                turn_id=turn_id,
                technique=technique,
            )
        except BaseException:
            if new:
                _delete(thread_id)
            raise
        if new and not supervisor.thread_history(thread_id):
            _delete(thread_id)
        else:
            self.id = thread_id
        return outcome

    def pending_confirmation(self):
        """The supervisor.Confirmation the Thread waits for, if any."""
        from perry import supervisor

        return supervisor.pending_confirmation(self.id) if self.id else None

    def approve(self, edits: dict | None = None):
        """Approve the pending Write Action, with any fields edited, and return what the
        Turn shows next: its answer or another Confirmation."""
        from perry import supervisor

        return self._resume(supervisor.Approve(edits or {}))

    def reject(self, note: str = ""):
        """Reject the pending Write Action; Perry revises with the note, if there is one."""
        from perry import supervisor

        return self._resume(supervisor.Reject(note.strip()))

    def _resume(self, decision):
        from perry import supervisor

        if self.id is None:
            return supervisor.NOTHING_PENDING
        return supervisor.resume_turn(self.user, self.id, decision)

    def read_aloud(self, heard: Heard, answer: str):
        """The answer to a voice request as speech, in that request's Turn: a voice.Speech,
        whose note says why there is no audio when a limit stopped it."""
        from perry import voice

        return voice.read_aloud(self.user, answer, self.id, heard.turn_id)

    def history(self) -> list[dict]:
        """The Thread's messages, oldest first; none until its first Turn is kept."""
        from perry import supervisor

        return supervisor.thread_history(self.id) if self.id else []


def _delete(thread_id: str) -> None:
    with db.pool().connection() as conn:
        conn.execute("DELETE FROM threads WHERE id = %s", (thread_id,))


def list_recent(user: "User", limit: int = 50) -> list[tuple[str, str]]:
    """`user`'s Threads as (title, id) pairs, newest first: the shape a Gradio dropdown wants."""
    with db.pool().connection() as conn:
        rows = conn.execute(
            "SELECT id, title FROM threads WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
            (user.id, limit),
        ).fetchall()
    return [(r["title"], r["id"]) for r in rows]
