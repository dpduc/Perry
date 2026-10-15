"""The usage ledger: every model call's tokens and cost, and the per-Turn token ceiling.

`perry.models` records each call here; nothing else needs to. Calls made inside a
`turn(user, thread_id)` block are attributed to that Turn, its Thread and its User,
and count toward its token ceiling. Calls outside one (e.g. ingestion) have no
Thread, Turn or User.

Prices come from a TOML price table (USD per million tokens, per model), read on
every call so edits apply without a restart:

    [models."gemini-3.8-flash"]
    input_per_million = 0.30
    output_per_million = 2.50
"""

import logging
import os
import re
import tomllib
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, tzinfo
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from perry import db

if TYPE_CHECKING:
    from perry.users import User

log = logging.getLogger(__name__)

DEFAULT_PRICES_FILE = Path(__file__).resolve().parent.parent / "prices.toml"
DEFAULT_TURN_TOKEN_CEILING = 50_000
DEFAULT_DAILY_CAP_USD = 0.5
# The date a provider appends to name a model's snapshot, e.g. "-2026-03-17".
_SNAPSHOT_DATE = re.compile(r"-\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class Usage:
    tokens: int
    cost_usd: float


@dataclass(frozen=True)
class Turn:
    thread_id: str | None
    id: str
    user_id: int | None = None
    time_zone: str | None = None  # the User's Profile time zone


class DailyCapReached(Exception):
    """Today's usage has reached the daily cap; nothing new may start until local midnight."""

    def __init__(self, cap_usd: float):
        super().__init__(
            f"Perry has used today's budget of ${cap_usd:.2f}, so it can't take new requests "
            "until the budget resets at midnight."
        )
        self.cap_usd = cap_usd


class TurnTokenCeilingReached(Exception):
    """The current Turn has used its token budget; no further model calls are allowed."""

    def __init__(self, ceiling: int):
        super().__init__(f"This Turn reached its token ceiling of {ceiling:,} tokens.")
        self.ceiling = ceiling


_current: ContextVar[Turn | None] = ContextVar("perry_turn", default=None)


@contextmanager
def turn(
    user: "User | None", thread_id: str | None, turn_id: str | None = None
) -> Iterator[Turn]:
    """Attribute every model call in this block to `user`'s Turn `turn_id` (by default a
    new one) on `thread_id`. Re-entering a Turn (e.g. after its Confirmation) keeps adding
    to its token ceiling."""
    current = Turn(
        thread_id,
        turn_id or new_turn_id(),
        user.id if user else None,
        user.profile.time_zone if user else None,
    )
    token = _current.set(current)
    try:
        yield current
    finally:
        _current.reset(token)


def new_turn_id() -> str:
    return uuid.uuid4().hex


def current_turn() -> Turn | None:
    return _current.get()


def local_zone() -> tzinfo:
    """The time zone "today" means: the current Turn's User's (from their Profile), or
    outside a Turn this machine's."""
    current = current_turn()
    if current is not None and current.time_zone:
        return ZoneInfo(current.time_zone)
    return datetime.now().astimezone().tzinfo


def local_now() -> datetime:
    """The time now in `local_zone()`."""
    return datetime.now(local_zone())


def turn_token_ceiling() -> int:
    return int(os.getenv("PERRY_TURN_TOKEN_CEILING", str(DEFAULT_TURN_TOKEN_CEILING)))


def check_turn_ceiling() -> None:
    """Raise TurnTokenCeilingReached if the current Turn has used up its tokens."""
    current = current_turn()
    if current is None:
        return
    ceiling = turn_token_ceiling()
    if turn_tokens(current.id) >= ceiling:
        raise TurnTokenCeilingReached(ceiling)


def daily_cap_usd() -> float:
    return float(os.getenv("PERRY_DAILY_CAP_USD", str(DEFAULT_DAILY_CAP_USD)))


def remaining_today() -> float:
    """USD left under the daily cap today (never below 0)."""
    return max(0.0, daily_cap_usd() - today().cost_usd)


def check_daily_cap() -> None:
    """Raise DailyCapReached if today's usage has reached the daily cap.

    Checked before starting something new (a Turn), never during it, so a Turn
    that is already running always finishes. Anything that calls a model before
    the Turn starts (e.g. transcribing a voice request) must check it first too.
    """
    if remaining_today() <= 0:
        raise DailyCapReached(daily_cap_usd())


def daily_cap_refusal() -> str | None:
    """What to tell the user when the daily cap stops something new; None if it doesn't."""
    try:
        check_daily_cap()
    except DailyCapReached as refused:
        return str(refused)
    return None


def record(kind: str, model: str, input_tokens: int, output_tokens: int = 0) -> None:
    """Add one model call to the ledger, priced with the current price table.

    `kind` says what the call was for, e.g. "chat", "vision" (a chat call given an
    image) or "embedding".
    """
    current = current_turn()
    with db.pool().connection() as conn:
        conn.execute(
            """INSERT INTO usage_ledger
               (thread_id, turn_id, user_id, kind, model, input_tokens, output_tokens,
                cost_usd)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                current.thread_id if current else None,
                current.id if current else None,
                current.user_id if current else None,
                kind,
                model,
                input_tokens,
                output_tokens,
                cost(model, input_tokens, output_tokens),
            ),
        )


def cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """USD for a call, from the price table. A dated snapshot ("gpt-5.4-mini-2026-03-17",
    as OpenAI's replies name it) is priced as its model. Unpriced models cost 0 (with a
    warning)."""
    prices = _prices()
    rates = prices.get(model) or prices.get(_SNAPSHOT_DATE.sub("", model))
    if rates is None:
        log.warning("No price for model %r in %s; recording its cost as 0", model, _prices_file())
        return 0.0
    return (
        input_tokens * rates.get("input_per_million", 0.0)
        + output_tokens * rates.get("output_per_million", 0.0)
    ) / 1_000_000


def turn_tokens(turn_id: str) -> int:
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT coalesce(sum(input_tokens + output_tokens), 0) AS tokens "
            "FROM usage_ledger WHERE turn_id = %s",
            (turn_id,),
        ).fetchone()
    return int(row["tokens"])


def turn_usage(turn_id: str) -> Usage:
    """Everything recorded for the Turn `turn_id`."""
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT coalesce(sum(input_tokens + output_tokens), 0) AS tokens, "
            "coalesce(sum(cost_usd), 0) AS cost FROM usage_ledger WHERE turn_id = %s",
            (turn_id,),
        ).fetchone()
    return Usage(int(row["tokens"]), float(row["cost"]))


def turn_models(turn_id: str, excluding_kinds: tuple[str, ...] = ()) -> list[str]:
    """The models the Turn `turn_id` called, sorted, leaving out calls of `excluding_kinds`."""
    with db.pool().connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT model FROM usage_ledger "
            "WHERE turn_id = %s AND NOT (kind = ANY(%s)) ORDER BY model",
            (turn_id, list(excluding_kinds)),
        ).fetchall()
    return [row["model"] for row in rows]


def since(start: datetime, kind: str | None = None) -> Usage:
    """Everything recorded from `start` (timezone-aware) until now, optionally of one kind."""
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT coalesce(sum(input_tokens + output_tokens), 0) AS tokens, "
            "coalesce(sum(cost_usd), 0) AS cost FROM usage_ledger "
            "WHERE recorded_at >= %s AND (%s::text IS NULL OR kind = %s)",
            (start, kind, kind),
        ).fetchone()
    return Usage(int(row["tokens"]), float(row["cost"]))


def today(kind: str | None = None) -> Usage:
    """Usage since local midnight, optionally of one kind (e.g. "vision")."""
    midnight = datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
    return since(midnight, kind)


def _prices_file() -> Path:
    return Path(os.getenv("PERRY_PRICES_FILE", str(DEFAULT_PRICES_FILE)))


def _prices() -> dict[str, dict[str, float]]:
    path = _prices_file()
    if not path.exists():
        return {}
    with path.open("rb") as file:
        return tomllib.load(file).get("models", {})
