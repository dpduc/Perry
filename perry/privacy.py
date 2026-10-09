"""Privacy switches: what Perry refuses to do with a real Google account.

Every connected Google account is declared a test account (fake data only) or a
real one. A real account's calendar and email must never reach a cloud model or a
trace, so:
- connecting a real account is refused unless text generation is local;
- the Task Agent refuses to run on a real account unless text generation is local;
- tracing is switched off while a real account is connected, whatever .env says.
"""

import os

import langsmith

from perry import models, workspace
from perry.workspace import AccountKind

__all__ = [
    "AccountKind",
    "Refused",
    "check_connect",
    "check_task_agent",
    "connected",
    "startup",
]

# Every variable LangSmith/LangChain read to decide whether to trace.
_TRACING_VARIABLES = (
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING",
)


class Refused(Exception):
    """Perry won't do this with a real account. The message is written for the user."""


def check_connect(kind: AccountKind) -> None:
    """Raise Refused if an account of this kind may not be connected right now."""
    if kind not in ("test", "real"):
        raise ValueError(f"An account is either 'test' or 'real', not {kind!r}")
    if kind == "real" and not models.text_generation_is_local():
        raise Refused(
            "Perry won't connect a real Google account while text generation uses a "
            "cloud model: your email would be sent to it. Switch text generation to a "
            "local model first, or connect a test account."
        )


def connected(kind: AccountKind) -> None:
    """Apply the switches as soon as an account of this kind is connected."""
    if kind == "real":
        _turn_tracing_off()


def _turn_tracing_off() -> None:
    for name in _TRACING_VARIABLES:
        os.environ[name] = "false"  # for anything that reads the environment later
    # LangSmith caches what it read from the environment, so also switch it off directly.
    langsmith.configure(enabled=False)


def _real_account_connected() -> bool:
    return workspace.current().account_kind() == "real"


def _task_agent_refusal() -> str | None:
    if _real_account_connected() and not models.text_generation_is_local():
        return (
            "The Task Agent is off: a real Google account is connected (or one connected "
            "without saying it was a test account) and text generation uses a cloud model, "
            "which would see your email. Switch text generation to a local model, or "
            "reconnect Google as a test account."
        )
    return None


def check_task_agent() -> None:
    """Raise Refused if the Task Agent may not read the connected account."""
    if _real_account_connected():
        _turn_tracing_off()
    if refusal := _task_agent_refusal():
        raise Refused(refusal)


def startup() -> str | None:
    """Apply the privacy switches when Perry starts. Returns a notice for the user
    if the Task Agent is off."""
    if _real_account_connected():
        _turn_tracing_off()
    return _task_agent_refusal()
