"""Model provider seam: the only module that knows which model API is used.

Switching to a local model (see the design decisions in GLOSSARY.md / docs/adr)
is a change to this file only. Tests swap the whole provider with `use()`.

Every call made through this module is recorded in the usage ledger (`perry.usage`),
whichever provider is in use. Inside a Turn, a chat or embedding call raises
`usage.TurnTokenCeilingReached` once the Turn has used up its tokens.
"""

import math
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import GoogleRateLimitError

# The provider's "quota exhausted" error, so callers can report it without knowing the provider.
RateLimitError = GoogleRateLimitError

LLM_MODEL = os.getenv("PERRY_LLM_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = os.getenv("PERRY_EMBEDDING_MODEL", "gemini-embedding-001")

# pgvector's HNSW index supports at most 2000 dimensions.
# Changing the model or this number means re-ingesting every Document.
EMBEDDING_DIMENSIONS = 1536


@dataclass(frozen=True)
class Provider:
    """Everything Perry needs from a model API."""

    chat_model: Callable[[], BaseChatModel]
    embed_documents: Callable[[list[str]], list[list[float]]]
    embed_query: Callable[[str], list[float]]
    # The name the usage ledger prices embedding calls by.
    embedding_model: str = "unknown"


@cache
def _gemini() -> Provider:
    # Shared across every chat model so all Agents together stay under the
    # free tier's per-minute request limit (5/min for Flash). No bursts: a burst
    # followed by steady pacing can still exceed the per-minute window.
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=float(os.getenv("PERRY_REQUESTS_PER_SECOND", str(4 / 60))),
        max_bucket_size=1,
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, output_dimensionality=EMBEDDING_DIMENSIONS
    )
    return Provider(
        chat_model=lambda: ChatGoogleGenerativeAI(
            model=LLM_MODEL, temperature=0, rate_limiter=rate_limiter, max_retries=3
        ),
        embed_documents=lambda texts: embeddings.embed_documents(
            texts, task_type="RETRIEVAL_DOCUMENT"
        ),
        embed_query=lambda text: embeddings.embed_query(text, task_type="RETRIEVAL_QUERY"),
        embedding_model=EMBEDDING_MODEL,
    )


_override: Provider | None = None


def _provider() -> Provider:
    return _override or _gemini()


@contextmanager
def use(provider: Provider) -> Iterator[Provider]:
    """Route every model call to `provider` until the block exits (e.g. fakes in tests)."""
    global _override
    previous, _override = _override, provider
    try:
        yield provider
    finally:
        _override = previous


def chat_model() -> BaseChatModel:
    model = _provider().chat_model()
    callbacks = model.callbacks or []
    if not any(isinstance(handler, _UsageRecorder) for handler in callbacks):
        model.callbacks = [*callbacks, _UsageRecorder()]
    return model


def embedding_model() -> str:
    """The name the usage ledger prices embedding calls by."""
    return _provider().embedding_model


def embed_documents(texts: list[str]) -> list[list[float]]:
    _record_embedding(texts)
    return _provider().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    _record_embedding([text])
    return _provider().embed_query(text)


# --- Usage recording -----------------------------------------------------------


def estimate_tokens(texts: list[str]) -> int:
    """Gemini's Developer API doesn't report embedding usage: about 4 characters per token."""
    return sum(math.ceil(len(text) / 4) for text in texts)


def _record_embedding(texts: list[str]) -> None:
    from perry import usage

    usage.check_turn_ceiling()
    usage.record("embedding", embedding_model(), estimate_tokens(texts))


def _model_name(name: str) -> str:
    return name.removeprefix("models/")


class _UsageRecorder(BaseCallbackHandler):
    """Records each chat call in the usage ledger; stops calls past the Turn's ceiling."""

    raise_error = True  # so TurnTokenCeilingReached stops the call instead of being logged

    def __init__(self):
        self._models: dict = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        from perry import usage

        usage.check_turn_ceiling()
        params = kwargs.get("invocation_params") or {}
        # Fallback name for the ledger if the response doesn't name its model (fakes don't).
        self._models[run_id] = (
            params.get("model") or params.get("model_name") or params.get("_type", "unknown")
        )

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        from perry import usage

        fallback = self._models.pop(run_id, "unknown")
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                tokens = (message.usage_metadata if message is not None else None) or {}
                metadata = message.response_metadata if message is not None else {}
                usage.record(
                    "chat",
                    _model_name(metadata.get("model_name") or fallback),
                    tokens.get("input_tokens", 0),
                    tokens.get("output_tokens", 0),
                )

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self._models.pop(run_id, None)
