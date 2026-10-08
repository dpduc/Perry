"""Model provider seam: the only module that knows which model API is used.

Switching to a local model (see the design decisions in GLOSSARY.md / docs/adr)
is a change to this file only. Tests swap the whole provider with `use()`.
"""

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache

import httpx
from langchain_core.exceptions import ModelError
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import GoogleRateLimitError

# The provider's "quota exhausted" error, so callers can report it without knowing the provider.
RateLimitError = GoogleRateLimitError


def is_transient(error: BaseException) -> bool:
    """Whether a call that raised `error` could succeed if tried again a little later:
    rate limits, timeouts, dropped connections and the provider's own server errors."""
    if isinstance(error, ModelError):
        return error.is_retryable
    # The Gemini client lets network timeouts and connection drops through as httpx errors.
    return isinstance(error, (TimeoutError, ConnectionError, httpx.TransportError))

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
    return _provider().chat_model()


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _provider().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    return _provider().embed_query(text)
