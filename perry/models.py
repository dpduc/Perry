"""Model provider seam: the only module that knows which model API is used.

Switching to a local model (see the design decisions in GLOSSARY.md / docs/adr)
is a change to this file only.
"""

import os

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

# Shared across every chat model so all Agents together stay under the
# free tier's per-minute request limit (5/min for Flash). No bursts: a burst
# followed by steady pacing can still exceed the per-minute window.
_rate_limiter = InMemoryRateLimiter(
    requests_per_second=float(os.getenv("PERRY_REQUESTS_PER_SECOND", str(4 / 60))),
    max_bucket_size=1,
)


def chat_model() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL, temperature=0, rate_limiter=_rate_limiter, max_retries=3
    )


_embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL, output_dimensionality=EMBEDDING_DIMENSIONS
)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _embeddings.embed_documents(texts, task_type="RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    return _embeddings.embed_query(text, task_type="RETRIEVAL_QUERY")
