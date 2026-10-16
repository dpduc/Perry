"""Model provider seam: the only module that knows which model API is used.

Which provider does what is decided here (docs/adr/0003); switching a model
is a change to this file only. Tests swap the whole provider with `use()`.

Every call made through this module is recorded in the usage ledger (`perry.usage`),
whichever provider is in use. Inside a Turn, a chat or embedding call raises
`usage.TurnTokenCeilingReached` once the Turn has used up its tokens.
"""

import base64
import math
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from functools import cache, cached_property
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import openai
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from langchain_core.exceptions import ModelError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import GoogleRateLimitError
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import ConfigDict

from perry import images

# The free model's "quota exhausted" error (tests raise it to simulate one).
RateLimitError = GoogleRateLimitError
# Every provider's "rate limit or quota reached" error, so callers can report one without
# knowing the provider.
RATE_LIMIT_ERRORS = (GoogleRateLimitError, openai.RateLimitError)


def is_transient(error: BaseException) -> bool:
    """Whether a call that raised `error` could succeed if tried again a little later:
    rate limits, timeouts, dropped connections and the provider's own server errors."""
    if isinstance(error, ModelError):
        return error.is_retryable
    if isinstance(error, (openai.RateLimitError, openai.APIConnectionError)):
        return True  # APITimeoutError is an APIConnectionError
    if isinstance(error, openai.APIStatusError):
        return error.status_code >= 500
    # The Gemini and Ollama clients let network timeouts and connection drops through as
    # httpx errors (e.g. ConnectError when Ollama isn't running).
    return isinstance(error, (TimeoutError, ConnectionError, httpx.TransportError))


# Chat runs on Gemini's free tier, and on the paid OpenAI model only when Gemini's free
# quota is used up or it stays overloaded (see `_FreeFirst`). Without OPENAI_API_KEY there
# is no fallback.
LLM_MODEL = os.getenv("PERRY_LLM_MODEL", "gemini-3.8-flash")
# What a person registering is told about where their messages go.
CHAT_DATA_NOTICE = (
    "Perry answers with Google Gemini or, as a fallback, OpenAI: your messages are sent "
    "to Google or OpenAI and may be logged."
)
FALLBACK_LLM_MODEL = os.getenv("PERRY_FALLBACK_LLM_MODEL", "gpt-5.4-mini")
# The benchmark grid runs every Turn on one paid model, never mixing in the free one.
BENCHMARK_LLM_MODEL = os.getenv("PERRY_BENCHMARK_LLM_MODEL", "gpt-5.4-nano")
# GPT-5 models can think before answering, billed as output. "none" is the cheapest, and
# the only setting the chat completions API accepts together with tools for these models.
OPENAI_REASONING_EFFORT = os.getenv("PERRY_OPENAI_REASONING_EFFORT", "none")
# Seconds one free chat call may take before it counts as overloaded and the paid model
# answers instead. Without it, an overloaded Gemini call can hold a Turn for minutes.
CHAT_TIMEOUT = float(os.getenv("PERRY_CHAT_TIMEOUT", "45"))
# Embeddings run on this machine (docs/adr/0003): free, no quota, and nothing embedded
# leaves it. Changing the model re-embeds every Chunk and Summary when Perry next starts.
EMBEDDING_MODEL = os.getenv("PERRY_EMBEDDING_MODEL", "nomic-embed-text")
# Text-to-speech: a Gemini model whose name ends in "-tts" answers with audio.
TTS_MODEL = os.getenv("PERRY_TTS_MODEL", "gemini-2.5-flash-preview-tts")
TTS_VOICE = os.getenv("PERRY_TTS_VOICE", "Kore")
# The local model (Ollama) that writes Summaries and extracts Document metadata in the
# background, so they cost nothing and the Documents never leave this machine.
SUMMARY_MODEL = os.getenv("PERRY_SUMMARY_MODEL", "qwen3:8b")
# 127.0.0.1, not localhost: on Windows, localhost tries IPv6 first and stalls.
OLLAMA_URL = os.getenv("PERRY_OLLAMA_URL", "http://127.0.0.1:11434")
# Ollama's default context is 4096 tokens and silently cuts longer prompts.
SUMMARY_CONTEXT_TOKENS = int(os.getenv("PERRY_SUMMARY_CONTEXT_TOKENS", "8192"))
# The local late-interaction model behind the ColBERT Retrieval Technique (ADR-0002).
# Changing it re-encodes every Chunk when Perry next starts. Empty turns ColBERT off.
DEFAULT_COLBERT_MODEL = "answerdotai/answerai-colbert-small-v1"
COLBERT_MODEL = os.getenv("PERRY_COLBERT_MODEL", DEFAULT_COLBERT_MODEL).strip()
# The local cross-encoder that reranks every Retrieval Technique's Chunks. Empty turns
# reranking off. For Documents not in English, try BAAI/bge-reranker-v2-m3 (multilingual,
# slower on a CPU).
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_MODEL = os.getenv("PERRY_RERANK_MODEL", DEFAULT_RERANK_MODEL).strip()
# If set, a Chunk the reranker scores below it is left out (the best one is always kept).
# Off by default: on the benchmark every required page scored above -6.4, but on the
# eval set the default model scored the one Document that answers "which file is about
# the vegetable patch?" at -11 (its words are tomatoes and basil), so any cutoff that
# drops noise there drops the answer too.
RERANK_MIN_SCORE = (
    float(os.environ["PERRY_RERANK_MIN_SCORE"])
    if os.getenv("PERRY_RERANK_MIN_SCORE", "").strip()
    else None
)

# nomic-embed-text's size. pgvector's HNSW index supports at most 2000 dimensions.
# Changing it resizes the columns and re-embeds everything (perry.reembed).
EMBEDDING_DIMENSIONS = int(os.getenv("PERRY_EMBEDDING_DIMENSIONS", "768"))
# nomic-embed-text is trained with these task prefixes; other models get none.
_EMBEDDING_PREFIXES = (
    ("search_document: ", "search_query: ") if EMBEDDING_MODEL.startswith("nomic-embed") else ("", "")
)


@dataclass(frozen=True)
class Transcription:
    """What a speech-to-text call heard, with the model and tokens the ledger records."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Speech:
    """WAV audio from a text-to-speech call, with the model and tokens the ledger records."""

    wav: bytes
    model: str
    input_tokens: int
    output_tokens: int


class ColbertEncoder(Protocol):
    """Turns text into ColBERT token vectors: one row per token, for late interaction."""

    model: str  # the name the usage ledger records (and prices) its calls by

    def encode_documents(self, texts: list[str]) -> list[np.ndarray]: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class LocalColbert:
    """A ColBERT model run on this machine's CPU with PyLate. Loaded on first use: importing
    torch takes tens of seconds, and the first use downloads the model (~130 MB)."""

    def __init__(self, model: str = DEFAULT_COLBERT_MODEL):
        self.model = model

    def encode_documents(self, texts: list[str]) -> list[np.ndarray]:
        return list(self._loaded.encode(texts, is_query=False, show_progress_bar=False))

    def encode_query(self, text: str) -> np.ndarray:
        return self._loaded.encode([text], is_query=True, show_progress_bar=False)[0]

    @cached_property
    def _loaded(self):
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from pylate import models as pylate_models

        # 512 tokens (the model's maximum) so no Chunk is cut short; the default is 300.
        return pylate_models.ColBERT(self.model, device="cpu", document_length=512)


class Reranker(Protocol):
    """Scores how well each text answers a question, reading the two together."""

    model: str  # the name the usage ledger records (and prices) its calls by

    def score(self, question: str, texts: list[str]) -> list[float]: ...


class LocalReranker:
    """A cross-encoder run on this machine's CPU with sentence-transformers. Loaded on
    first use: importing torch takes tens of seconds, and the first use downloads it."""

    def __init__(self, model: str = DEFAULT_RERANK_MODEL):
        self.model = model

    def score(self, question: str, texts: list[str]) -> list[float]:
        pairs = [(question, text) for text in texts]
        return [float(s) for s in self._loaded.predict(pairs, show_progress_bar=False)]

    @cached_property
    def _loaded(self):
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.model, device="cpu", max_length=512)


class SpeechUnavailable(Exception):
    """The current model provider can't transcribe or synthesise speech."""


class VisionUnavailable(Exception):
    """The current model provider can't read images."""


class SummariesUnavailable(Exception):
    """The current model provider has no local summary model."""


@dataclass(frozen=True)
class Provider:
    """Everything Perry needs from a model API."""

    chat_model: Callable[[], BaseChatModel]
    embed_documents: Callable[[list[str]], list[list[float]]]
    embed_query: Callable[[str], list[float]]
    # The name the usage ledger prices embedding calls by.
    embedding_model: str = "unknown"
    # Speech, if the provider has it: (audio bytes, MIME type) -> text, and text -> WAV.
    transcribe: Callable[[bytes, str], Transcription] | None = None
    synthesize: Callable[[str], Speech] | None = None
    # Vision, if the provider has it: (image bytes, MIME type) -> the text on the image,
    # and the name the usage ledger prices those calls by.
    read_image: Callable[[bytes, str], Transcription] | None = None
    vision_model: str = "unknown"
    # True only when chat (text generation) runs on this machine, so nothing the
    # Agents read leaves it. Embeddings may still be remote: they never see email.
    local_text_generation: bool = False
    # The local text model that writes Summaries and extracts Document metadata in the
    # background, if the provider has one, and the name the usage ledger prices it by.
    summary_model: Callable[[], BaseChatModel] | None = None
    summary_model_name: str = "unknown"
    # The ColBERT encoder, if there is one. Without it, ingestion leaves the ColBERT
    # index alone and the ColBERT Retrieval Technique finds nothing.
    colbert: ColbertEncoder | None = None
    # The reranker, if there is one. Without it, searches keep their technique's order.
    reranker: Reranker | None = None


class FreeQuota:
    """Whether the free model's daily quota is used up. Gemini's free tier resets at
    midnight Pacific time; until then, chat goes straight to the paid model."""

    RESETS_IN = ZoneInfo("America/Los_Angeles")

    def __init__(self):
        self._until: datetime | None = None
        self._lock = threading.Lock()

    def exhausted(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(self.RESETS_IN)
        with self._lock:
            return self._until is not None and now < self._until

    def use_up(self, now: datetime | None = None) -> None:
        now = (now or datetime.now(self.RESETS_IN)).astimezone(self.RESETS_IN)
        midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        with self._lock:
            self._until = midnight

    def reset(self) -> None:
        with self._lock:
            self._until = None


free_quota = FreeQuota()


def is_daily_quota(error: BaseException) -> bool:
    """Whether `error` says the free model's quota for the day is used up (Gemini names
    the quota it hit, e.g. GenerateRequestsPerDayPerProjectPerModel-FreeTier)."""
    return isinstance(error, GoogleRateLimitError) and "PerDay" in str(error)


def _falls_back(error: BaseException) -> bool:
    """Whether a failed free call goes to the paid model: the free quota is used up
    (daily or per minute) or the provider stays overloaded after the client's retries."""
    return isinstance(error, GoogleRateLimitError) or is_transient(error)


class _FreeFirst(BaseChatModel):
    """A chat model that answers on the free model when it can and on the paid one when
    it can't (see `_falls_back`). Agents get it like any chat model: `bind_tools` keeps
    the tools and binds them to whichever model answers each call.

    The usage ledger records each call once, by the model that answered (its reply
    names it); the inner models carry no recorder of their own."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    free: BaseChatModel
    paid: BaseChatModel | None = None
    tools: tuple[list, dict] | None = None

    @property
    def _llm_type(self) -> str:
        return "free-first"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"tools": (list(tools), kwargs)})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        config = {"callbacks": _child_callbacks(run_manager)} if run_manager else None
        if self.paid is None or not free_quota.exhausted():
            try:
                return self._answer(self.free, messages, stop, config, kwargs)
            except Exception as error:
                if self.paid is None or not _falls_back(error):
                    raise
                if is_daily_quota(error):
                    free_quota.use_up()
        return self._answer(self.paid, messages, stop, config, kwargs)

    def _answer(self, model, messages, stop, config, kwargs: dict[str, Any]) -> ChatResult:
        if self.tools is not None:
            tools, options = self.tools
            model = model.bind_tools(tools, **options)
        reply = model.invoke(messages, config, stop=stop, **kwargs)
        # The usage ledger records the call by the model its reply names.
        if not reply.response_metadata.get("model_name"):
            name = getattr(model, "model_name", None) or getattr(model, "model", None)
            reply.response_metadata["model_name"] = name or model._llm_type
        return ChatResult(generations=[ChatGeneration(message=reply)])


def _child_callbacks(run_manager) -> CallbackManager:
    """Callbacks for the inner model's call, nested under this call (e.g. in LangSmith).
    Only inherited handlers pass on, so the usage recorder doesn't record it twice."""
    return CallbackManager(
        handlers=run_manager.inheritable_handlers,
        inheritable_handlers=run_manager.inheritable_handlers,
        parent_run_id=run_manager.run_id,
        tags=run_manager.inheritable_tags,
        inheritable_tags=run_manager.inheritable_tags,
        metadata=run_manager.inheritable_metadata,
        inheritable_metadata=run_manager.inheritable_metadata,
    )


def free_first(free: BaseChatModel, paid: BaseChatModel | None) -> BaseChatModel:
    """A chat model that answers on `free` and falls back to `paid` (see `_FreeFirst`)."""
    return _FreeFirst(free=free, paid=paid)


def _openai_chat(model: str) -> BaseChatModel:
    # No temperature: GPT-5 models only accept the default.
    return ChatOpenAI(model=model, reasoning_effort=OPENAI_REASONING_EFFORT, max_retries=3)


def paid_chat_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


@cache
def _default() -> Provider:
    # Shared across every chat model so all Agents together stay under the
    # free tier's per-minute request limit (5/min for Flash). No bursts: a burst
    # followed by steady pacing can still exceed the per-minute window.
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=float(os.getenv("PERRY_REQUESTS_PER_SECOND", str(4 / 60))),
        max_bucket_size=1,
    )
    # A stopped Ollama raises httpx.ConnectError on the first call, not here.
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
    document_prefix, query_prefix = _EMBEDDING_PREFIXES

    def gemini() -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0,
            rate_limiter=rate_limiter,
            max_retries=3,
            timeout=CHAT_TIMEOUT,
        )

    def chat_model() -> BaseChatModel:
        paid = _openai_chat(FALLBACK_LLM_MODEL) if paid_chat_available() else None
        return free_first(gemini(), paid)

    # The TTS model has its own per-minute quota, so it gets its own pacing.
    tts = ChatGoogleGenerativeAI(
        model=TTS_MODEL,
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=float(os.getenv("PERRY_REQUESTS_PER_SECOND", str(4 / 60))),
            max_bucket_size=1,
        ),
        max_retries=3,
    )

    return Provider(
        chat_model=chat_model,
        embed_documents=lambda texts: embeddings.embed_documents(
            [document_prefix + text for text in texts]
        ),
        embed_query=lambda text: embeddings.embed_query(query_prefix + text),
        embedding_model=EMBEDDING_MODEL,
        # Speech and scanned pages stay on the free model, with no paid fallback.
        transcribe=lambda audio, mime_type: _gemini_transcribe(gemini(), audio, mime_type),
        synthesize=lambda text: _gemini_synthesize(tts, text),
        read_image=lambda image, mime_type: _gemini_read_image(gemini(), image, mime_type),
        vision_model=LLM_MODEL,
        summary_model=_ollama_summary_model,
        summary_model_name=SUMMARY_MODEL,
        colbert=LocalColbert(COLBERT_MODEL) if COLBERT_MODEL else None,
        reranker=LocalReranker(RERANK_MODEL) if RERANK_MODEL else None,
    )


@cache
def _ollama_summary_model() -> ChatOllama:
    # A stopped Ollama raises httpx.ConnectError on the first call, not here.
    return ChatOllama(
        model=SUMMARY_MODEL,
        base_url=OLLAMA_URL,
        temperature=0,
        reasoning=False,  # qwen3's thinking makes a Summary ~18x slower for no gain here
        num_ctx=SUMMARY_CONTEXT_TOKENS,
    )


_TRANSCRIBE_PROMPT = (
    "Transcribe this recording verbatim, in the language it is spoken in. "
    "Reply with the transcript only, no comments. If nothing is said, reply with nothing."
)


def _gemini_transcribe(model: ChatGoogleGenerativeAI, audio: bytes, mime_type: str):
    # Gemini has no separate speech-to-text API: the chat model takes audio as input.
    reply = model.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": _TRANSCRIBE_PROMPT},
                    {
                        "type": "audio",
                        "base64": base64.b64encode(audio).decode(),
                        "mime_type": mime_type,
                    },
                ]
            )
        ]
    )
    return Transcription(reply.text.strip(), *_gemini_usage(reply, LLM_MODEL))


_READ_PAGE_PROMPT = (
    "This is a scanned page of a document. Transcribe all the text on it verbatim, "
    "in reading order and in its original language. Write tables as Markdown tables. "
    "Reply with the text only, no comments. If the page has no text, reply with nothing."
)


def _gemini_read_image(model: ChatGoogleGenerativeAI, image: bytes, mime_type: str):
    reply = model.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": _READ_PAGE_PROMPT},
                    images.image_block(image, mime_type),
                ]
            )
        ]
    )
    return Transcription(reply.text.strip(), *_gemini_usage(reply, LLM_MODEL))


def tts_options() -> dict:
    """Keyword arguments for invoking the text-to-speech model with TTS_VOICE. The voice
    goes in speech_config: a bare `voice_name` is rejected by google-genai."""
    voice = {"prebuilt_voice_config": {"voice_name": TTS_VOICE}}
    return {"generation_config": {"speech_config": {"voice_config": voice}}}


def _gemini_synthesize(model: ChatGoogleGenerativeAI, text: str) -> Speech:
    # A "-tts" model answers with WAV bytes (24 kHz mono) in additional_kwargs["audio"].
    reply = model.invoke(text, **tts_options())
    wav = reply.additional_kwargs.get("audio")
    if not wav:
        raise RuntimeError(f"{TTS_MODEL} returned no audio")
    return Speech(wav, *_gemini_usage(reply, TTS_MODEL))


def _gemini_usage(reply: AIMessage, fallback_model: str) -> tuple[str, int, int]:
    tokens = reply.usage_metadata or {}
    return (
        _model_name(reply.response_metadata.get("model_name") or fallback_model),
        tokens.get("input_tokens", 0),
        tokens.get("output_tokens", 0),
    )


_override: Provider | None = None


def _provider() -> Provider:
    return _override or _default()


@contextmanager
def paid_chat_only(model: str = BENCHMARK_LLM_MODEL) -> Iterator[Provider]:
    """Answer every chat call in the block on the paid `model`, never the free one (the
    benchmark grid, so its rows aren't a mix of two models). Raises RuntimeError without
    OPENAI_API_KEY. Everything else stays on the current provider."""
    if not paid_chat_available():
        raise RuntimeError("OPENAI_API_KEY isn't set, so there is no paid chat model.")
    with use(replace(_provider(), chat_model=lambda: _openai_chat(model))) as provider:
        yield provider


def paid_chat_model(model: str = BENCHMARK_LLM_MODEL) -> str:
    """The name the usage ledger prices `paid_chat_only` calls by (for estimates)."""
    return model


@contextmanager
def use(provider: Provider) -> Iterator[Provider]:
    """Route every model call to `provider` until the block exits (e.g. fakes in tests)."""
    global _override
    previous, _override = _override, provider
    try:
        yield provider
    finally:
        _override = previous


@contextmanager
def without_reranker() -> Iterator[Provider]:
    """The current provider with no reranker until the block exits, so searches keep
    their technique's own order (e.g. the benchmark's un-reranked setups)."""
    with use(replace(_provider(), reranker=None)) as provider:
        yield provider


def text_generation_is_local() -> bool:
    """Whether the configured text-generation provider runs locally (Gemini doesn't)."""
    return _provider().local_text_generation


def chat_model(kind: str = "chat") -> BaseChatModel:
    """The chat model. `kind` is what its calls are for in the usage ledger: "chat" for
    an Agent's own reasoning, or e.g. "rewrite" or "pick" for a Retrieval Technique's
    calls (a call given an image is "vision" whatever the kind)."""
    return _recorded(_provider().chat_model(), kind)


def summaries_supported() -> bool:
    """Whether the current provider has a local summary model."""
    return _provider().summary_model is not None


def summary_model_hint() -> str:
    """One line for the user on what writes Summaries, for when they aren't being written,
    so no other module needs to know which local model or server that is."""
    if not summaries_supported():
        return (
            "Summaries need the local summary model, and this model provider has none "
            f"(Perry's default is {SUMMARY_MODEL} on Ollama)."
        )
    return (
        f"Summaries are written by the local model {summary_model_name()}: is Ollama "
        f"running at {OLLAMA_URL} with it (PERRY_SUMMARY_MODEL)?"
    )


def local_models_problem() -> str | None:
    """Why the local models (embeddings and Summaries, on Ollama) can't be reached, as one
    line for the Owner, or None when they can. Asked before serving Perry to others."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3):
            return None
    except (urllib.error.URLError, OSError):
        return (
            f"Ollama isn't reachable at {OLLAMA_URL} (PERRY_OLLAMA_URL). Start it with "
            "`ollama serve` (or open the Ollama app): Perry embeds every question there."
        )


def summary_model_name() -> str:
    """The name the usage ledger prices summary calls by."""
    return _provider().summary_model_name


def summary_model(kind: str = "summary") -> BaseChatModel:
    """The local text model for background work: Summaries, Document metadata and (later)
    RAPTOR. Slow (minutes per call on a CPU) but free; never use it in a Turn. Its
    calls are recorded in the usage ledger as `kind`, priced from the price table
    (zero for a local model)."""
    build = _provider().summary_model
    if build is None:
        raise SummariesUnavailable("This model provider has no local summary model.")
    return _recorded(build(), kind)


def _recorded(model: BaseChatModel, kind: str) -> BaseChatModel:
    """`model`, with its calls recorded in the usage ledger as `kind`."""
    callbacks = model.callbacks or []
    if not any(isinstance(handler, _UsageRecorder) for handler in callbacks):
        model.callbacks = [*callbacks, _UsageRecorder()]
    if kind == "chat":
        return model
    # Bound, not set on the model: a provider may hand every caller the same instance.
    return model.with_config(metadata={_USAGE_KIND: kind})


def embedding_model() -> str:
    """The name the usage ledger prices embedding calls by."""
    return _provider().embedding_model


def embed_documents(texts: list[str], kind: str = "embedding") -> list[list[float]]:
    """Each text's embedding, as a Document's. `kind` is what the call is for in the
    usage ledger, e.g. "pick" when semantic picking embeds a question."""
    _record_embedding(texts, kind)
    return _provider().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    _record_embedding([text])
    return _provider().embed_query(text)


def speech_supported() -> bool:
    """Whether the current provider can do push-to-talk (speech-to-text and text-to-speech)."""
    provider = _provider()
    return provider.transcribe is not None and provider.synthesize is not None


def transcribe(audio: bytes, mime_type: str) -> str:
    """The text spoken in `audio` (e.g. a WAV recording)."""
    stt = _provider().transcribe
    if stt is None:
        raise SpeechUnavailable("This model provider has no speech-to-text.")
    from perry import usage

    usage.check_turn_ceiling()
    result = stt(audio, mime_type)
    usage.record("speech_to_text", result.model, result.input_tokens, result.output_tokens)
    return result.text


def synthesize(text: str) -> bytes:
    """`text` read aloud, as WAV audio."""
    tts = _provider().synthesize
    if tts is None:
        raise SpeechUnavailable("This model provider has no text-to-speech.")
    from perry import usage

    usage.check_turn_ceiling()
    result = tts(text)
    usage.record("text_to_speech", result.model, result.input_tokens, result.output_tokens)
    return result.wav


def vision_supported() -> bool:
    """Whether the current provider can read text off images, e.g. scanned PDF pages."""
    return _provider().read_image is not None


def vision_model() -> str:
    """The name the usage ledger prices vision calls by."""
    return _provider().vision_model


def reranking_supported() -> bool:
    """Whether the current provider has a reranker."""
    return _provider().reranker is not None


def warm_up_reranker() -> threading.Thread | None:
    """Load the reranker in the background now, so the first question doesn't wait for
    it (tens of seconds on a CPU). None if there is no reranker. If loading fails, the
    first search reports it."""
    reranker = _provider().reranker
    if reranker is None:
        return None

    def load() -> None:
        try:
            reranker.score("warm up", ["warm up"])
        except Exception:  # noqa: BLE001, S110 - the first search reports it
            pass

    thread = threading.Thread(target=load, name="perry-reranker-warm-up", daemon=True)
    thread.start()
    return thread


def colbert_supported() -> bool:
    """Whether the current provider has a ColBERT encoder."""
    return _provider().colbert is not None


def colbert_model() -> str:
    """The name the usage ledger records ColBERT calls by."""
    colbert = _provider().colbert
    return colbert.model if colbert is not None else "unknown"


def colbert_encode_documents(texts: list[str]) -> list[np.ndarray]:
    """Each text's ColBERT token vectors (rows: tokens), for the ColBERT index."""
    _record_colbert(texts)
    return [np.asarray(v, np.float32) for v in _colbert().encode_documents(texts)]


def colbert_encode_query(text: str) -> np.ndarray:
    """The question's ColBERT token vectors, to search the ColBERT index with."""
    _record_colbert([text])
    return np.asarray(_colbert().encode_query(text), np.float32)


def _colbert() -> ColbertEncoder:
    colbert = _provider().colbert
    if colbert is None:
        raise RuntimeError("This model provider has no ColBERT encoder.")
    return colbert


def _record_colbert(texts: list[str]) -> None:
    # Local and free (priced 0 in prices.toml), but every model call is in the ledger.
    from perry import usage

    usage.check_turn_ceiling()
    usage.record("colbert", colbert_model(), estimate_tokens(texts))


def rerank(question: str, texts: list[str]) -> list[float]:
    """How well each of `texts` answers `question`, by the provider's reranker (higher
    is better). Raises RuntimeError if the provider has none."""
    reranker = _provider().reranker
    if reranker is None:
        raise RuntimeError("This model provider has no reranker.")
    from perry import usage

    usage.check_turn_ceiling()
    # Local and free, but every model call is in the ledger. The question is read once
    # with each text.
    tokens = estimate_tokens([f"{question} {text}" for text in texts])
    usage.record("rerank", reranker.model, tokens)
    return reranker.score(question, texts)


# What transcribing one scanned page is expected to cost, for estimates made before
# any call: a page rendered at 150 dpi plus the prompt in, a dense page of text out.
PAGE_TRANSCRIPTION_INPUT_TOKENS = 1600
PAGE_TRANSCRIPTION_OUTPUT_TOKENS = 800


def transcribe_page(image: bytes, mime_type: str) -> str:
    """The text on `image`, a rendered page of a Document (e.g. a scanned PDF page)."""
    read = _provider().read_image
    if read is None:
        raise VisionUnavailable("This model provider can't read images.")
    from perry import usage

    usage.check_turn_ceiling()
    result = read(image, mime_type)
    usage.record("transcription", result.model, result.input_tokens, result.output_tokens)
    return result.text


# --- Usage recording -----------------------------------------------------------


def estimate_tokens(texts: list[str]) -> int:
    """Embedding calls don't report their tokens here: about 4 characters per token."""
    return sum(math.ceil(len(text) / 4) for text in texts)


def _record_embedding(texts: list[str], kind: str = "embedding") -> None:
    from perry import usage

    usage.check_turn_ceiling()
    usage.record(kind, embedding_model(), estimate_tokens(texts))


def _model_name(name: str) -> str:
    return name.removeprefix("models/")


# Run metadata naming the usage ledger kind of a chat call, set by `chat_model(kind)`.
_USAGE_KIND = "perry_usage_kind"


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
        model = params.get("model") or params.get("model_name") or params.get("_type", "unknown")
        # A call given an image (e.g. one still in the Thread) pays for it again: "vision".
        kind = (kwargs.get("metadata") or {}).get(_USAGE_KIND, "chat")
        if any(images.has_images(m) for m in messages):
            kind = "vision"
        self._models[run_id] = (model, kind)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        from perry import usage

        fallback, kind = self._models.pop(run_id, ("unknown", "chat"))
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                tokens = (message.usage_metadata if message is not None else None) or {}
                metadata = message.response_metadata if message is not None else {}
                usage.record(
                    kind,
                    _model_name(metadata.get("model_name") or fallback),
                    tokens.get("input_tokens", 0),
                    tokens.get("output_tokens", 0),
                )

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self._models.pop(run_id, None)
