"""Model provider seam: the only module that knows which model API is used.

Switching to a local model (see the design decisions in GLOSSARY.md / docs/adr)
is a change to this file only. Tests swap the whole provider with `use()`.

Every call made through this module is recorded in the usage ledger (`perry.usage`),
whichever provider is in use. Inside a Turn, a chat or embedding call raises
`usage.TurnTokenCeilingReached` once the Turn has used up its tokens.
"""

import base64
import math
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import GoogleRateLimitError

from perry import images

# The provider's "quota exhausted" error, so callers can report it without knowing the provider.
RateLimitError = GoogleRateLimitError

LLM_MODEL = os.getenv("PERRY_LLM_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = os.getenv("PERRY_EMBEDDING_MODEL", "gemini-embedding-001")
# Text-to-speech: a Gemini model whose name ends in "-tts" answers with audio.
TTS_MODEL = os.getenv("PERRY_TTS_MODEL", "gemini-2.5-flash-preview-tts")
TTS_VOICE = os.getenv("PERRY_TTS_VOICE", "Kore")

# pgvector's HNSW index supports at most 2000 dimensions.
# Changing the model or this number means re-ingesting every Document.
EMBEDDING_DIMENSIONS = 1536


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


class SpeechUnavailable(Exception):
    """The current model provider can't transcribe or synthesise speech."""


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

    def chat_model() -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL, temperature=0, rate_limiter=rate_limiter, max_retries=3
        )

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
            texts, task_type="RETRIEVAL_DOCUMENT"
        ),
        embed_query=lambda text: embeddings.embed_query(text, task_type="RETRIEVAL_QUERY"),
        embedding_model=EMBEDDING_MODEL,
        transcribe=lambda audio, mime_type: _gemini_transcribe(chat_model(), audio, mime_type),
        synthesize=lambda text: _gemini_synthesize(tts, text),
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


def _gemini_synthesize(model: ChatGoogleGenerativeAI, text: str) -> Speech:
    # A "-tts" model answers with WAV bytes (24 kHz mono) in additional_kwargs["audio"].
    reply = model.invoke(text, voice_name=TTS_VOICE)
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


def embed_documents(texts: list[str]) -> list[list[float]]:
    _record_embedding(texts)
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


# --- Usage recording -----------------------------------------------------------


def estimate_tokens(texts: list[str]) -> int:
    """Gemini's Developer API doesn't report embedding usage: about 4 characters per token."""
    return sum(math.ceil(len(text) / 4) for text in texts)


def _record_embedding(texts: list[str]) -> None:
    from perry import usage

    usage.check_turn_ceiling()
    usage.record("embedding", _provider().embedding_model, estimate_tokens(texts))


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
        model = params.get("model") or params.get("model_name") or params.get("_type", "unknown")
        # A call given an image (e.g. one still in the Thread) pays for it again: "vision".
        kind = "vision" if any(images.has_images(m) for m in messages) else "chat"
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
