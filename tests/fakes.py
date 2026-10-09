"""Fake models for tests: a scripted chat model and deterministic embeddings. No network."""

import hashlib
import json
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from perry.models import EMBEDDING_DIMENSIONS, Provider, Speech, Transcription


def fake_embed_one(text: str) -> list[float]:
    """Deterministic bag-of-words embedding: texts sharing words point the same way."""
    vector = [0.0] * EMBEDDING_DIMENSIONS
    for word in re.findall(r"\w+", text.lower()):
        index = int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBEDDING_DIMENSIONS
        vector[index] += 1.0
    return vector


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [fake_embed_one(t) for t in texts]


@dataclass(frozen=True)
class ToolCall:
    """A scripted reply that calls a tool (e.g. an Agent) instead of answering."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)


def call(tool: str, **args: Any) -> ToolCall:
    return ToolCall(tool, args)


# One scripted reply. A callable receives the messages the model was given and
# returns one of the others, so a reply can depend on what a tool returned.
Step = str | ToolCall | AIMessage | Exception | Callable[[list[BaseMessage]], Any]


def last_message(messages: list[BaseMessage]) -> str:
    """A Step that answers with the latest message's text, e.g. echoing a tool result."""
    return messages[-1].text


class ScriptedChatModel(BaseChatModel):
    """Replies with the scripted Steps in order, shared by the Supervisor and every Agent.

    Running out of Steps fails the test: the Turn made a model call nobody expected.
    """

    steps: list[Any] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs) -> "ScriptedChatModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        if not self.steps:
            raise AssertionError(
                f"Unexpected model call #{len(self.calls)}: the script has no replies left. "
                f"Last message: {messages[-1].text!r}"
            )
        step = self.steps.pop(0)
        if _is_callable_step(step):
            step = step(messages)
        return ChatResult(generations=[ChatGeneration(message=_reply(step))])


def _reply(step: Step) -> AIMessage:
    if isinstance(step, Exception):
        raise step
    if isinstance(step, str):
        return AIMessage(step)
    if isinstance(step, ToolCall):
        # Fresh ids every time, so the same ToolCall can be scripted repeatedly.
        tool_call = {"name": step.name, "args": step.args, "id": f"call_{uuid.uuid4().hex}"}
        return AIMessage("", tool_calls=[tool_call])
    if isinstance(step, AIMessage):
        return step.model_copy()
    raise TypeError(f"Not a scripted reply: {step!r}")


def with_usage(step: Step, input_tokens: int, output_tokens: int = 0) -> Callable:
    """A Step that replies like `step` and reports this many tokens used."""

    def reply(messages: list[BaseMessage]) -> AIMessage:
        inner = step(messages) if _is_callable_step(step) else step
        message = _reply(inner)
        message.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        return message

    return reply


def _is_callable_step(step: Any) -> bool:
    return callable(step) and not isinstance(step, (BaseMessage, ToolCall))


_SPOKEN = b"fake-speech:"


def spoken(text: str) -> bytes:
    """Fake audio of someone saying `text`: what the fake speech models hear and produce."""
    return _SPOKEN + text.encode()


def fake_transcribe(audio: bytes, mime_type: str) -> Transcription:
    assert audio.startswith(_SPOKEN), f"Not fake speech: {audio[:20]!r}"
    return Transcription(audio.removeprefix(_SPOKEN).decode(), "fake-stt", 40, 10)


def fake_synthesize(text: str) -> Speech:
    return Speech(spoken(text), "fake-tts", 10, 200)


class FakeVision:
    """Reads images (e.g. scanned PDF pages) with scripted replies, in call order.

    A reply that is an Exception is raised instead. Running out of replies fails the
    test: something was sent for transcription that nobody expected.
    """

    model = "fake-vision"

    def __init__(self, *replies: str | Exception):
        self.replies = list(replies)
        self.images: list[tuple[bytes, str]] = []

    def __call__(self, image: bytes, mime_type: str) -> Transcription:
        self.images.append((image, mime_type))
        if not self.replies:
            raise AssertionError(f"Unexpected transcription #{len(self.images)}")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Transcription(reply, self.model, 300, 50)


class FakeSummaryModel(BaseChatModel):
    """The local summary model: replies with the scripted replies in order (an Exception
    is raised instead), then with a deterministic default.

    The default reply is the JSON a Summary call asks for: the Summary is "Summary: "
    plus the text it was given, so a Summary shares its Document's words (and so its
    fake embedding); no Document date, topic "general". It names its model as the real
    local one does, so the ledger prices it from the price table.
    """

    replies: list[Any] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    model: str = "qwen3:8b"

    @property
    def _llm_type(self) -> str:
        return "fake-summary"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        if self.replies:
            reply = self.replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
        else:
            reply = json.dumps(
                {"summary": f"Summary: {messages[-1].text}", "document_date": None,
                 "topic": "general"}
            )
        message = AIMessage(
            reply,
            response_metadata={"model_name": self.model},
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def fake_provider(
    chat: ScriptedChatModel,
    *,
    speech: bool = True,
    local: bool = False,
    vision: FakeVision | None = None,
    summary: FakeSummaryModel | None = None,
) -> Provider:
    """Fake models; `local=True` makes text generation count as local (for privacy checks).
    Only with a `summary` model can the provider write Summaries."""
    vision = vision or FakeVision()
    return Provider(
        chat_model=lambda: chat,
        embed_documents=fake_embed,
        embed_query=fake_embed_one,
        embedding_model="fake-embedding",
        transcribe=fake_transcribe if speech else None,
        synthesize=fake_synthesize if speech else None,
        read_image=vision,
        vision_model=vision.model,
        local_text_generation=local,
        summary_model=(lambda: summary) if summary else None,
        summary_model_name=summary.model if summary else "unknown",
    )
