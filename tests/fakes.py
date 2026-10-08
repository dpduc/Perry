"""Fake models for tests: a scripted chat model and deterministic embeddings. No network."""

import hashlib
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from perry.models import EMBEDDING_DIMENSIONS, Provider


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
        if callable(step) and not isinstance(step, (BaseMessage, ToolCall)):
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


def fake_provider(chat: ScriptedChatModel) -> Provider:
    return Provider(chat_model=lambda: chat, embed_documents=fake_embed, embed_query=fake_embed_one)
