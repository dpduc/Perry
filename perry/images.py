"""Images as model input: LangChain's provider-neutral image content blocks.

`perry.models` turns these into whatever the model API wants, so nothing here is
specific to a provider.
"""

import base64
import mimetypes
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage


def image_block(data: bytes, mime_type: str) -> dict:
    return {"type": "image", "base64": base64.b64encode(data).decode("ascii"), "mime_type": mime_type}


def file_block(path: str | Path) -> dict:
    """An image file as a content block, typed by its extension."""
    path = Path(path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"{path.name} is not an image")
    return image_block(path.read_bytes(), mime_type)


def user_message(text: str, images: list[str | Path] = ()) -> HumanMessage:
    """The user's message, with any attached images as multimodal content."""
    if not images:
        return HumanMessage(text)
    return HumanMessage([{"type": "text", "text": text}, *(file_block(i) for i in images)])


def has_images(messages: list[BaseMessage]) -> bool:
    return any(
        isinstance(block, dict) and block.get("type") in ("image", "image_url")
        for message in messages
        if isinstance(message.content, list)
        for block in message.content
    )
