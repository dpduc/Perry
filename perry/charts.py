"""Charts made in the Sandbox, kept as PNG files and referenced from answers by a marker.

A marker like `[chart: <id>.png]` travels in the answer text through the Agents and the
Supervisor, and is stored in the Thread with it. The chat UI turns markers into images.
Code, not the model, makes sure every chart made during a Turn stays in its answer.
"""

import re
import uuid
from pathlib import Path

from perry.paths import app_data_dir

_MARKER = re.compile(r"\[chart: ([0-9a-f]{32}\.png)\]")


def directory() -> Path:
    return app_data_dir() / "charts"


def save(png: bytes) -> str:
    """Keep the chart and return its marker."""
    folder = directory()
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}.png"
    (folder / name).write_bytes(png)
    return f"[chart: {name}]"


def markers(text: str) -> list[str]:
    return [match.group(0) for match in _MARKER.finditer(text)]


def keep(answer: str, sources: list[str]) -> str:
    """`answer` with every chart marker found in `sources` that it left out appended."""
    missing = []
    for marker in (m for source in sources for m in markers(source)):
        if marker not in answer and marker not in missing:
            missing.append(marker)
    return "\n\n".join([answer.rstrip(), *missing]) if missing else answer


def chat_messages(answer: str) -> list[dict]:
    """The answer as chat messages: one image per chart, then the text without markers."""
    paths = [directory() / match.group(1) for match in _MARKER.finditer(answer)]
    # A chart file deleted since is left out rather than breaking the chat.
    images = [{"role": "assistant", "content": {"path": str(p)}} for p in paths if p.exists()]
    text = re.sub(r"\n{3,}", "\n\n", _MARKER.sub("", answer)).strip()
    return [*images, {"role": "assistant", "content": text}] if text else images
