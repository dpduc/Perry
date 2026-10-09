"""Picking a Retrieval Technique for a question, before the Knowledge Agent searches.

Three picking modes, set by PERRY_PICKING_MODE:

- **semantic**: the question's embedding is compared with every technique's hand-written
  example questions, and the technique with the nearest example runs. No chat call.
- **logical**: the Knowledge Agent's model chooses from the techniques' descriptions.
- **cascade** (the default): semantic when the nearest technique beats the runner-up by
  PERRY_CASCADE_MARGIN, else logical.

The per-request override (a forced technique) wins over every mode; the Knowledge Agent
applies it before asking here.
"""

import math
import os

from perry import models
from perry.techniques import Technique

SEMANTIC = "semantic"
LOGICAL = "logical"
CASCADE = "cascade"
MODES = (SEMANTIC, LOGICAL, CASCADE)

# How much nearer (cosine similarity) the closest technique's examples must be than the
# runner-up's for cascade picking to trust them. Tuned on the eval set, never the benchmark:
# `uv run python -m evals --tune-margin` (2026-10-09, nomic-embed-text).
DEFAULT_CASCADE_MARGIN = 0.116


def mode(override: str | None = None) -> str:
    """The picking mode in force: `override` if given, else PERRY_PICKING_MODE (default
    cascade). Raises ValueError for a mode that isn't one of MODES."""
    setting = (override or os.getenv("PERRY_PICKING_MODE", CASCADE)).strip().lower()
    if setting not in MODES:
        raise ValueError(f"The picking mode must be one of {MODES}, not {setting!r}")
    return setting


def cascade_margin() -> float:
    return float(os.getenv("PERRY_CASCADE_MARGIN", str(DEFAULT_CASCADE_MARGIN)))


def pick(
    question: str, candidates: list[Technique], picking_mode: str | None = None
) -> Technique | None:
    """The technique semantic picking settles on for `question`, or None to leave the
    choice to the Knowledge Agent's model (logical picking). `picking_mode` overrides
    the setting (the benchmark compares the modes)."""
    picking = mode(picking_mode)
    if picking == LOGICAL or not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    ranked = sorted(nearness(question, candidates).items(), key=lambda item: -item[1])
    (top, top_score), (_, runner_up) = ranked[0], ranked[1]
    if picking == SEMANTIC or top_score - runner_up >= cascade_margin():
        return next(t for t in candidates if t.name == top)
    return None


def nearness(question: str, candidates: list[Technique]) -> dict[str, float]:
    """Each technique's nearness to `question`: the cosine similarity of its nearest
    example question."""
    examples = [example for t in candidates for example in t.examples]
    vectors = dict(zip(examples, _embed_examples(examples), strict=True))
    asked = models.embed_documents([question], kind="pick")[0]
    return {
        t.name: max((_cosine(asked, vectors[example]) for example in t.examples), default=-1.0)
        for t in candidates
    }


# Example questions only change with the code, so each is embedded once per embedding model.
_example_vectors: dict[tuple[str, str], list[float]] = {}


def _embed_examples(examples: list[str]) -> list[list[float]]:
    model = models.embedding_model()
    missing = [e for e in dict.fromkeys(examples) if (model, e) not in _example_vectors]
    if missing:
        # Embedded like the question (as documents), so the two are compared like for like.
        vectors = models.embed_documents(missing, kind="pick")
        for example, vector in zip(missing, vectors, strict=True):
            _example_vectors[(model, example)] = vector
    return [_example_vectors[(model, e)] for e in examples]


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0
