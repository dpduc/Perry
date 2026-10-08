"""Eval Turns as data, one module per Agent.

To add eval Turns, create `evals/turns/<agent>.py` with a module-level
`TURNS: list[EvalTurn]`; `all_turns()` picks it up. Documents the Turns rely on go
in `evals/documents/`; calendar and email in `evals/workspace.py`. Other fixtures
(e.g. Datasets) are set up in `evals/__main__.py`'s `environment()`.
"""

import importlib
import pkgutil
from pathlib import Path

from evals.harness import EvalTurn

DOCUMENTS_DIR = Path(__file__).parent.parent / "documents"


def all_turns() -> list[EvalTurn]:
    turns = []
    for module in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        turns.extend(importlib.import_module(f"{__name__}.{module.name}").TURNS)
    return turns
