"""Score a benchmark answer for correctness (expected facts) and Citation accuracy (pages).

Scoring is deterministic string matching, so results are reproducible and cost no
model calls. Each fact lists its acceptable phrasings; an answer is correct when
every fact appears in some phrasing.
"""

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

QUESTION_TYPES = ("lookup", "multi_hop", "thematic", "entity")

# "report.pdf#page=4", as both Perry and QA-bot cite PDF pages.
CITATION = re.compile(r"[\w\-.]+\.pdf#page=\d+", re.IGNORECASE)


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    question: str
    # Each fact is a tuple of acceptable phrasings.
    facts: tuple[tuple[str, ...], ...]
    # Pages the answer must cite (the facts' sources).
    pages: tuple[str, ...]
    # Pages that are also correct to cite, but not required.
    related_pages: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class Score:
    fact_recall: float
    citation_recall: float
    citation_precision: float
    cited: tuple[str, ...]

    @property
    def correct(self) -> bool:
        return self.fact_recall == 1.0


def load_questions(path: Path) -> list[Question]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return [
        Question(
            id=q["id"],
            type=q["type"],
            question=q["question"],
            facts=tuple(tuple(f) if isinstance(f, list) else (f,) for f in q["facts"]),
            pages=tuple(q["pages"]),
            related_pages=tuple(q.get("related_pages", ())),
        )
        for q in data["question"]
    ]


def _normalise(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text.lower())
    return " ".join(text.split())


def extract_citations(answer: str) -> tuple[str, ...]:
    return tuple(sorted(set(CITATION.findall(answer))))


def score(question: Question, answer: str) -> Score:
    text = _normalise(answer)
    found = sum(
        any(_normalise(phrasing) in text for phrasing in fact) for fact in question.facts
    )
    cited = extract_citations(answer)
    relevant = set(question.pages) | set(question.related_pages)
    right = [c for c in cited if c in relevant]
    return Score(
        fact_recall=found / len(question.facts),
        citation_recall=len(set(question.pages) & set(cited)) / len(question.pages),
        citation_precision=len(right) / len(cited) if cited else 0.0,
        cited=cited,
    )
