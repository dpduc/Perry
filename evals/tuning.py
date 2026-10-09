"""Tuning the cascade margin (`PERRY_CASCADE_MARGIN`) on the eval Turns, never the benchmark.

Cascade picking trusts semantic picking when the nearest technique's example questions
beat the runner-up's by the margin, and leaves the question to logical picking (the
Knowledge Agent's model) otherwise. Semantic scores need only embeddings, so tuning
costs no chat calls.

For every eval Turn that expects a Retrieval Technique, the tuning measures the gap
between the nearest technique and the runner-up, and whether the nearest is the one
expected. The tuned margin is the smallest one above every wrong pick's gap: every
semantic pick it lets through was right on the eval Turns, and as many questions as
possible skip the extra model call. Questions it leaves to logical picking are assumed
picked right (logical picking needs chat calls to measure; the eval run's soft
technique check does that).
"""

from dataclasses import dataclass

from evals.harness import EvalTurn
from perry import picking
from perry.techniques import Technique

# How far above the largest wrong gap the margin sits when no right pick lies above it.
HEADROOM = 0.01


@dataclass(frozen=True)
class Scored:
    """One eval Turn under semantic picking."""

    turn: EvalTurn
    nearest: str  # the technique semantic picking would choose
    gap: float  # how much nearer it is than the runner-up

    @property
    def right(self) -> bool:
        return self.nearest == self.turn.technique


@dataclass(frozen=True)
class Tuning:
    margin: float
    scored: list[Scored]

    @property
    def turns(self) -> int:
        return len(self.scored)

    @property
    def semantic(self) -> int:
        """How many eval Turns cascade picking settles semantically at this margin."""
        return sum(s.gap >= self.margin for s in self.scored)


def tune_cascade_margin(turns: list[EvalTurn], candidates: list[Technique]) -> Tuning:
    """The cascade margin tuned on the `turns` that expect a technique."""
    scored = [score(t, candidates) for t in turns if t.technique]
    if not scored:
        raise ValueError("No eval Turns expect a Retrieval Technique to tune on")
    wrong = [s.gap for s in scored if not s.right]
    if not wrong:
        return Tuning(0.0, scored)
    worst = max(wrong)
    above = [s.gap for s in scored if s.gap > worst]
    # Midway between the worst wrong gap and the next right one generalises better
    # than sitting right on top of the wrong one.
    margin = (worst + min(above)) / 2 if above else worst + HEADROOM
    return Tuning(margin, scored)


def score(turn: EvalTurn, candidates: list[Technique]) -> Scored:
    ranked = sorted(picking.nearness(turn.message, candidates).items(), key=lambda i: -i[1])
    (nearest, top), (_, runner_up) = ranked[0], ranked[1]
    return Scored(turn, nearest, top - runner_up)


def report(tuning: Tuning) -> str:
    lines = [f"{'gap':>7}  {'nearest':<24}{'expected':<24}turn"]
    for s in sorted(tuning.scored, key=lambda s: -s.gap):
        mark = "" if s.right else "  WRONG"
        lines.append(f"{s.gap:7.4f}  {s.nearest:<24}{s.turn.technique:<24}{s.turn.id}{mark}")
    right = sum(s.right for s in tuning.scored)
    lines.append(
        f"\nSemantic picking alone: {right}/{tuning.turns} right."
        f"\nTuned cascade margin: {tuning.margin:.4f} "
        f"({tuning.semantic}/{tuning.turns} picked semantically, all right; "
        "the rest left to logical picking)."
    )
    return "\n".join(lines)
