"""Tuning the cascade margin on the eval Turns, with the fake (bag-of-words) embeddings.

Two stand-in techniques with far-apart example questions make each question's nearest
technique plain; the tuned margin is then checked by cascade picking itself.
"""

import pytest

from evals.harness import EvalTurn
from evals.tuning import tune_cascade_margin
from perry import picking
from perry.techniques import Technique, hybrid

BUDGET = Technique(
    name="budget_search",
    label="Budget search",
    description="Search for money figures.",
    examples=(
        "How much is the marketing budget?",
        "Which budget pays for marketing?",
    ),
    chain=hybrid,
)
TRAILS = Technique(
    name="trail_search",
    label="Trail search",
    description="Search hiking notes.",
    examples=(
        "Is the ridge path steep?",
        "How long is the ridge hike?",
    ),
    chain=hybrid,
)
CANDIDATES = [BUDGET, TRAILS]


def turn(id: str, message: str, technique: str | None) -> EvalTurn:
    return EvalTurn(id=id, message=message, agents=("knowledge",), technique=technique)


CLEAR_BUDGET = turn("budget", "How much is the marketing budget for Q3?", "budget_search")
CLEAR_TRAIL = turn("trail", "Is the ridge path long?", "trail_search")
# Its words are nearer the budget examples, but it is about a hike: semantic picking
# gets it wrong, so the margin must leave it to logical picking.
MISLEADING = turn("misleading", "Which budget pays for the ridge hike?", "trail_search")
UNLABELLED = turn("unlabelled", "Hello there", None)


def cascade_pick(question: str, margin: float, monkeypatch) -> str | None:
    monkeypatch.setenv("PERRY_PICKING_MODE", "cascade")
    monkeypatch.setenv("PERRY_CASCADE_MARGIN", str(margin))
    picked = picking.pick(question, CANDIDATES)
    return picked.name if picked else None


def test_tuned_margin_leaves_wrong_semantic_picks_to_logical_picking(perry, monkeypatch):
    tuning = tune_cascade_margin([CLEAR_BUDGET, CLEAR_TRAIL, MISLEADING, UNLABELLED], CANDIDATES)

    assert cascade_pick(MISLEADING.message, tuning.margin, monkeypatch) is None
    assert cascade_pick(CLEAR_BUDGET.message, tuning.margin, monkeypatch) == "budget_search"
    assert cascade_pick(CLEAR_TRAIL.message, tuning.margin, monkeypatch) == "trail_search"
    assert (tuning.turns, tuning.semantic) == (3, 2)


def test_with_every_semantic_pick_right_the_margin_trusts_them_all(perry, monkeypatch):
    tuning = tune_cascade_margin([CLEAR_BUDGET, CLEAR_TRAIL], CANDIDATES)

    assert cascade_pick(CLEAR_BUDGET.message, tuning.margin, monkeypatch) == "budget_search"
    assert cascade_pick(CLEAR_TRAIL.message, tuning.margin, monkeypatch) == "trail_search"
    assert tuning.semantic == 2


def test_a_confidently_wrong_pick_leaves_every_question_to_logical_picking(
    perry, monkeypatch
):
    confidently_wrong = turn("wrong", "Which budget pays for marketing?", "trail_search")

    tuning = tune_cascade_margin([CLEAR_BUDGET, CLEAR_TRAIL, confidently_wrong], CANDIDATES)

    for question in [CLEAR_BUDGET, CLEAR_TRAIL, confidently_wrong]:
        assert cascade_pick(question.message, tuning.margin, monkeypatch) is None
    assert tuning.semantic == 0


def test_tuning_needs_turns_that_expect_a_technique(perry):
    with pytest.raises(ValueError, match="expect"):
        tune_cascade_margin([UNLABELLED], CANDIDATES)
