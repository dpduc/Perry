"""Turns the Supervisor answers itself, without consulting any Agent."""

from evals.harness import EvalTurn

TURNS = [
    EvalTurn(
        id="no-agent-greeting",
        message="Hi Perry! What can you help me with?",
        agents=(),
        facts=("A short, friendly answer that mentions answering from the user's Documents.",),
    ),
]
