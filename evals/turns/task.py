"""Task Agent Turns, against the Workspace fake in `evals/workspace.py`."""

from evals.harness import EvalTurn

TURNS = [
    EvalTurn(
        id="task-tomorrow",
        message="What's on my calendar tomorrow?",
        agents=("task",),
        facts=("Roadmap review at 10:00 in Room 4B with Sam Patel.", "Lunch with Priya at 12:30."),
    ),
    EvalTurn(
        id="task-sam-budget",
        message="What did Sam say about the budget in his email?",
        agents=("task",),
        facts=("Sam wants to move 2,000 dollars from online ads to the Lyon trade fair stand.",),
    ),
]
