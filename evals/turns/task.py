"""Task Agent Turns, against the Workspace fake in `evals/workspace.py`.

Write Action Turns must pause on a Confirmation for each listed Write Action; the
harness approves them, and they act only on the fake.
"""

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
    EvalTurn(
        id="task-write-email",
        message="Email sam@example.com that I'll be 10 minutes late to the roadmap review.",
        agents=("task",),
        write_actions=("send_email",),
        facts=("Says an email telling Sam about being 10 minutes late was sent.",),
    ),
    EvalTurn(
        id="task-write-reply",
        message="Reply to Priya's lunch email: yes, the noodle place sounds great.",
        agents=("task",),
        write_actions=("reply_email",),
        facts=("Says a reply accepting the noodle place was sent to Priya.",),
    ),
    EvalTurn(
        id="task-write-create",
        message="Add a dentist appointment to my calendar the day after tomorrow, 9:00 to 10:00.",
        agents=("task",),
        write_actions=("create_event",),
        facts=("Says a dentist event was created for the day after tomorrow, 9:00-10:00.",),
    ),
    EvalTurn(
        id="task-write-move",
        message="Move my lunch with Priya tomorrow to 13:00, same length.",
        agents=("task",),
        write_actions=("move_event",),
        facts=("Says the lunch with Priya now runs 13:00-14:00 tomorrow.",),
    ),
    EvalTurn(
        id="task-write-delete",
        message="Cancel my yoga session today, take it off the calendar.",
        agents=("task",),
        write_actions=("delete_event",),
        facts=("Says the Yoga event today was deleted.",),
    ),
    EvalTurn(
        id="task-write-chain",
        message="Find my notes on the Q3 marketing budget and email a short summary to "
        "sam@example.com.",
        agents=("knowledge", "task"),
        write_actions=("send_email",),
        facts=(
            "Says a summary was emailed to Sam.",
            "The summary mentions the 40,000 dollar Q3 marketing budget.",
        ),
    ),
]
