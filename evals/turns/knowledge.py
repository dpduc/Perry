"""Knowledge Agent Turns, answered from the Documents in `evals/documents/`."""

from evals.harness import EvalTurn

TURNS = [
    EvalTurn(
        id="knowledge-q3-budget",
        message="What's my Q3 marketing budget?",
        agents=("knowledge",),
        citations=("q3-budget.md",),
        facts=("The Q3 marketing budget is 40,000 dollars.",),
        technique="hybrid_search",
    ),
    EvalTurn(
        id="knowledge-trade-fair",
        message="How much of the Q3 budget goes to the trade fair, and where is it held?",
        agents=("knowledge",),
        citations=("q3-budget.md",),
        facts=("12,000 dollars.", "The trade fair is in Lyon."),
    ),
    EvalTurn(
        id="knowledge-budget-approval",
        message="Who approved the Q3 marketing budget, and when?",
        agents=("knowledge",),
        citations=("q3-budget.md",),
        facts=("Maria Okafor approved it.", "On July 2."),
    ),
    EvalTurn(
        id="knowledge-offsite-packing",
        message="Where is the team offsite, and what should I pack?",
        agents=("knowledge",),
        citations=("team-offsite.md",),
        facts=("Lakeview Lodge in Annecy.", "Hiking boots and a rain jacket."),
    ),
    EvalTurn(
        id="knowledge-tomato-harvest",
        message="When did I first harvest tomatoes this year, and what variety are they?",
        agents=("knowledge",),
        citations=("garden-log.md",),
        facts=("July 28.", "San Marzano."),
    ),
    EvalTurn(
        id="knowledge-aphids",
        message="How did I get rid of the aphids in the garden?",
        agents=("knowledge",),
        citations=("garden-log.md",),
        facts=("Sprayed neem oil twice, a week apart.",),
    ),
    EvalTurn(
        id="knowledge-book-rating",
        message="What did I rate The Design of Everyday Things?",
        agents=("knowledge",),
        citations=("reading-notes.txt",),
        facts=("4 out of 5.",),
    ),
    EvalTurn(
        id="knowledge-offsite-vs-fair",
        message="Which comes first, the Lyon trade fair or the team offsite?",
        agents=("knowledge",),
        citations=("q3-budget.md", "team-offsite.md"),
        facts=(
            "The trade fair is September 14-16 and the offsite October 21-22.",
            "So the trade fair comes first.",
        ),
        technique="question_decomposition",
    ),
    # Not in the Documents: Perry must say so instead of answering from general knowledge.
    EvalTurn(
        id="knowledge-missing-q4-budget",
        message="What's my Q4 marketing budget?",
        agents=("knowledge",),
        facts=(
            "Says the Documents don't contain a Q4 budget.",
            "Does not invent a figure (mentioning the Q3 budget is fine).",
        ),
    ),
    EvalTurn(
        id="knowledge-missing-dentist",
        message="When is my next dentist appointment, according to my notes?",
        agents=("knowledge",),
        facts=("Says the Documents don't mention a dentist appointment.",),
    ),
]
