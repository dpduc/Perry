"""Analysis Agent Turns, answered from the Datasets in `evals/datasets/`.

`sales.csv` holds 2025 monthly ad spend, units and revenue for three regions. The
first Turns need only SQL; the later ones need Python in the Sandbox (statistics,
a forecast, charts), so running them needs Docker.
"""

from evals.harness import EvalTurn

TURNS = [
    # Read-only SQL
    EvalTurn(
        id="analysis-revenue-per-region",
        message="What was total 2025 revenue per region in my sales data?",
        agents=("analysis",),
        facts=("North 181,300.", "South 156,350.", "West 220,850.", "Shows the SQL it used."),
    ),
    EvalTurn(
        id="analysis-best-month",
        message="Which month had the highest total revenue across all regions, and how much?",
        agents=("analysis",),
        facts=("December 2025.", "51,700.", "Shows the SQL it used."),
    ),
    EvalTurn(
        id="analysis-q4-leader",
        message="Which region brought in the most revenue in Q4 2025?",
        agents=("analysis",),
        facts=("West, with 57,850.",),
    ),
    EvalTurn(
        id="analysis-average-units",
        message="On average, how many units did a region sell per month?",
        agents=("analysis",),
        facts=("About 310 units.",),
    ),
    EvalTurn(
        id="analysis-list-datasets",
        message="What data have I uploaded, and what columns does it have?",
        agents=("analysis",),
        facts=("A sales Dataset.", "Columns month, region, ad_spend, units and revenue."),
    ),
    # Python in the Sandbox
    EvalTurn(
        id="analysis-python-correlation",
        message="Is there a correlation between ad spend and revenue in my sales data?",
        agents=("analysis",),
        facts=(
            "The correlation is about -0.10.",
            "So there is essentially no (linear) relationship.",
        ),
    ),
    EvalTurn(
        id="analysis-python-forecast",
        message="Fit a linear trend to total monthly revenue and forecast January 2026.",
        agents=("analysis",),
        facts=(
            "Revenue grows by roughly 466 per month.",
            "January 2026 forecast is about 49,600.",
            "Shows the Python it used.",
        ),
    ),
    EvalTurn(
        id="analysis-python-volatility",
        message="Which region's monthly revenue is the most volatile? Use the standard deviation.",
        agents=("analysis",),
        facts=("South (standard deviation about 1,164).",),
    ),
    EvalTurn(
        id="analysis-python-chart",
        message="Plot monthly revenue per region as a line chart.",
        agents=("analysis",),
        facts=(
            "Includes a chart marker like [chart: <id>.png].",
            "Briefly describes the trend; West is highest throughout.",
        ),
    ),
]
