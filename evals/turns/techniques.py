"""Knowledge Agent Turns that each expect one Retrieval Technique (a soft check).

Two or more per technique, answered from `evals/documents/`. They are also the data the
cascade margin is tuned on (`python -m evals --tune-margin`), so they must never copy a
benchmark question or a technique's example question (a test checks).
"""

from evals.harness import EvalTurn


def _turn(id, message, technique, citations, facts) -> EvalTurn:
    return EvalTurn(
        id=f"technique-{id}",
        message=message,
        agents=("knowledge",),
        citations=citations,
        facts=facts,
        technique=technique,
    )


TURNS = [
    # Hybrid search: plain lookups of one fact.
    _turn(
        "hybrid-meeting-time",
        "What time do we meet at the lodge on the first day of the offsite?",
        "hybrid_search",
        ("team-offsite.md",),
        ("At 10:00.",),
    ),
    _turn(
        "hybrid-contractors",
        "How much of the Q3 budget is set aside for contractors?",
        "hybrid_search",
        ("q3-budget.md",),
        ("10,000 dollars, for a copywriter and a designer.",),
    ),
    # Question Decomposition: comparing or chaining facts.
    _turn(
        "decomposition-ads-vs-fair",
        "Did online ads or the trade fair get more of the Q3 budget, and by how much?",
        "question_decomposition",
        ("q3-budget.md",),
        ("Online ads got 18,000 dollars, the trade fair 12,000.", "6,000 dollars more."),
    ),
    _turn(
        "decomposition-aphids-vs-basil",
        "Did the aphids show up before or after I sowed the basil?",
        "question_decomposition",
        ("garden-log.md",),
        ("Basil was sown on May 20 and the aphids came on June 3, so after.",),
    ),
    # Multi-query: vague, loosely-worded questions.
    _turn(
        "multi-query-plant-trouble",
        "Anything about plants having a rough time this year?",
        "multi_query",
        ("garden-log.md",),
        ("Aphids on the tomatoes in June.", "The basil bolted in the August heat."),
    ),
    _turn(
        "multi-query-trip-stuff",
        "Is there something about getting to the lake trip?",
        "multi_query",
        ("team-offsite.md",),
        ("Trains from Lyon every two hours; meet at the lodge at 10:00.",),
    ),
    # RAG fusion: broad questions where the most relevant passages should come first.
    _turn(
        "rag-fusion-marketing-spend",
        "How is the marketing money being spent this quarter?",
        "rag_fusion",
        ("q3-budget.md",),
        ("Online ads 18,000, trade fair 12,000, contractors 10,000 dollars.",),
    ),
    _turn(
        "rag-fusion-gardening-lessons",
        "What have I learned from gardening so far?",
        "rag_fusion",
        ("garden-log.md",),
        ("Neem oil cleared the aphids.", "Sow basil in partial shade next year."),
    ),
    # Step-back: "why" questions and the reasons behind something.
    _turn(
        "step-back-basil",
        "Why did my basil bolt, and what should I change next year?",
        "step_back",
        ("garden-log.md",),
        ("The August heat.", "Sow it in partial shade."),
    ),
    _turn(
        "step-back-blame-design",
        "Why should a device's design be blamed before its user when people make errors?",
        "step_back",
        ("reading-notes.txt",),
        ("From The Design of Everyday Things: blame the design before the user.",),
    ),
    # HyDE: casual wording far from how the Documents phrase the answer.
    _turn(
        "hyde-leftover-money",
        "If we don't spend all the ad cash, do we get to keep it?",
        "hyde",
        ("q3-budget.md",),
        ("No: leftover money goes back to the general fund, not to Q4.",),
    ),
    _turn(
        "hyde-walking-shoes",
        "Do I need proper shoes for the work trip?",
        "hyde",
        ("team-offsite.md",),
        ("Yes: hiking boots (and a rain jacket) for the guided hike.",),
    ),
    # Document Summaries: which Document covers a subject.
    _turn(
        "summaries-garden-file",
        "Which of my files is about the vegetable patch?",
        "document_summaries",
        ("garden-log.md",),
        ("The garden log (garden-log.md).",),
    ),
    _turn(
        "summaries-book-file",
        "Do I have a document where I wrote about books?",
        "document_summaries",
        ("reading-notes.txt",),
        ("Yes, the reading notes (reading-notes.txt).",),
    ),
    # RAPTOR: thematic, big-picture questions across Documents.
    _turn(
        "raptor-work-themes",
        "Overall, what are the big work events and costs across my files?",
        "raptor",
        ("q3-budget.md", "team-offsite.md"),
        ("The Lyon trade fair in the Q3 budget.", "The team offsite in Annecy."),
    ),
    _turn(
        "raptor-hobbies",
        "Taken together, what do my personal notes say about how I spend my free time?",
        "raptor",
        ("garden-log.md", "reading-notes.txt"),
        ("Gardening (tomatoes, basil).", "Reading (Don Norman, Donella Meadows)."),
    ),
    # Self-query: questions limited to certain Documents.
    _turn(
        "self-query-text-files",
        "In my .txt files, which book is next on the list?",
        "self_query",
        ("reading-notes.txt",),
        ("Thinking in Systems, by Donella Meadows.",),
    ),
    _turn(
        "self-query-offsite-doc",
        "According to the team offsite document, how often do trains leave Lyon?",
        "self_query",
        ("team-offsite.md",),
        ("Every two hours.",),
    ),
    # ColBERT: questions that hinge on an exact name or rare term.
    _turn(
        "colbert-lakeview",
        "What is said about Lakeview Lodge?",
        "colbert",
        ("team-offsite.md",),
        ("The offsite is there, October 21-22, in Annecy.",),
    ),
    _turn(
        "colbert-san-marzano",
        "Where are the San Marzano seedlings planted?",
        "colbert",
        ("garden-log.md",),
        ("Along the south fence.",),
    ),
]
