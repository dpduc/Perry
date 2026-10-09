"""The Retrieval Techniques the Knowledge Agent can pick from, all registered in TECHNIQUES.

Every Retrieval Technique is a `@chain` with one signature: a question in, Chunks
(`SearchHit`s, each with its Citation) out. Adding one is one chain plus one entry
in TECHNIQUES; the Knowledge Agent's tools, the sidebar dropdown and (later) the
picking modes and the benchmark all read the registry.

A technique that calls a chat model gets it from `models.chat_model(kind=...)`, with a
kind naming what the call is for (e.g. "rewrite"), so the Usage Ledger shows what each
technique costs and its calls count toward the Turn's Token Ceiling.
"""

from dataclasses import dataclass

from langchain_core.runnables import Runnable, chain

from perry.search import SearchHit, hybrid_search, summary_search


@dataclass(frozen=True)
class Technique:
    """One Retrieval Technique.

    `name` is its id: the Knowledge Agent's tool name and the per-request override's
    value. `description` says when to use it; it is the tool description the Knowledge
    Agent's model picks by. `examples` are hand-written questions it suits best, for
    picking by similarity; never copy them from the benchmark.
    """

    name: str
    label: str
    description: str
    examples: tuple[str, ...]
    chain: Runnable[str, list[SearchHit]]


@chain
def hybrid(question: str) -> list[SearchHit]:
    """Today's baseline: one hybrid search for the question as asked."""
    return hybrid_search(question)


@chain
def document_summaries(question: str) -> list[SearchHit]:
    """Multi-representation indexing: match the question against Document Summaries
    (only Documents whose Summaries are ready), then return those Documents' best
    Chunks. A Summary is never returned, so never cited."""
    return summary_search(question)


TECHNIQUES: tuple[Technique, ...] = (
    Technique(
        name="hybrid_search",
        label="Hybrid search",
        description=(
            "Search the user's Documents for the question as asked (meaning plus exact "
            "words). Best for plain lookups of one fact, figure, date or definition."
        ),
        examples=(
            "What is the deadline for the grant application?",
            "How much did we pay for the venue?",
            "Who is the contact person for the supplier?",
            "What does the contract say about the notice period?",
            "When is the next team offsite?",
        ),
        chain=hybrid,
    ),
    Technique(
        name="document_summaries",
        label="Document Summaries",
        description=(
            "Find the Documents whose Summaries best match the question, then return "
            "their best-matching passages. Best for finding which Document covers a "
            "subject, or questions about what a Document as a whole is about."
        ),
        examples=(
            "Which document talks about the office move?",
            "Do I have any file about the insurance renewal?",
            "Where did I write down the onboarding process?",
            "Which of my notes covers the pricing discussion?",
            "What is the vendor evaluation report about?",
        ),
        chain=document_summaries,
    ),
)


def get(name: str) -> Technique:
    """The registered technique called `name`."""
    for technique in TECHNIQUES:
        if technique.name == name:
            return technique
    raise KeyError(f"No Retrieval Technique {name!r}; known: {[t.name for t in TECHNIQUES]}")
