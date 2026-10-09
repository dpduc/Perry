"""The Retrieval Techniques the Knowledge Agent can pick from, all registered in TECHNIQUES.

Every Retrieval Technique is a `@chain` with one signature: a question in, Chunks
(`SearchHit`s, each with its Citation) out. Adding one is one chain plus one entry
in TECHNIQUES; the Knowledge Agent's tools, the sidebar dropdown, the picking
modes (`perry.picking`) and (later) the benchmark all read the registry.

A technique that calls a chat model gets it from `models.chat_model(kind=...)`, with a
kind naming what the call is for (e.g. "rewrite"), so the Usage Ledger shows what each
technique costs and its calls count toward the Turn's Token Ceiling.
"""

from dataclasses import dataclass

from langchain_core.runnables import Runnable, chain

from perry.search import SearchHit, hybrid_search


@dataclass(frozen=True)
class Technique:
    """One Retrieval Technique.

    `name` is its id: the Knowledge Agent's tool name and the per-request override's
    value. `description` says when to use it; it is the tool description the Knowledge
    Agent's model picks by. `examples` are hand-written questions it suits best, for
    picking by similarity; never copy them from the benchmark (a test checks).
    `needs_summaries` marks a technique that searches Summaries: it isn't picked while
    Summaries are pending.
    """

    name: str
    label: str
    description: str
    examples: tuple[str, ...]
    chain: Runnable[str, list[SearchHit]]
    needs_summaries: bool = False


@chain
def hybrid(question: str) -> list[SearchHit]:
    """Today's baseline: one hybrid search for the question as asked."""
    return hybrid_search(question)


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
)


def summaries_pending() -> bool:
    """Whether any Document's Summaries are still being built.

    Always False until background summarising (#21) exists; it replaces this check
    with the Documents' summarising status.
    """
    return False


def available() -> list[Technique]:
    """The registered techniques that can be picked right now: all of them, except
    those that search Summaries while any Document's Summaries are pending. (A forced
    technique runs regardless.)"""
    pending = summaries_pending()
    return [t for t in TECHNIQUES if not (t.needs_summaries and pending)]


def get(name: str) -> Technique:
    """The registered technique called `name`."""
    for technique in TECHNIQUES:
        if technique.name == name:
            return technique
    raise KeyError(f"No Retrieval Technique {name!r}; known: {[t.name for t in TECHNIQUES]}")
