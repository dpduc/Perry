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

from perry import colbert as colbert_index
from perry.search import SearchHit, hybrid_search


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
def colbert(question: str) -> list[SearchHit]:
    """Late interaction: every word of the question matched against every word of each
    Chunk, over ColBERT's own index (docs/adr/0002)."""
    return colbert_index.search(question)


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
        name="colbert",
        label="ColBERT",
        description=(
            "Search the user's Documents matching the question word by word against "
            "every passage. Best when the question hinges on an exact name, code, "
            "product or rare term that must appear in the passage."
        ),
        examples=(
            "What does the Orion-4 maintenance log say about the hydraulic pump?",
            "Which clause mentions the Fennick Holdings guarantee?",
            "Where is invoice INV-20931 referenced?",
            "What did Dr. Okonkwo-Lindqvist recommend in her review?",
            "Which document mentions the term thermocline drift?",
        ),
        chain=colbert,
    ),
)


def get(name: str) -> Technique:
    """The registered technique called `name`."""
    for technique in TECHNIQUES:
        if technique.name == name:
            return technique
    raise KeyError(f"No Retrieval Technique {name!r}; known: {[t.name for t in TECHNIQUES]}")
