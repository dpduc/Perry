"""The Retrieval Techniques the Knowledge Agent can pick from, all registered in TECHNIQUES.

Every Retrieval Technique is a `@chain` with one signature: a question in, Chunks
(`SearchHit`s, each with its Citation) out. Adding one is one chain plus one entry
in TECHNIQUES; the Knowledge Agent's tools, the sidebar dropdown and (later) the
picking modes and the benchmark all read the registry.

A technique that calls a chat model gets it from `models.chat_model(kind=...)`, with a
kind naming what the call is for (e.g. "rewrite"), so the Usage Ledger shows what each
technique costs and its calls count toward the Turn's Token Ceiling.
"""

from dataclasses import dataclass, replace

from langchain_core.runnables import Runnable, chain

from perry import models
from perry.search import RRF_K, SearchHit, hybrid_search


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


REWRITE_PROMPT = """Rewrite the question below into {count} different search queries \
for finding passages in the user's Documents. Vary the wording and the angle: use \
synonyms, more specific and more general phrasings. Reply with the queries only, \
one per line, with no numbering or comments.

Question: {question}"""
REWRITES = 3
# How many Chunks RAG fusion keeps: as many as one hybrid search returns by default.
FUSED_LIMIT = 6


@chain
def rewrites(question: str) -> list[str]:
    """`question` rewritten into several search queries on the paid model. Multi-query
    and RAG fusion share these. A reply with no usable query falls back to the question."""
    reply = models.chat_model(kind="rewrite").invoke(
        REWRITE_PROMPT.format(count=REWRITES, question=question)
    )
    queries = [line.strip(" \t-*•").strip() for line in reply.text.splitlines()]
    queries = list(dict.fromkeys(q for q in queries if q))
    return queries or [question]


def _rewrite_results(question: str) -> list[list[SearchHit]]:
    return [hybrid_search(query) for query in rewrites.invoke(question)]


def _key(hit: SearchHit) -> tuple[str, int, str]:
    return (hit.document, hit.page, hit.text)


@chain
def multi_query(question: str) -> list[SearchHit]:
    """The union of every rewrite's Chunks, each once, in the order they were found."""
    union: dict[tuple[str, int, str], SearchHit] = {}
    for hits in _rewrite_results(question):
        for hit in hits:
            union.setdefault(_key(hit), hit)
    return list(union.values())


@chain
def rag_fusion(question: str) -> list[SearchHit]:
    """Every rewrite's Chunks ranked by reciprocal rank fusion: the Chunks most rewrites
    rank highly come first. Keeps as many as one hybrid search returns."""
    scores: dict[tuple[str, int, str], float] = {}
    first: dict[tuple[str, int, str], SearchHit] = {}
    for hits in _rewrite_results(question):
        for rank, hit in enumerate(hits, start=1):
            key = _key(hit)
            first.setdefault(key, hit)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:FUSED_LIMIT]
    return [replace(first[key], score=scores[key]) for key in ranked]


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
        name="multi_query",
        label="Multi-query",
        description=(
            "Rewrite a vague or loosely-worded question into several search queries and "
            "gather the passages any of them finds. Best when the question's wording may "
            "not match the Documents' wording and recall matters most."
        ),
        examples=(
            "Anything about money problems with the suppliers?",
            "What stuff did we agree on for the new office?",
            "Is there something on staff feeling burned out?",
            "What did people say about the launch going badly?",
            "Do my notes mention issues with the website?",
        ),
        chain=multi_query,
    ),
    Technique(
        name="rag_fusion",
        label="RAG fusion",
        description=(
            "Rewrite the question into several search queries and rank the passages by "
            "how many queries agree on them. Best for broad or ambiguous questions where "
            "the most relevant passages should come first."
        ),
        examples=(
            "What are the main risks in our expansion plans?",
            "How do we usually handle customer complaints?",
            "What is our approach to hiring engineers?",
            "What were the key takeaways from the quarterly reviews?",
            "How is the company thinking about sustainability?",
        ),
        chain=rag_fusion,
    ),
)


def get(name: str) -> Technique:
    """The registered technique called `name`."""
    for technique in TECHNIQUES:
        if technique.name == name:
            return technique
    raise KeyError(f"No Retrieval Technique {name!r}; known: {[t.name for t in TECHNIQUES]}")
