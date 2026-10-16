"""The Retrieval Techniques the Knowledge Agent can pick from, all registered in TECHNIQUES.

Every Retrieval Technique is a `@chain` with one signature: a question in, Chunks
(`SearchHit`s, each with its Citation) out. Adding one is one chain plus one entry
in TECHNIQUES; the Knowledge Agent's tools, the sidebar dropdown, the picking
modes (`perry.picking`) and (later) the benchmark all read the registry.

A technique that calls a chat model gets it from `models.chat_model(kind=...)`, with a
kind naming what the call is for (e.g. "rewrite"), so the Usage Ledger shows what each
technique costs and its calls count toward the Turn's Token Ceiling.
"""

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from itertools import zip_longest

from langchain_core.runnables import Runnable, chain
from langsmith import traceable

from perry import colbert as colbert_index
from perry import models, summaries, usage
from perry.search import (
    FINDINGS,
    RRF_K,
    SearchHit,
    default_limit,
    document_count,
    hybrid_search,
    summary_search,
)
from perry.self_query import self_query


@dataclass(frozen=True)
class Technique:
    """One Retrieval Technique.

    `name` is its id: the Knowledge Agent's tool name and the per-request override's
    value. `description` says when to use it; it is the tool description the Knowledge
    Agent's model picks by. `examples` are hand-written questions it suits best, for
    picking by similarity; never copy them from the benchmark (a test checks).
    `summary_kind` is the kind of Summaries a technique searches (`summaries.DOCUMENT`
    or `summaries.RAPTOR`), if it searches any: it searches only Documents whose
    Summaries of that kind are ready, and isn't picked until at least one is.
    """

    name: str
    label: str
    description: str
    examples: tuple[str, ...]
    chain: Runnable[str, list[SearchHit]]
    summary_kind: str | None = None


@chain
def hybrid(question: str) -> list[SearchHit]:
    """Today's baseline: one hybrid search for the question as asked."""
    return hybrid_search(question)


# Question Decomposition asks for at most this many sub-questions, and searches each
# for this many Chunks: together they bound its calls and the findings' length.
MAX_SUB_QUESTIONS = 5
CHUNKS_PER_SUB_QUESTION = 4

DECOMPOSE_PROMPT = """Split the question below into the simpler sub-questions that together answer it, such as one per fact it compares or chains. Each sub-question must be self-contained and searchable on its own. Write at most {most}, one per line, with no numbering and nothing else. If the question asks for only one fact, write it unchanged.

Question: {question}"""


# A bullet or number starting a line, e.g. "- ", "* ", "2. " or "3) ".
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def sub_questions(question: str) -> list[str]:
    """`question` split by the model into at most MAX_SUB_QUESTIONS sub-questions, or
    the question itself if the model gave none."""
    prompt = DECOMPOSE_PROMPT.format(most=MAX_SUB_QUESTIONS, question=question)
    reply = models.chat_model(kind="decompose").invoke(prompt).text
    return _lines(reply)[:MAX_SUB_QUESTIONS] or [question]


def _lines(reply: str) -> list[str]:
    """The distinct non-empty lines of a model's reply, without the bullets or numbers
    models often put on them anyway."""
    lines = (_LIST_MARKER.sub("", line).strip() for line in reply.splitlines())
    return list(dict.fromkeys(line for line in lines if line))


def _key(hit: SearchHit) -> tuple[str, int, str]:
    """What makes two hits the same Chunk."""
    return (hit.document, hit.page, hit.text)


def _union(*rankings: Iterable[SearchHit | None]) -> list[SearchHit]:
    """Every Chunk in `rankings`, each once, in the order first found (None is skipped)."""
    union: dict[tuple[str, int, str], SearchHit] = {}
    for hits in rankings:
        for hit in hits:
            if hit is not None:
                union.setdefault(_key(hit), hit)
    return list(union.values())


@chain
def question_decomposition(question: str) -> list[SearchHit]:
    """Question Decomposition: hybrid search for each sub-question, then their Chunks
    together, taking each sub-question's best before anyone's second best."""
    rankings = [
        hybrid_search(sub_question, limit=CHUNKS_PER_SUB_QUESTION)
        for sub_question in sub_questions(question)
    ]
    # Tier by tier: every sub-question's best, then every second best, and so on.
    return _union(*zip_longest(*rankings))


REWRITE_PROMPT = """Rewrite the question below into {count} different search queries \
for finding Chunks (excerpts) of the user's Documents. Vary the wording and the angle: use \
synonyms, more specific and more general phrasings. Reply with the queries only, \
one per line, with no numbering or comments.

Question: {question}"""
REWRITES = 3


@chain
def rewrites(question: str) -> list[str]:
    """`question` rewritten into several search queries on the paid model. Multi-query
    and RAG fusion share these. A reply with no usable query falls back to the question."""
    reply = models.chat_model(kind="rewrite").invoke(
        REWRITE_PROMPT.format(count=REWRITES, question=question)
    )
    return _lines(reply.text) or [question]


def _rewrite_results(question: str) -> list[list[SearchHit]]:
    return [hybrid_search(query) for query in rewrites.invoke(question)]


@chain
def multi_query(question: str) -> list[SearchHit]:
    """The union of every rewrite's Chunks, each once, in the order they were found."""
    return _union(*_rewrite_results(question))


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
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[: default_limit()]
    return [replace(first[key], score=scores[key]) for key in ranked]


STEP_BACK_PROMPT = """Write a broader, more general version of this question: one about \
the principle, cause or background behind it, so that searching for it finds the context \
needed to answer the original. Reply with the broader question only.

Question: {question}"""


@chain
def step_back(question: str) -> list[SearchHit]:
    """Hybrid search for the question and for a broader version of it, the question's
    own Chunks first, then the background the broader question found."""
    prompt = STEP_BACK_PROMPT.format(question=question)
    broader = models.chat_model(kind="step_back").invoke(prompt).text.strip()
    return _union(hybrid_search(question), hybrid_search(broader) if broader else [])


HYDE_PROMPT = """Write a short passage (two or three sentences) that could appear in a \
document and answers this question. Make up plausible details if you must: the passage \
is only used to search with, never shown or cited. Reply with the passage only.

Question: {question}"""


@chain
def hyde(question: str) -> list[SearchHit]:
    """Hybrid search with a hypothetical answer instead of the question: its embedding
    (taken as a Document's, since it reads like one) and its words. Only real Chunks
    come back; the hypothetical answer itself is never a hit, so it is never cited."""
    prompt = HYDE_PROMPT.format(question=question)
    hypothetical = models.chat_model(kind="hyde").invoke(prompt).text.strip()
    if not hypothetical:
        return hybrid_search(question)
    return hybrid_search(hypothetical, embed=lambda text: models.embed_documents([text])[0])


@chain
def document_summaries(question: str) -> list[SearchHit]:
    """Multi-representation indexing: match the question against Document Summaries
    (only Documents whose Summaries are ready), then return those Documents' best
    Chunks. A Summary is never returned, so never cited."""
    return summary_search(question)


@chain
def colbert(question: str) -> list[SearchHit]:
    """Late interaction: every word of the question matched against every word of each
    Chunk, over ColBERT's own index (docs/adr/0002)."""
    return colbert_index.search(question)


@chain
def raptor(question: str) -> list[SearchHit]:
    """RAPTOR: match the question against every node of the Documents' RAPTOR trees and
    the cross-Document level above them (only ready trees), then return the leaf Chunks
    beneath the best nodes. A node's Summary is never returned, so never cited."""
    return summary_search(question, kind=summaries.RAPTOR)


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
        name="question_decomposition",
        label="Question Decomposition",
        description=(
            "Split the question into sub-questions and search the user's Documents for "
            "each. Best for one question that compares or chains several facts, e.g. two "
            "figures side by side, or a fact that depends on another. Not for requests "
            "that need other Agents (such as emailing what was found)."
        ),
        examples=(
            "Did we spend more on catering or on travel for the conference?",
            "How does this year's rent compare with last year's?",
            "Who approved the vendor contract, and when does it expire?",
            "Which office has more staff, Hanoi or Da Nang, and who manages each?",
            "What changed between the first and second drafts of the proposal?",
            "Is the project deadline before or after the funding decision date?",
        ),
        chain=question_decomposition,
    ),
    Technique(
        name="multi_query",
        label="Multi-query",
        description=(
            "Rewrite a vague or loosely-worded question into several search queries and "
            "gather the Chunks any of them finds. Best when the question's wording may "
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
            "Rewrite the question into several search queries and rank the Chunks by "
            "how many queries agree on them. Best for broad or ambiguous questions where "
            "the most relevant Chunks should come first."
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
    Technique(
        name="step_back",
        label="Step-back",
        description=(
            "Search the user's Documents for the question and for a broader version of "
            "it, so background and principles are found alongside the specifics. Best "
            "for 'why' questions and questions about the reasons or rules behind something."
        ),
        examples=(
            "Why did we switch to a four-day sprint cycle?",
            "Why was the supplier contract not renewed?",
            "What is the reasoning behind the travel reimbursement rules?",
            "Why does the onboarding plan start with a shadowing week?",
            "What principle decides which projects get funded first?",
        ),
        chain=step_back,
    ),
    Technique(
        name="hyde",
        label="HyDE",
        description=(
            "Write a short hypothetical answer to the question and search the user's "
            "Documents for Chunks like it. Best for short or casually worded questions "
            "whose words are far from how the Documents would phrase the answer."
        ),
        examples=(
            "How much can we blow on snacks?",
            "Are we allowed to work from abroad?",
            "Who pays when my laptop breaks?",
            "Can I bring my dog to the office?",
            "How long until I get paid back for the train?",
        ),
        chain=hyde,
    ),
    Technique(
        name="document_summaries",
        label="Document Summaries",
        description=(
            "Find the Documents whose Summaries best match the question, then return "
            "their best-matching Chunks. Best for finding which Document covers a "
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
        summary_kind=summaries.DOCUMENT,
    ),
    Technique(
        name="raptor",
        label="RAPTOR",
        description=(
            "Match the question against Summaries of related Chunks at every level, "
            "from groups of Chunks up to themes across all the user's Documents, then "
            "return the Chunks beneath the best matches. Best for thematic or 'overall' "
            "questions that need the big picture across many Chunks or Documents."
        ),
        examples=(
            "What themes keep coming up across all my meeting notes?",
            "Overall, how did the team feel about the reorganisation?",
            "What is the big picture of our product strategy across these files?",
            "What do my documents say in general about working with contractors?",
            "Across everything I uploaded, what are the recurring cost concerns?",
        ),
        chain=raptor,
        summary_kind=summaries.RAPTOR,
    ),
    Technique(
        name="self_query",
        label="Self-query",
        description=(
            "Turn the question into a filter on which Documents to search (by name, file "
            "type, upload date, page, Document date or topic) plus a search query. Best "
            "when the question limits itself to certain Documents, e.g. 'in the 2025 "
            "budget notes', 'in my PDFs', 'on page 3', or 'in documents from last year'."
        ),
        examples=(
            "In the 2023 lease agreement, what is the monthly rent?",
            "What do my PDF files say about overtime pay?",
            "According to the travel policy document, page 4, who signs off on flights?",
            "What did the documents I uploaded last week say about the new hire?",
            "In the meeting notes from March 2024, what was decided about the logo?",
            "Which deadlines are mentioned in my markdown notes on fundraising?",
        ),
        chain=self_query,
    ),
    Technique(
        name="colbert",
        label="ColBERT",
        description=(
            "Search the user's Documents matching the question word by word against "
            "every Chunk. Best when the question hinges on an exact name, code, "
            "product or rare term that must appear in the Chunk."
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


def available() -> list[Technique]:
    """The registered techniques that can be picked right now: all of them, except one
    that searches Summaries while no Document's Summaries of its kind are ready. (It
    searches only ready Documents; a forced technique runs regardless.)"""
    kinds = {t.summary_kind for t in TECHNIQUES if t.summary_kind}
    ready = {kind for kind in kinds if summaries.count(kind, ready=True)}
    return [t for t in TECHNIQUES if not t.summary_kind or t.summary_kind in ready]


_SUMMARIES_NAMED = {summaries.DOCUMENT: "Summaries", summaries.RAPTOR: "RAPTOR trees"}


def nothing_found(technique: Technique, hits: list[SearchHit]) -> str:
    """Why `technique` found no Chunks (`hits` is its empty result), truthfully, for the
    Knowledge Agent's model to tell the user or search with another technique."""
    if document_count() == 0:
        return "No Chunks found: the user has not uploaded any Documents yet."
    nothing = f"No Chunks found with {technique.label}"
    if reason := getattr(hits, "reason", ""):
        return f"{nothing}: {reason}. Tell the user, or search with another technique."
    if technique.summary_kind:
        waiting = summaries.count(technique.summary_kind, ready=False)
        if waiting:
            what = _SUMMARIES_NAMED[technique.summary_kind]
            documents = "1 Document" if waiting == 1 else f"{waiting} Documents"
            note = (
                f"{nothing}: the {what} of {documents} aren't ready yet, and only ready "
                "ones are searched. Search with another technique."
            )
            if not models.summaries_supported():
                note += f" ({models.summary_model_hint()})"
            return note
    return (
        f"{nothing}. Tell the user the Documents don't seem to contain it, or search "
        "with another technique."
    )


def get(name: str) -> Technique:
    """The registered technique called `name`."""
    for technique in TECHNIQUES:
        if technique.name == name:
            return technique
    raise KeyError(f"No Retrieval Technique {name!r}; known: {[t.name for t in TECHNIQUES]}")


@dataclass(frozen=True)
class Candidate:
    """One Chunk the reranker scored: its Citation, its score, its place in the
    technique's own order (1 = first), and whether it was among the Chunks kept."""

    citation: str
    score: float
    rank_before: int
    kept: bool


@dataclass(frozen=True)
class Found:
    """What one search found: the Chunks for the Knowledge Agent, whether the reranker
    chose and ordered them, and, if it did, every candidate it scored, best first."""

    hits: list[SearchHit]
    reranked: bool = False
    problem: str = ""  # why the Chunks weren't reranked, if reranking failed
    candidates: list[Candidate] = field(default_factory=list)


def search(technique: Technique, question: str) -> Found:
    """Search with `technique`, then, if there is a reranker, keep the FINDINGS Chunks it
    scores best against the question, best first, leaving out those it scores below
    `models.RERANK_MIN_SCORE` if set (the best is always kept). The technique gathers a wider pool of
    candidates for it to choose from (`search.default_limit`). Without a reranker, the
    technique's own Chunks, as they came; if reranking fails, its best FINDINGS."""
    hits = technique.chain.invoke(question)
    if not hits or not models.reranking_supported():
        return Found(hits)
    try:
        return _rerank(question, hits)
    except usage.TurnTokenCeilingReached:
        raise  # the Turn stops: not a reranker failure
    except Exception as error:  # noqa: BLE001 - the technique's own order still answers
        return Found(hits[:FINDINGS], problem=f"Reranking failed ({error})")


def _candidates_only(found: Found) -> dict:
    return {"candidates": [asdict(candidate) for candidate in found.candidates]}


def _citations_only(inputs: dict) -> dict:
    return {"question": inputs["question"], "pool": [hit.citation for hit in inputs["hits"]]}


# Traced (when tracing is on) so a Turn's trace shows every candidate's score.
@traceable(
    name="Reranking", process_inputs=_citations_only, process_outputs=_candidates_only
)
def _rerank(question: str, hits: list[SearchHit]) -> Found:
    scores = models.rerank(question, [hit.text for hit in hits])
    ranked = sorted(
        zip(scores, range(1, len(hits) + 1), hits, strict=True),
        key=lambda scored: scored[0],
        reverse=True,
    )
    cutoff = models.RERANK_MIN_SCORE
    kept = [
        place == 0
        or (place < FINDINGS and (cutoff is None or score >= cutoff))
        for place, (score, _, _) in enumerate(ranked)
    ]
    candidates = [
        Candidate(hit.citation, float(score), rank, kept=keep)
        for (score, rank, hit), keep in zip(ranked, kept, strict=True)
    ]
    hits = [hit for (_, _, hit), keep in zip(ranked, kept, strict=True) if keep]
    return Found(hits, reranked=True, candidates=candidates)
