"""The Retrieval Techniques the Knowledge Agent can pick from, all registered in TECHNIQUES.

Every Retrieval Technique is a `@chain` with one signature: a question in, Chunks
(`SearchHit`s, each with its Citation) out. Adding one is one chain plus one entry
in TECHNIQUES; the Knowledge Agent's tools, the sidebar dropdown and (later) the
picking modes and the benchmark all read the registry.

A technique that calls a chat model gets it from `models.chat_model(kind=...)`, with a
kind naming what the call is for (e.g. "rewrite"), so the Usage Ledger shows what each
technique costs and its calls count toward the Turn's Token Ceiling.
"""

import re
from dataclasses import dataclass
from itertools import zip_longest

from langchain_core.runnables import Runnable, chain

from perry import models
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
    # Models often number or bullet the lines anyway: drop that.
    lines = (_LIST_MARKER.sub("", line).strip() for line in reply.splitlines())
    found = list(dict.fromkeys(line for line in lines if line))
    return found[:MAX_SUB_QUESTIONS] or [question]


@chain
def question_decomposition(question: str) -> list[SearchHit]:
    """Question Decomposition: hybrid search for each sub-question, then their Chunks
    together, taking each sub-question's best before anyone's second best."""
    rankings = [
        hybrid_search(sub_question, limit=CHUNKS_PER_SUB_QUESTION)
        for sub_question in sub_questions(question)
    ]
    union: dict[tuple[str, int, str], SearchHit] = {}
    for tier in zip_longest(*rankings):
        for hit in tier:
            if hit is not None:
                union.setdefault((hit.document, hit.page, hit.text), hit)
    return list(union.values())


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
    hits = hybrid_search(question)
    seen = {(hit.document, hit.page, hit.text) for hit in hits}
    for hit in hybrid_search(broader) if broader else []:
        if (hit.document, hit.page, hit.text) not in seen:
            seen.add((hit.document, hit.page, hit.text))
            hits.append(hit)
    return hits


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
            "Documents for passages like it. Best for short or casually worded questions "
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
)


def get(name: str) -> Technique:
    """The registered technique called `name`."""
    for technique in TECHNIQUES:
        if technique.name == name:
            return technique
    raise KeyError(f"No Retrieval Technique {name!r}; known: {[t.name for t in TECHNIQUES]}")
