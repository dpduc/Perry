"""Knowledge Agent: answers from your Documents, with a Citation for every claim.

Interface (keep it retrieval-method agnostic, see docs/adr/0001): question in,
cited findings out.
"""

import re

from langchain.agents import create_agent
from langchain_core.tools import tool

from perry import models, retries
from perry.search import hybrid_search

SYSTEM_PROMPT = """You are Perry's Knowledge Agent. You answer questions using ONLY \
passages from the user's Documents, found with the search_documents tool.

- Search before answering. If the first search misses, rephrase and search again \
(at most 3 searches).
- Cite every claim with the passage's citation in square brackets, exactly as given, \
e.g. [report.pdf#page=4].
- If the passages don't answer the question, say plainly that the Documents don't \
contain it. Never fill gaps from general knowledge.
- Be concise: report the findings, not a polished essay."""


@tool
def search_documents(query: str) -> str:
    """Search the user's Documents. Returns the most relevant passages with their citations."""
    hits = hybrid_search(query)
    if not hits:
        return "No passages found. The user has not uploaded any Documents yet."
    return "\n\n".join(f"[{hit.citation}]\n{hit.text}" for hit in hits)


def knowledge_agent():
    # Built per call (cheap) so it always uses the current model provider.
    return create_agent(
        models.chat_model(),
        tools=[search_documents],
        middleware=[retries.RetryFailedSteps()],
        system_prompt=SYSTEM_PROMPT,
        name="knowledge_agent",
    )


def ask(question: str) -> str:
    result = knowledge_agent().invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].text


# A Citation as the findings give it, e.g. [report.pdf#page=4].
_CITATION = re.compile(r"\[[^\[\]\n]+\]")


def citations_at_end(answer: str, findings: list[str]) -> str:
    """`answer` with every Citation in `findings` moved to a last line, so they don't
    interrupt the reading. Citations the answer used come first, then any it left out.

    Code, not the prompt, makes sure an answer built on the findings keeps their Citations.
    Only Citations the findings gave are moved; other brackets (links, chart markers) stay.
    """
    known = list(dict.fromkeys(c for text in findings for c in _CITATION.findall(text)))
    if not known:
        return answer
    used = sorted((c for c in known if c in answer), key=answer.index)
    body = answer
    for found in known:
        body = re.sub(rf"[ \t]*{re.escape(found)}", "", body)
    # Tidy what removal leaves: lines now empty, and space left before punctuation.
    body = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", body)
    body = re.sub(r"(?m)^[ \t]+$", "", body).strip()
    cited = used + [c for c in known if c not in used]
    return f"{body}\n\nCitations: {' '.join(cited)}"
