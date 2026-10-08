"""Knowledge Agent: answers from your Documents, with a Citation for every claim.

Interface (keep it retrieval-method agnostic, see docs/adr/0001): question in,
cited findings out.
"""

from functools import cache

from langchain.agents import create_agent
from langchain_core.tools import tool

from perry import models
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


@cache
def knowledge_agent():
    return create_agent(
        models.chat_model(),
        tools=[search_documents],
        system_prompt=SYSTEM_PROMPT,
        name="knowledge_agent",
    )


def ask(question: str) -> str:
    result = knowledge_agent().invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].text
