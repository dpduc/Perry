"""Knowledge Agent: answers from your Documents, with a Citation for every claim.

Interface (keep it retrieval-method agnostic, see docs/adr/0001): question in,
cited findings out.
"""

import os
import re
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.tools import BaseTool, StructuredTool

from perry import models, picking, retries, techniques
from perry.techniques import Technique

SYSTEM_PROMPT = """You are Perry's Knowledge Agent. You answer questions using ONLY \
Chunks (excerpts) of the user's Documents, found with your search tools. Each tool is \
a different way of searching (a Retrieval Technique); pick the one whose description \
best fits the question.

- Search before answering. If the first search misses, rephrase and search again \
(at most 3 searches).
- Cite every claim with the Chunk's citation in square brackets, exactly as given, \
e.g. [report.pdf#page=4].
- If the Chunks don't answer the question, say plainly that the Documents don't \
contain it. Never fill gaps from general knowledge.
- Be concise: report the findings, not a polished essay."""

# How a technique was chosen, as the "Found with" line says it.
PICKED_LOGICALLY = "picked logically"  # the Knowledge Agent's model chose it
PICKED_SEMANTICALLY = "picked semantically"  # its example questions were nearest
FORCED = "forced"  # the per-request override named it


class _Search:
    """One question's searches: which Retrieval Technique answers it. The first technique
    to find Chunks answers the question, so every later search for it uses that one too
    (one technique per question); a technique that found nothing doesn't count."""

    def __init__(self):
        self.answering: Technique | None = None  # the technique that found Chunks
        self.last: Technique | None = None  # the technique the latest search used


def technique_tool(technique: Technique, search: _Search | None = None) -> BaseTool:
    """`technique` as a tool the Knowledge Agent can call, described for picking. Within
    one question's `search`, calling it after another technique found Chunks searches
    with that technique instead."""
    search = search or _Search()

    def run(question: str) -> str:
        used = search.answering or technique
        hits = used.chain.invoke(question)
        search.last = used
        if not hits:
            return techniques.nothing_found(used, hits)
        search.answering = used
        found = "\n\n".join(f"[{hit.citation}]\n{hit.text}" for hit in hits)
        if used is not technique:
            found = (
                f"(Searched with {used.label}: one Retrieval Technique per question, and "
                f"{used.label} already found Chunks for this one.)\n\n{found}"
            )
        return found

    return StructuredTool.from_function(
        run, name=technique.name, description=technique.description
    )


def knowledge_agent(tools: list[BaseTool]):
    # Built per call (cheap) so it always uses the current model provider.
    return create_agent(
        models.chat_model(),
        tools=tools,
        middleware=[retries.RetryFailedSteps()],
        system_prompt=SYSTEM_PROMPT,
        name="knowledge_agent",
    )


@dataclass(frozen=True)
class Findings:
    """The Knowledge Agent's cited findings, and the Retrieval Technique that found them
    as the "Found with" line names it, e.g. "Hybrid search (picked logically)" (none if
    the Knowledge Agent didn't search)."""

    text: str
    found_with: tuple[str, ...] = ()


def find(
    question: str, forced: str | None = None, picking_mode: str | None = None
) -> Findings:
    """Answer `question` from the Documents. `forced` names the one Retrieval Technique
    to use (the per-request override); by default one is picked in `picking_mode`, or
    the picking mode in force (see `perry.picking`)."""
    if forced:
        offered, how = [techniques.get(forced)], FORCED
    else:
        candidates = techniques.available()
        picked = picking.pick(question, candidates, picking_mode)
        if picked:
            offered, how = [picked], PICKED_SEMANTICALLY
        else:
            offered, how = candidates, PICKED_LOGICALLY
    search = _Search()
    agent = knowledge_agent([technique_tool(t, search) for t in offered])
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    # The technique that found the Chunks or, if none did, the one that searched last.
    used = search.answering or search.last
    found_with = (f"{used.label} ({how})",) if used else ()
    return Findings(result["messages"][-1].text, found_with)


def ask(question: str, forced: str | None = None) -> str:
    """The findings for `question` as an answer: text, then the "Found with" line."""
    findings = find(question, forced)
    return with_found_with(findings.text, findings.found_with)


def found_with_shown() -> bool:
    """Whether answers end with the "Found with" line (PERRY_SHOW_FOUND_WITH, default on)."""
    setting = os.getenv("PERRY_SHOW_FOUND_WITH", "true").strip().lower()
    return setting not in {"0", "false", "no", "off"}


def with_found_with(answer: str, found_with: list[str] | tuple[str, ...]) -> str:
    """`answer` ending with the line naming the Retrieval Techniques that found it, unless
    none did or the setting turns the line off. Added in code, like the Citations."""
    found_with = list(dict.fromkeys(found_with))
    if not found_with or not found_with_shown():
        return answer
    # Straight under the Citations line, if there is one: together they end the answer.
    last_line = answer.rsplit("\n", 1)[-1]
    gap = "\n" if last_line.startswith(CITATIONS) else "\n\n"
    return f"{answer}{gap}Found with: {'; '.join(found_with)}"


CITATIONS = "Citations: "

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
