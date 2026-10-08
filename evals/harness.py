"""Run one eval Turn: hard checks on routing and Citations, plus a non-gating LLM judge score.

The harness drives Turns through anything with `send(message) -> reply` and
`consulted() -> [Agent tool names]`: the real `Session` below, or the tests' `perry` fixture.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from perry import models


@dataclass(frozen=True)
class EvalTurn:
    """One request and what a good answer to it must show.

    `agents`: the Agents the Supervisor must consult, by short name ("knowledge",
    "analysis", "task"); empty means none. `citations`: Citations the reply must contain,
    as a Document name ("notes.md", matches any page) or an exact Citation
    ("report.pdf#page=4"). `facts`: what the answer should say, scored by the judge only.
    """

    id: str
    message: str
    agents: tuple[str, ...]
    citations: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()


class Driver(Protocol):
    def send(self, message: str) -> str: ...

    def consulted(self) -> list[str]: ...


@dataclass
class Result:
    turn: EvalTurn
    reply: str = ""
    consulted: list[str] = field(default_factory=list)
    missing_citations: list[str] = field(default_factory=list)
    error: str | None = None
    quality: float | None = None  # 0..1 from the judge; None if it gave no usable score
    judge_comment: str = ""

    @property
    def routing_ok(self) -> bool:
        return self.error is None and set(self.consulted) == set(self.turn.agents)

    @property
    def citations_ok(self) -> bool:
        return self.error is None and not self.missing_citations

    @property
    def passed(self) -> bool:
        """Only the hard checks count. The judge's score never fails a Turn."""
        return self.routing_ok and self.citations_ok


def run_eval_turn(turn: EvalTurn, driver: Driver) -> Result:
    result = Result(turn)
    try:
        result.reply = driver.send(turn.message)
        result.consulted = [agent_name(tool) for tool in driver.consulted()]
    except Exception as error:  # noqa: BLE001 - a crashed Turn is a failed Turn, not a crashed run
        result.error = f"{type(error).__name__}: {error}"
        return result
    result.missing_citations = [c for c in turn.citations if not cites(result.reply, c)]
    result.quality, result.judge_comment = judge(turn, result.reply)
    return result


def agent_name(tool: str) -> str:
    """`ask_knowledge_agent` -> `knowledge`."""
    match = re.fullmatch(r"ask_(\w+)_agent", tool)
    return match.group(1) if match else tool


def cites(reply: str, citation: str) -> bool:
    """Whether `reply` contains `citation` in square brackets (a Document name matches any page)."""
    for group in re.findall(r"\[([^\[\]]+)\]", reply):
        for cited in re.split(r"[,;]", group):
            cited = cited.strip()
            if cited == citation or cited.startswith(citation + "#"):
                return True
    return False


JUDGE_PROMPT = """You grade answers from Perry, a personal assistant that answers from \
the user's own Documents and cites them in square brackets.

Request: {message}

Perry's answer:
{reply}

What a good answer says:
{facts}

Score the answer from 1 (wrong or unhelpful) to 5 (correct, complete and concise). \
Reply with JSON only: {{"score": <1-5>, "comment": "<one sentence>"}}"""


def judge(turn: EvalTurn, reply: str) -> tuple[float | None, str]:
    """Answer quality from 0 to 1 with a one-line reason. Never raises: it only informs."""
    facts = "\n".join(f"- {f}" for f in turn.facts) or "- (no specific facts; judge helpfulness)"
    prompt = JUDGE_PROMPT.format(message=turn.message, reply=reply, facts=facts)
    try:
        text = models.chat_model().invoke(prompt).text
        verdict = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
        score = float(verdict["score"])
    except Exception as error:  # noqa: BLE001
        return None, f"judge gave no usable score ({type(error).__name__})"
    if not 1 <= score <= 5:
        return None, f"judge score out of range: {score}"
    return (score - 1) / 4, str(verdict.get("comment", ""))
