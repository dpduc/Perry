"""The Supervisor: plans each Turn, delegates to Agents, and composes the answer.

Turn interface: `run_turn` answers a User's message with either the final answer (a str) or
a pending `Confirmation` for one Write Action. `resume_turn` continues that Turn with
the user's decision (`Approve`, with optional edits, or `Reject` with a note), and
again returns an answer or the next Confirmation. A new message on a Thread with a
pending Confirmation cancels it. Confirmations are LangGraph interrupts raised by the
Task Agent's Confirmation gate, kept in the Thread's checkpoint.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.config import get_config
from langgraph.errors import GraphRecursionError
from langgraph.types import Command, StateSnapshot

from perry import charts, db, models, privacy, retries, usage, workspace
from perry.agents import analysis, knowledge, task
from perry.images import user_message
from perry.users import Profile, User

# The per-request override: the Retrieval Technique the Knowledge Agent must use, if any.
RETRIEVAL_TECHNIQUE = "perry_retrieval_technique"
# The Profile of the User the Turn is for, which the Supervisor is given on every model call.
_PROFILE = "perry_profile"

# Each step is one Supervisor decision (call an Agent, or answer).
MAX_STEPS_PER_TURN = 6

_PROMPT = """You are Perry, a personal assistant. You coordinate specialist Agents \
and answer the user.

Agents available as tools:
- ask_knowledge_agent: anything that could be answered from the user's own Documents \
(notes, reports, PDFs they uploaded).
{task_agent}- ask_analysis_agent: questions about the user's Datasets (CSV or spreadsheet data), \
such as totals, averages, counts or comparisons.

Rules:
- If a question might be answered by the user's Documents, ask the Knowledge Agent \
rather than answering from general knowledge.
- Keep every citation in square brackets exactly as the Agent gave it.
- If an Agent reports it couldn't find something, tell the user so. Never invent results.
- Small talk and questions about yourself need no Agent.
- When the Analysis Agent shows the SQL or Python it used, include it unchanged in your answer.
- Keep chart markers like [chart: <id>.png] exactly as given; they show as images.
- If a step comes back FAILED, give the results of the steps that worked, then say which \
step failed. Never guess what a failed step would have returned.
{task_agent_rules}"""

# Email and calendar are the Owner's (ADR-0004): only the Owner's Supervisor has the Task Agent.
_TASK_AGENT = """- ask_task_agent: the user's calendar and email: what's on, what someone wrote, \
and sending, replying, scheduling, moving or deleting (the user confirms each of those).
"""
_TASK_AGENT_RULES = """- To act on a request that needs other information first (e.g. email a \
summary of some notes), get that information, then give the Task Agent the complete content \
to send.
- If the user rejected a Write Action with a note, ask the Task Agent again with a \
request revised to follow the note."""
_NOT_THE_OWNER = """- Email and calendar aren't available to this user: they belong to Perry's \
Owner. If asked about email, calendar or meetings, say that email and calendar aren't \
available to them."""

SYSTEM_PROMPT = _PROMPT.format(task_agent=_TASK_AGENT, task_agent_rules=_TASK_AGENT_RULES)
NON_OWNER_PROMPT = _PROMPT.format(task_agent="", task_agent_rules=_NOT_THE_OWNER)


@tool(response_format="content_and_artifact")
def ask_knowledge_agent(question: str, config: RunnableConfig) -> tuple[str, dict]:
    """Ask the Knowledge Agent a self-contained question about the user's Documents.
    Returns findings with citations."""
    # The per-request override rides in the Turn's config, unseen by the Supervisor's model.
    forced = config.get("configurable", {}).get(RETRIEVAL_TECHNIQUE)
    findings = knowledge.find(question, forced)
    # The model reads the findings; the code composing the answer reads the artifact.
    return findings.text, {"found_with": list(findings.found_with)}


# Never retried: a retry would re-run the Task Agent, and with it any Write Action.
@retries.never_retry
@tool
def ask_task_agent(request: str) -> str:
    """Ask the Task Agent to look something up in, or act on, the user's calendar or email.
    Give a self-contained request, with the full content of anything to send.
    Returns what it found or did."""
    return task.ask(request)


@tool
def ask_analysis_agent(question: str) -> str:
    """Ask the Analysis Agent a self-contained question about the user's Datasets.
    Returns the findings and the SQL that produced them."""
    return analysis.ask(question)


def profile_block(profile: Profile, now: datetime) -> str:
    """What the Supervisor is told about the User it is answering, at `now` (their time)."""
    lines = [
        "About the user you are answering:",
        f"- Name: {profile.display_name}. Address them by this name.",
        (
            f"- Time zone: {profile.time_zone}. It is now {now:%A %Y-%m-%d %H:%M} there. "
            "Work out relative dates (today, tomorrow, this week) in this time zone, and "
            "give Agents exact dates."
        ),
    ]
    if profile.about:
        lines.append(f"- About them, in their words (take it into account): {profile.about}")
    return "\n".join(lines)


class GiveProfile(AgentMiddleware):
    """Adds the Turn's User's Profile, with their local date and time, to the system prompt."""

    def wrap_model_call(self, request, handler):
        profile = get_config().get("configurable", {}).get(_PROFILE)
        if profile is None:
            return handler(request)
        block = profile_block(profile, datetime.now(ZoneInfo(profile.time_zone)))
        prompt = f"{request.system_prompt}\n\n{block}" if request.system_prompt else block
        return handler(request.override(system_message=SystemMessage(prompt)))


@cache
def _checkpointer() -> PostgresSaver:
    checkpointer = PostgresSaver(db.pool())
    checkpointer.setup()
    return checkpointer


def supervisor(user: User | None = None):
    """The Supervisor for `user`'s Turns: only the Owner's has the Task Agent. With no
    `user` it is the Owner's, for reading a Thread's state (which the tools don't change)."""
    owner = user is None or user.is_owner
    # Built per call (cheap) so it always uses the current model provider.
    return create_agent(
        models.chat_model(),
        tools=[
            ask_knowledge_agent,
            *([ask_task_agent] if owner else []),
            ask_analysis_agent,
        ],
        middleware=[
            retries.RetryFailedSteps(),
            GiveProfile(),
        ],
        system_prompt=SYSTEM_PROMPT if owner else NON_OWNER_PROMPT,
        checkpointer=_checkpointer(),
        name="supervisor",
    )


@dataclass(frozen=True)
class Confirmation:
    """One Write Action waiting for the user's decision.

    `summary` says exactly what will happen; `fields` are the Write Action's arguments,
    any of which the user may edit before approving.
    """

    action: str
    summary: str
    fields: dict[str, Any]


@dataclass(frozen=True)
class Approve:
    """Perform the Write Action, with `edits` replacing some of its fields."""

    edits: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Reject:
    """Don't perform the Write Action; the note goes back to the Supervisor to revise."""

    note: str = ""


Decision = Approve | Reject
TurnResult = str | Confirmation

NOTHING_PENDING = "There is nothing waiting for your Confirmation."
CANCELLED_BY_NEW_MESSAGE = (
    "Cancelled: the user sent a new message instead of confirming. Nothing was done."
)
ONE_AT_A_TIME = "Not performed: propose one Write Action at a time, then this one again."


# A step is a model call plus its tool calls: 2 graph steps each, plus the final answer.
RECURSION_LIMIT = 2 * MAX_STEPS_PER_TURN + 1

# Kept in the Thread's checkpoints, so a Turn resumed after a Confirmation stays the
# same Turn: same ledger attribution and token ceiling, and only what's left of its steps.
_TURN_ID = "perry_turn_id"
_TURN_STARTED_AT_STEP = "perry_turn_started_at_step"
_TURN_RESUMES = "perry_turn_resumes"
_TURN_TECHNIQUE = "perry_turn_technique"


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _graph_step(state: StateSnapshot) -> int:
    # -2 is LangGraph's step before a Thread's first checkpoint.
    return (state.metadata or {}).get("step", -2)


@dataclass(frozen=True)
class _TurnRun:
    """One run of a Turn: the graph, its config, the Thread as it was before this run,
    and the user's message (None when resuming after a Confirmation)."""

    graph: Any
    config: dict
    before: StateSnapshot
    request: HumanMessage | None

    @classmethod
    def start(
        cls,
        graph,
        user: User,
        thread_id: str,
        before: StateSnapshot,
        request: HumanMessage | None,
        turn_id: str | None = None,
        technique: str | None = None,
    ) -> "_TurnRun":
        """A new Turn for `user`'s `request` (`turn_id`, or a new id; `technique` forced,
        if given), or with no `request` the Turn `before` is paused in."""
        paused = (before.metadata or {}) if request is None else {}
        turn_id = paused.get(_TURN_ID) or turn_id or usage.new_turn_id()
        technique = paused.get(_TURN_TECHNIQUE, technique)
        started_at = paused.get(_TURN_STARTED_AT_STEP, _graph_step(before))
        # Each resume runs the paused graph step again, so it isn't counted twice.
        resumes = paused.get(_TURN_RESUMES, -1) + 1
        steps_used = max(0, _graph_step(before) - started_at - resumes)
        config = {
            "configurable": {
                "thread_id": thread_id,
                RETRIEVAL_TECHNIQUE: technique,
                _PROFILE: user.profile,
            },
            "metadata": {
                _TURN_ID: turn_id,
                _TURN_STARTED_AT_STEP: started_at,
                _TURN_RESUMES: resumes,
                _TURN_TECHNIQUE: technique,
            },
            "recursion_limit": RECURSION_LIMIT - steps_used,
        }
        return cls(graph, config, before, request)

    @property
    def thread_id(self) -> str:
        return self.config["configurable"]["thread_id"]

    @property
    def turn_id(self) -> str:
        return self.config["metadata"][_TURN_ID]


def run_turn(
    user: User,
    thread_id: str,
    message: str,
    images: list[str | Path] = (),
    turn_id: str | None = None,
    technique: str | None = None,
) -> TurnResult:
    """Answer `user`'s `message`, with any attached image files given to the model as
    well. Returns the answer, or the Confirmation the Turn is waiting for.

    `turn_id` continues a Turn whose model calls began before the message existed
    (e.g. transcribing a voice request); by default the Turn is new. `technique` (the
    per-request override) names the Retrieval Technique the Knowledge Agent must use
    if the Supervisor asks it; by default the Knowledge Agent picks."""
    config = _config(thread_id)
    graph = supervisor(user)
    cancelled = _cancel_pending(graph, config)
    if refused := usage.daily_cap_refusal():
        outcome = refused
    else:
        request = user_message(message, images)
        before = graph.get_state(config)
        run = _TurnRun.start(graph, user, thread_id, before, request, turn_id, technique)
        outcome = _run(run, {"messages": [request]})
    if cancelled and isinstance(outcome, str):
        return f"{cancelled}\n\n{outcome}"
    return outcome


def resume_turn(user: User, thread_id: str, decision: Decision) -> TurnResult:
    """Continue the Turn waiting on a Confirmation with `user`'s decision.
    Returns the answer, or the next Confirmation."""
    graph = supervisor(user)
    before = graph.get_state(_config(thread_id))
    if not before.interrupts:
        return NOTHING_PENDING
    requests = before.interrupts[0].value["action_requests"]
    rest = [{"type": "reject", "message": ONE_AT_A_TIME}] * (len(requests) - 1)
    resume = Command(resume={"decisions": [_decision(decision, requests[0]), *rest]})
    return _run(_TurnRun.start(graph, user, thread_id, before, None), resume)


def pending_confirmation(thread_id: str) -> Confirmation | None:
    """The Confirmation the Thread is waiting for, if any."""
    interrupts = supervisor().get_state(_config(thread_id)).interrupts
    return _confirmation(interrupts[0].value) if interrupts else None


def _ceiling_reply(ceiling: usage.TurnTokenCeilingReached) -> str:
    return f"I stopped because this Turn reached its limit of {ceiling.ceiling:,} tokens."


# Deliberate stops, and what the user is told when one ends a Turn early.
_STOP_REPLIES: dict[type[BaseException], Callable[[Any], str]] = {
    usage.TurnTokenCeilingReached: _ceiling_reply,
    GraphRecursionError: lambda _: (
        f"I stopped after {MAX_STEPS_PER_TURN} steps without finishing. "
        "Try splitting the request into smaller ones."
    ),
    # The model clients already retried with backoff (and fell back to the paid model,
    # if there is one) before giving up.
    **{
        error: lambda _: (
            "The model's rate limit or quota was reached and retrying didn't help. "
            "Try again in a minute, or later today if the daily quota is used up."
        )
        for error in models.RATE_LIMIT_ERRORS
    },
    workspace.ReconnectNeeded: str,
    privacy.Refused: str,
}
_DELIBERATE_STOPS = tuple(_STOP_REPLIES)


def _run(run: _TurnRun, run_input) -> TurnResult:
    """Run (or, with no `request`, resume) a Turn and clean up the Thread if it stops early."""
    try:
        with usage.turn(run.thread_id, run.turn_id):
            result = run.graph.invoke(run_input, run.config)
    except _DELIBERATE_STOPS as stopped:
        if isinstance(stopped, usage.TurnTokenCeilingReached) and run.request is not None:
            return _stop_at_ceiling(run, stopped)
        reply = next(say for kind, say in _STOP_REPLIES.items() if isinstance(stopped, kind))
        return _stop(run, reply(stopped))
    except BaseException as error:
        _stop(run, f"Something went wrong: {type(error).__name__}.")
        raise
    if interrupts := result.get("__interrupt__"):
        return _confirmation(interrupts[0].value)
    return _final_answer(result["messages"])


def _stop(run: _TurnRun, reply: str) -> str:
    """End a Turn that stopped early: a resumed one keeps `reply` as its answer, a new one
    is forgotten so a retry doesn't repeat it."""
    if run.request is None:
        _abandon_confirmation(run, reply)
    else:
        _forget_turn(run)
    return reply


def agent_results(messages: list) -> list[ToolMessage]:
    """What each Agent the Supervisor consulted in the latest Turn returned, in order:
    the tool messages since the last user message."""
    last_user = max((i for i, m in enumerate(messages) if m.type == "human"), default=-1)
    return [m for m in messages[last_user + 1 :] if m.type == "tool"]


def agents_consulted(thread_id: str) -> list[str]:
    """The Agent tools the Supervisor called during the Thread's latest Turn, in order."""
    messages = supervisor().get_state(_config(thread_id)).values.get("messages", [])
    return [m.name for m in agent_results(messages)]


def _final_answer(messages: list) -> str:
    """The Turn's final answer, keeping every chart an Agent made and every Citation the
    Knowledge Agent gave during the Turn, then saying which Retrieval Techniques found them."""
    results = agent_results(messages)
    answer = charts.keep(messages[-1].text, [m.text for m in results])
    knowledge_results = [m for m in results if m.name == ask_knowledge_agent.name]
    answer = knowledge.citations_at_end(answer, [m.text for m in knowledge_results])
    found_with = [f for m in knowledge_results for f in (m.artifact or {}).get("found_with", [])]
    return knowledge.with_found_with(answer, found_with)


def _confirmation(request: dict) -> Confirmation:
    action = request["action_requests"][0]
    return Confirmation(action["name"], action.get("description", ""), _fields(action))


def _fields(action: dict) -> dict[str, Any]:
    """Every argument the Write Action takes, so optional ones can be edited too."""
    write = next(t for t in task.WRITE_ACTIONS if t.name == action["name"])
    return {
        name: action["args"].get(name, schema.get("default", ""))
        for name, schema in write.args.items()
    }


def _decision(decision: Decision, action: dict) -> dict:
    if isinstance(decision, Reject):
        return {"type": "reject", "message": decision.note} if decision.note else {"type": "reject"}
    fields = _fields(action)
    unknown = set(decision.edits) - set(fields)
    if unknown:
        raise ValueError(f"{action['name']} has no field(s) {sorted(unknown)}")
    edited = {**fields, **decision.edits}
    if edited == fields:
        return {"type": "approve"}
    return {"type": "edit", "edited_action": {"name": action["name"], "args": edited}}


def _cancel_pending(graph, config: dict) -> str | None:
    """Cancel a Confirmation the Thread is waiting for, saying which; None if there is none."""
    interrupts = graph.get_state(config).interrupts
    if not interrupts:
        return None
    summary = _confirmation(interrupts[0].value).summary.splitlines()
    _close_open_tool_calls(graph, config, CANCELLED_BY_NEW_MESSAGE)
    what = f" ({summary[0]})" if summary else ""
    return f"I cancelled the Write Action that was waiting for your Confirmation{what}. Nothing was done."


def _close_open_tool_calls(graph, config: dict, text: str) -> None:
    """Answer the Supervisor's unanswered tool calls with `text`, which ends the paused run.

    A pending Confirmation leaves the Agent call that raised it without a result, and
    the model refuses a history with a tool call that has no result.
    """
    messages = graph.get_state(config).values.get("messages", [])
    answered = {m.tool_call_id for m in messages if m.type == "tool"}
    last_ai = next((m for m in reversed(messages) if m.type == "ai"), None)
    open_calls = [c for c in (last_ai.tool_calls if last_ai else []) if c["id"] not in answered]
    if open_calls:
        closing = [ToolMessage(text, tool_call_id=c["id"], name=c["name"]) for c in open_calls]
        graph.update_state(config, {"messages": closing}, as_node="tools")


def _abandon_confirmation(run: _TurnRun, reply: str) -> None:
    """End a resumed Turn that stopped early, keeping `reply` in the Thread as its answer."""
    _close_open_tool_calls(run.graph, run.config, f"Stopped: {reply}")
    run.graph.update_state(run.config, {"messages": [AIMessage(reply)]}, as_node="model")


def _forget_turn(run: _TurnRun) -> None:
    """Put the Thread back as it was before an unfinished Turn, so a retry doesn't repeat it."""
    kept = {m.id for m in run.before.values.get("messages", [])}
    if not kept:
        _checkpointer().delete_thread(run.thread_id)
        return
    now = run.graph.get_state(run.config).values.get("messages", [])
    added = [m for m in now if m.id not in kept]
    if added:
        # As if the model node had run: the last kept message is a final answer, so the graph ends.
        run.graph.update_state(
            run.config, {"messages": [RemoveMessage(id=m.id) for m in added]}, as_node="model"
        )


def _stop_at_ceiling(run: _TurnRun, ceiling: usage.TurnTokenCeilingReached) -> str:
    """End the Turn with the Agents' findings so far, kept in the Thread like an answer."""
    kept = {m.id for m in run.before.values.get("messages", [])}
    findings = [
        m.text
        for m in run.graph.get_state(run.config).values.get("messages", [])
        if m.id not in kept and m.type == "tool" and m.text
    ]
    reply = _ceiling_reply(ceiling)
    if findings:
        reply += " Here is what I found so far:\n\n" + "\n\n".join(findings)
    else:
        reply += " I hadn't found anything yet. Try a narrower question."
    _forget_turn(run)
    run.graph.update_state(
        run.config, {"messages": [run.request, AIMessage(reply)]}, as_node="model"
    )
    return reply


def thread_history(thread_id: str) -> list[dict]:
    """The Thread as user/assistant messages, without the Agent calls in between."""
    state = supervisor().get_state(_config(thread_id))
    history, turn = [], []
    for msg in state.values.get("messages", []):
        turn.append(msg)
        if msg.type == "human":
            history.append({"role": "user", "content": msg.text})
            turn = [msg]
        elif msg.type == "ai" and not msg.tool_calls and msg.text:
            history.append({"role": "assistant", "content": _final_answer(turn)})
    return history
