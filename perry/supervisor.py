"""The Supervisor: plans each Turn, delegates to Agents, and composes the answer.

Turn interface: `run_turn` answers a message with either the final answer (a str) or
a pending `Confirmation` for one Write Action. `resume_turn` continues that Turn with
the user's decision (`Approve`, with optional edits, or `Reject` with a note), and
again returns an answer or the next Confirmation. A new message on a Thread with a
pending Confirmation cancels it. Confirmations are LangGraph interrupts raised by the
Task Agent's Confirmation gate, kept in the Thread's checkpoint.
"""

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command, StateSnapshot

from perry import charts, db, models, privacy, retries, usage, workspace
from perry.agents import analysis, knowledge, task
from perry.images import user_message

# Each step is one Supervisor decision (call an Agent, or answer).
MAX_STEPS_PER_TURN = 6

SYSTEM_PROMPT = """You are Perry, a personal assistant. You coordinate specialist Agents \
and answer the user.

Agents available as tools:
- ask_knowledge_agent: anything that could be answered from the user's own Documents \
(notes, reports, PDFs they uploaded).
- ask_task_agent: the user's calendar and email: what's on, what someone wrote, and \
sending, replying, scheduling, moving or deleting (the user confirms each of those).
- ask_analysis_agent: questions about the user's Datasets (CSV or spreadsheet data), \
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
- To act on a request that needs other information first (e.g. email a summary of some \
notes), get that information, then give the Task Agent the complete content to send.
- If the user rejected a Write Action with a note, ask the Task Agent again with a \
request revised to follow the note."""


@tool
def ask_knowledge_agent(question: str) -> str:
    """Ask the Knowledge Agent a self-contained question about the user's Documents.
    Returns findings with citations."""
    return knowledge.ask(question)


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


@cache
def _checkpointer() -> PostgresSaver:
    checkpointer = PostgresSaver(db.pool())
    checkpointer.setup()
    return checkpointer


def supervisor():
    # Built per call (cheap) so it always uses the current model provider.
    return create_agent(
        models.chat_model(),
        tools=[
            ask_knowledge_agent,
            ask_task_agent,
            ask_analysis_agent,
        ],
        middleware=[
            retries.RetryFailedSteps(),
        ],
        system_prompt=SYSTEM_PROMPT,
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
        cls, graph, thread_id: str, before: StateSnapshot, request: HumanMessage | None
    ) -> "_TurnRun":
        """A new Turn for `request`, or with no `request` the Turn `before` is paused in."""
        paused = (before.metadata or {}) if request is None else {}
        turn_id = paused.get(_TURN_ID) or usage.new_turn_id()
        started_at = paused.get(_TURN_STARTED_AT_STEP, _graph_step(before))
        # Each resume runs the paused graph step again, so it isn't counted twice.
        resumes = paused.get(_TURN_RESUMES, -1) + 1
        steps_used = max(0, _graph_step(before) - started_at - resumes)
        config = {
            **_config(thread_id),
            "metadata": {
                _TURN_ID: turn_id,
                _TURN_STARTED_AT_STEP: started_at,
                _TURN_RESUMES: resumes,
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
    thread_id: str, message: str, images: list[str | Path] = ()
) -> TurnResult:
    """Answer `message`, with any attached image files given to the model as well.
    Returns the answer, or the Confirmation the Turn is waiting for."""
    config = _config(thread_id)
    graph = supervisor()
    cancelled = _cancel_pending(graph, config)
    if refused := usage.daily_cap_refusal():
        outcome = refused
    else:
        request = user_message(message, images)
        run = _TurnRun.start(graph, thread_id, graph.get_state(config), request)
        outcome = _run(run, {"messages": [request]})
    if cancelled and isinstance(outcome, str):
        return f"{cancelled}\n\n{outcome}"
    return outcome


def resume_turn(thread_id: str, decision: Decision) -> TurnResult:
    """Continue the Turn waiting on a Confirmation with the user's decision.
    Returns the answer, or the next Confirmation."""
    graph = supervisor()
    before = graph.get_state(_config(thread_id))
    if not before.interrupts:
        return NOTHING_PENDING
    requests = before.interrupts[0].value["action_requests"]
    rest = [{"type": "reject", "message": ONE_AT_A_TIME}] * (len(requests) - 1)
    resume = Command(resume={"decisions": [_decision(decision, requests[0]), *rest]})
    return _run(_TurnRun.start(graph, thread_id, before, None), resume)


def pending_confirmation(thread_id: str) -> Confirmation | None:
    """The Confirmation the Thread is waiting for, if any."""
    interrupts = supervisor().get_state(_config(thread_id)).interrupts
    return _confirmation(interrupts[0].value) if interrupts else None


def _run(run: _TurnRun, run_input) -> TurnResult:
    """Run (or, with no `request`, resume) a Turn and clean up the Thread if it stops early."""

    def stop(reply: str) -> str:
        if run.request is None:
            _abandon_confirmation(run, reply)
        else:
            _forget_turn(run)
        return reply

    try:
        with usage.turn(run.thread_id, run.turn_id):
            result = run.graph.invoke(run_input, run.config)
    except usage.TurnTokenCeilingReached as ceiling:
        if run.request is None:
            return stop(f"I stopped because this Turn reached its limit of {ceiling.ceiling:,} tokens.")
        return _stop_at_ceiling(run, ceiling)
    except GraphRecursionError:
        return stop(
            f"I stopped after {MAX_STEPS_PER_TURN} steps without finishing. "
            "Try splitting the request into smaller ones."
        )
    except models.RateLimitError:
        # The model client already retried with backoff before giving up.
        return stop(
            "The model's rate limit was reached and retrying didn't help. Try again in a minute."
        )
    except (workspace.ReconnectNeeded, privacy.Refused) as error:
        return stop(str(error))
    except BaseException as error:
        stop(f"Something went wrong: {type(error).__name__}.")
        raise
    if interrupts := result.get("__interrupt__"):
        return _confirmation(interrupts[0].value)
    return _answer_with_charts(result["messages"])


def _answer_with_charts(messages: list) -> str:
    """The Turn's final answer, keeping every chart an Agent made during the Turn."""
    last_user = max((i for i, m in enumerate(messages) if m.type == "human"), default=-1)
    tool_results = [m.text for m in messages[last_user + 1 :] if m.type == "tool"]
    return charts.keep(messages[-1].text, tool_results)


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
    reply = f"I stopped because this Turn reached its limit of {ceiling.ceiling:,} tokens."
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
    state = supervisor().get_state({"configurable": {"thread_id": thread_id}})
    history, turn = [], []
    for msg in state.values.get("messages", []):
        turn.append(msg)
        if msg.type == "human":
            history.append({"role": "user", "content": msg.text})
            turn = [msg]
        elif msg.type == "ai" and not msg.tool_calls and msg.text:
            history.append({"role": "assistant", "content": _answer_with_charts(turn)})
    return history
