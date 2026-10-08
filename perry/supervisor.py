"""The Supervisor: plans each Turn, delegates to Agents, and composes the answer."""

from functools import cache

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.tools import tool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.errors import GraphRecursionError

from perry import db, models, usage
from perry.agents import knowledge

# Each step is one Supervisor decision (call an Agent, or answer).
MAX_STEPS_PER_TURN = 6

SYSTEM_PROMPT = """You are Perry, a personal assistant. You coordinate specialist Agents \
and answer the user.

Agents available as tools:
- ask_knowledge_agent: anything that could be answered from the user's own Documents \
(notes, reports, PDFs they uploaded).

Rules:
- If a question might be answered by the user's Documents, ask the Knowledge Agent \
rather than answering from general knowledge.
- Keep every citation in square brackets exactly as the Agent gave it.
- If an Agent reports it couldn't find something, tell the user so. Never invent results.
- Small talk and questions about yourself need no Agent."""


@tool
def ask_knowledge_agent(question: str) -> str:
    """Ask the Knowledge Agent a self-contained question about the user's Documents.
    Returns findings with citations."""
    return knowledge.ask(question)


@cache
def _checkpointer() -> PostgresSaver:
    checkpointer = PostgresSaver(db.pool())
    checkpointer.setup()
    return checkpointer


def supervisor():
    # Built per call (cheap) so it always uses the current model provider.
    return create_agent(
        models.chat_model(),
        tools=[ask_knowledge_agent],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=_checkpointer(),
        name="supervisor",
    )


def run_turn(thread_id: str, message: str) -> str:
    try:
        usage.check_daily_cap()
    except usage.DailyCapReached as refused:
        return str(refused)
    config = {
        "configurable": {"thread_id": thread_id},
        # A step is a model call plus its tool calls: 2 graph steps each, plus the final answer.
        "recursion_limit": 2 * MAX_STEPS_PER_TURN + 1,
    }
    graph = supervisor()
    before = graph.get_state(config)
    try:
        with usage.turn(thread_id):
            result = graph.invoke({"messages": [{"role": "user", "content": message}]}, config)
    except usage.TurnTokenCeilingReached as stop:
        return _stop_at_ceiling(graph, config, before, message, stop)
    except GraphRecursionError:
        _forget_turn(graph, config, before)
        return (
            f"I stopped after {MAX_STEPS_PER_TURN} steps without finishing. "
            "Try splitting the request into smaller ones."
        )
    except models.RateLimitError:
        _forget_turn(graph, config, before)
        # The model client already retried with backoff before giving up.
        return "The model's rate limit was reached and retrying didn't help. Try again in a minute."
    except BaseException:
        _forget_turn(graph, config, before)
        raise
    return result["messages"][-1].text


def _forget_turn(graph, config: dict, before) -> None:
    """Put the Thread back as it was before an unfinished Turn, so a retry doesn't repeat it."""
    kept = {m.id for m in before.values.get("messages", [])}
    if not kept:
        _checkpointer().delete_thread(config["configurable"]["thread_id"])
        return
    added = [m for m in graph.get_state(config).values.get("messages", []) if m.id not in kept]
    if added:
        # As if the model node had run: the last kept message is a final answer, so the graph ends.
        graph.update_state(
            config, {"messages": [RemoveMessage(id=m.id) for m in added]}, as_node="model"
        )


def _stop_at_ceiling(graph, config: dict, before, message: str, stop) -> str:
    """End the Turn with the Agents' findings so far, kept in the Thread like an answer."""
    kept = {m.id for m in before.values.get("messages", [])}
    findings = [
        m.text
        for m in graph.get_state(config).values.get("messages", [])
        if m.id not in kept and m.type == "tool" and m.text
    ]
    reply = f"I stopped because this Turn reached its limit of {stop.ceiling:,} tokens."
    if findings:
        reply += " Here is what I found so far:\n\n" + "\n\n".join(findings)
    else:
        reply += " I hadn't found anything yet. Try a narrower question."
    _forget_turn(graph, config, before)
    graph.update_state(
        config, {"messages": [HumanMessage(message), AIMessage(reply)]}, as_node="model"
    )
    return reply


def thread_history(thread_id: str) -> list[dict]:
    """The Thread as user/assistant messages, without the Agent calls in between."""
    state = supervisor().get_state({"configurable": {"thread_id": thread_id}})
    history = []
    for msg in state.values.get("messages", []):
        if msg.type == "human":
            history.append({"role": "user", "content": msg.text})
        elif msg.type == "ai" and not msg.tool_calls and msg.text:
            history.append({"role": "assistant", "content": msg.text})
    return history
