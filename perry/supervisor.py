"""The Supervisor: plans each Turn, delegates to Agents, and composes the answer."""

from functools import cache

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.errors import GraphRecursionError

from perry import db, models
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
def supervisor():
    checkpointer = PostgresSaver(db.pool())
    checkpointer.setup()
    return create_agent(
        models.chat_model(),
        tools=[ask_knowledge_agent],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        name="supervisor",
    )


def run_turn(thread_id: str, message: str) -> str:
    config = {
        "configurable": {"thread_id": thread_id},
        # A step is a model call plus its tool calls: 2 graph steps each, plus the final answer.
        "recursion_limit": 2 * MAX_STEPS_PER_TURN + 1,
    }
    try:
        result = supervisor().invoke(
            {"messages": [{"role": "user", "content": message}]}, config
        )
    except GraphRecursionError:
        return (
            f"I stopped after {MAX_STEPS_PER_TURN} steps without finishing. "
            "Try splitting the request into smaller ones."
        )
    except models.RateLimitError:
        # The model client already retried with backoff before giving up.
        return "The model's rate limit was reached and retrying didn't help. Try again in a minute."
    return result["messages"][-1].text


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
