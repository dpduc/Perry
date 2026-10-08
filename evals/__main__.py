"""Run the eval set against the real model. On demand only: it costs real requests.

    uv run python -m evals                      # every eval Turn
    uv run python -m evals knowledge-aphids ... # only these Turn ids (or id prefixes)
    uv run python -m evals --list

Uses its own database (`perry_eval`, override with PERRY_EVAL_DATABASE_URL), loads
`evals/documents/` into it, and runs each Turn on a fresh Thread. Model calls go through
the provider's request pacing, so a full run takes several minutes on the free tier.
Routing and Citation checks decide the exit code; the judge's quality score is
only reported, and logged to LangSmith (project `perry-eval`) when tracing is on.
"""

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

# Before any perry import: perry.db reads DATABASE_URL when it is imported.
EVAL_URL = os.getenv(
    "PERRY_EVAL_DATABASE_URL", "postgresql://perry:perry@127.0.0.1:5433/perry_eval"
)
os.environ["DATABASE_URL"] = EVAL_URL

import psycopg

from evals.harness import EvalTurn, Result, run_eval_turn
from evals.turns import DOCUMENTS_DIR, all_turns

LANGSMITH_PROJECT = "perry-eval"


class Session:
    """One Thread driven like the chat UI: `send()` a message, ask who was `consulted()`."""

    def __init__(self):
        self.thread_id: str | None = None

    def send(self, message: str) -> str:
        from perry import supervisor, threads

        if self.thread_id is None:
            self.thread_id = threads.create(message)
        return supervisor.run_turn(self.thread_id, message)

    def consulted(self) -> list[str]:
        """The Agent tools the Supervisor called during the latest Turn."""
        from perry import supervisor

        if self.thread_id is None:
            return []
        config = {"configurable": {"thread_id": self.thread_id}}
        messages = supervisor.supervisor().get_state(config).values.get("messages", [])
        last_user = max((i for i, m in enumerate(messages) if m.type == "human"), default=-1)
        return [m.name for m in messages[last_user + 1 :] if m.type == "tool"]


@contextmanager
def environment() -> Iterator[None]:
    """Everything the eval Turns rely on: the eval database, Documents, and fakes."""
    _create_database()
    from perry import db

    db.init_schema()
    _load_documents()
    # Datasets for Analysis Agent Turns load here too.
    from evals.workspace import fake_workspace
    from perry import workspace

    # Task Agent Turns never reach a real Google account.
    with workspace.use(fake_workspace()):
        yield


def _create_database() -> None:
    name = psycopg.conninfo.conninfo_to_dict(EVAL_URL)["dbname"]
    admin_url = psycopg.conninfo.make_conninfo(EVAL_URL, dbname="postgres")
    try:
        with psycopg.connect(admin_url, autocommit=True, connect_timeout=3) as conn:
            if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
                conn.execute(f'CREATE DATABASE "{name}"')
    except psycopg.OperationalError as error:
        sys.exit(f"Postgres not reachable ({error}). Start it with `docker compose up -d`.")


def _load_documents() -> None:
    """Make the eval database hold exactly `evals/documents/` (unchanged files cost nothing)."""
    from perry import db
    from perry.ingest import ingest_file

    files = sorted(DOCUMENTS_DIR.iterdir())
    with db.pool().connection() as conn:
        conn.execute(
            "DELETE FROM documents WHERE NOT (name = ANY(%s))", ([f.name for f in files],)
        )
    for file in files:
        print(ingest_file(file).describe())


def run(turns: list[EvalTurn]) -> list[Result]:
    results = []
    for number, turn in enumerate(turns, 1):
        print(f"[{number}/{len(turns)}] {turn.id}: {turn.message}", flush=True)
        with _traced(turn) as log:
            result = run_eval_turn(turn, Session())
            log(result)
        print(f"    {_summary(result)}", flush=True)
        results.append(result)
    return results


@contextmanager
def _traced(turn: EvalTurn):
    """Trace the Turn to LangSmith and attach its scores as feedback, if tracing is on.

    LangSmith failures (no key, rejected key) are reported but never stop the eval.
    """
    import langsmith
    from langsmith.utils import tracing_is_enabled

    if not tracing_is_enabled():
        yield lambda result: None
        return

    with langsmith.trace(
        f"eval: {turn.id}",
        inputs={"message": turn.message},
        project_name=LANGSMITH_PROJECT,
        metadata={"turn_id": turn.id, "expected_agents": list(turn.agents)},
    ) as run_tree:

        def log(result: Result) -> None:
            run_tree.end(outputs={"reply": result.reply, "consulted": result.consulted})
            try:
                client = langsmith.Client()
                for key, score in [
                    ("routing", result.routing_ok),
                    ("citations", result.citations_ok),
                    ("quality", result.quality),
                ]:
                    if score is not None:
                        client.create_feedback(
                            run_tree.id,
                            key,
                            score=float(score),
                            comment=result.judge_comment if key == "quality" else None,
                        )
            except Exception as error:  # noqa: BLE001
                print(f"    (LangSmith logging failed: {type(error).__name__}: {error})")

        yield log


def _summary(result: Result) -> str:
    if result.error:
        return f"FAIL  error: {result.error}"
    problems = []
    if not result.routing_ok:
        expected = list(result.turn.agents) or "no Agent"
        problems.append(f"consulted {result.consulted or 'no Agent'}, expected {expected}")
    if result.missing_citations:
        problems.append(f"missing Citations {result.missing_citations}")
    quality = "n/a" if result.quality is None else f"{result.quality:.2f}"
    verdict = "FAIL" if problems else "pass"
    return "  ".join([verdict, f"quality {quality}", *problems, result.judge_comment]).rstrip()


def main(args: list[str]) -> int:
    turns = all_turns()
    if "--list" in args:
        for turn in turns:
            print(f"{turn.id}: {turn.message}")
        return 0
    if args:
        turns = [t for t in turns if any(t.id.startswith(a) for a in args)]
        if not turns:
            sys.exit(f"No eval Turns match {args}. See `python -m evals --list`.")

    with environment():
        results = run(turns)

    failed = [r for r in results if not r.passed]
    scores = [r.quality for r in results if r.quality is not None]
    average = f"{sum(scores) / len(scores):.2f}" if scores else "n/a"
    print(f"\n{len(results) - len(failed)}/{len(results)} passed; average quality {average}")
    for result in failed:
        print(f"  FAIL {result.turn.id}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
