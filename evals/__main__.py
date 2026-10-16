"""Run the eval set against the real model. On demand only: it costs real requests.

    uv run python -m evals                      # every eval Turn
    uv run python -m evals knowledge-aphids ... # only these Turn ids (or id prefixes)
    uv run python -m evals --list
    uv run python -m evals --paid               # every chat call (and the judge) on the paid
                                                # model, PERRY_BENCHMARK_LLM_MODEL
    uv run python -m evals --tune-margin        # tune PERRY_CASCADE_MARGIN (embeddings only)

Uses its own database (`perry_eval`, override with PERRY_EVAL_DATABASE_URL), loads
`evals/documents/` into it, and runs each Turn on a fresh Thread. Model calls go through
the provider's request pacing, so a full run takes several minutes on the free tier.
Routing and Citation checks decide the exit code; the judge's quality score and the
soft technique check (did the expected Retrieval Technique find the answer?) are only
reported, and logged to LangSmith (project `perry-eval`) when tracing is on.
"""

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext

# Before any perry import: perry.db reads DATABASE_URL when it is imported.
EVAL_URL = os.getenv(
    "PERRY_EVAL_DATABASE_URL", "postgresql://perry:perry@127.0.0.1:5433/perry_eval"
)
os.environ["DATABASE_URL"] = EVAL_URL
# The soft technique check reads which Retrieval Technique ran from the "Found with" line.
os.environ["PERRY_SHOW_FOUND_WITH"] = "true"

import psycopg

from evals.harness import EvalTurn, Result, feedback, run_eval_turn
from evals.turns import DATASETS_DIR, DOCUMENTS_DIR, all_turns

LANGSMITH_PROJECT = "perry-eval"


class EvalThread:
    """One Thread driven like the chat UI: `send()` a message, ask who was `consulted()`."""

    def __init__(self):
        self.thread_id: str | None = None
        self.user = _eval_owner()

    def send(self, message: str) -> str:
        from perry import supervisor, threads

        if self.thread_id is None:
            self.thread_id = threads.create(self.user, message)
        return supervisor.run_turn(self.user, self.thread_id, message)

    def resume(self, decision):
        """Answer the Confirmation the Turn waits for (Write Actions hit the fake Workspace)."""
        from perry import supervisor

        return supervisor.resume_turn(self.user, self.thread_id, decision)

    def consulted(self) -> list[str]:
        """The Agent tools the Supervisor called during the latest Turn."""
        from perry import supervisor

        return supervisor.agents_consulted(self.thread_id) if self.thread_id else []


def _eval_owner():
    """The eval database's Owner, who sends every eval Turn (created on first use)."""
    import secrets

    from evals.workspace import EVAL_TIME_ZONE
    from perry import users

    return users.owner() or users.create_owner(
        "eval-owner", secrets.token_urlsafe(16), "Eval Owner", EVAL_TIME_ZONE
    )


@contextmanager
def environment() -> Iterator[None]:
    """Everything the eval Turns rely on: the eval database, Documents, and fakes."""
    _create_database()
    from perry import db

    db.init_schema()
    _load_documents()
    _summarise_documents()
    _load_datasets()
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
    from perry.ingest import ensure_indexes_current, ingest_file

    files = sorted(DOCUMENTS_DIR.iterdir())
    with db.pool().connection() as conn:
        conn.execute(
            "DELETE FROM documents WHERE NOT (name = ANY(%s))", ([f.name for f in files],)
        )
    ensure_indexes_current()  # drops the deleted Documents from the ColBERT index too
    for file in files:
        print(ingest_file(file).describe())


def _summarise_documents() -> None:
    """Summarise the eval Documents and build their RAPTOR trees, as the app does in the
    background: the Supervisor reads each Document's topic, and the Summary-based
    Retrieval Techniques search the Summaries. Local model only; done ones are kept."""
    from perry import models, summaries

    if not models.summaries_supported():
        print(f"Not summarising the eval Documents. {models.summary_model_hint()}")
        return
    summaries.retry_failed()
    print("Summarising and building RAPTOR trees on the local model (slow)...", flush=True)
    summaries.summarise_pending()


def _load_datasets() -> None:
    """Load `evals/datasets/` for the Analysis Agent Turns (same name replaces)."""
    from perry import datasets

    for file in sorted(DATASETS_DIR.iterdir()):
        for result in datasets.upload(file):
            print(result.describe())


def run(turns: list[EvalTurn]) -> list[Result]:
    results = []
    for number, turn in enumerate(turns, 1):
        print(f"[{number}/{len(turns)}] {turn.id}: {turn.message}", flush=True)
        with _traced(turn) as log:
            result = run_eval_turn(turn, EvalThread())
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
        metadata={
            "turn_id": turn.id,
            "expected_agents": list(turn.agents),
            "expected_technique": turn.technique,
        },
    ) as run_tree:

        def log(result: Result) -> None:
            run_tree.end(
                outputs={
                    "reply": result.reply,
                    "consulted": result.consulted,
                    "techniques": result.techniques,
                }
            )
            try:
                client = langsmith.Client()
                for key, (score, comment) in feedback(result).items():
                    client.create_feedback(run_tree.id, key, score=score, comment=comment)
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
    if not result.confirmations_ok:
        expected = list(result.turn.write_actions) or "none"
        problems.append(f"Confirmations {result.confirmations or 'none'}, expected {expected}")
    quality = "n/a" if result.quality is None else f"{result.quality:.2f}"
    verdict = "FAIL" if problems else "pass"
    notes = []
    if result.technique_ok is False:  # soft: noted, never a failure
        used = result.techniques or "no technique"
        notes.append(f"(found with {used}, expected {result.turn.technique})")
    parts = [verdict, f"quality {quality}", *problems, *notes, result.judge_comment]
    return "  ".join(parts).rstrip()


def main(args: list[str]) -> int:
    # Replies and judge comments can hold characters (e.g. "≈") that a Windows console's
    # code page can't print; a mangled character beats a run that dies mid-way.
    sys.stdout.reconfigure(errors="replace")
    turns = all_turns()
    if "--list" in args:
        for turn in turns:
            print(f"{turn.id}: {turn.message}")
        return 0
    if "--tune-margin" in args:
        return tune_margin(turns)
    paid = "--paid" in args
    args = [a for a in args if a != "--paid"]
    if args:
        turns = [t for t in turns if any(t.id.startswith(a) for a in args)]
        if not turns:
            sys.exit(f"No eval Turns match {args}. See `python -m evals --list`.")

    with _chat_model(paid), environment():
        results = run(turns)

    failed = [r for r in results if not r.passed]
    checked = [r for r in results if r.technique_ok is not None]
    matched = sum(bool(r.technique_ok) for r in checked)
    scores = [r.quality for r in results if r.quality is not None]
    average = f"{sum(scores) / len(scores):.2f}" if scores else "n/a"
    print(f"\n{len(results) - len(failed)}/{len(results)} passed; average quality {average}")
    if checked:
        print(f"Expected Retrieval Technique used in {matched}/{len(checked)} (not gating)")
    for result in failed:
        print(f"  FAIL {result.turn.id}")
    return 1 if failed else 0


def _chat_model(paid: bool):
    """With `paid`, every chat call in the run (the judge's too) goes to the paid model, so
    the run isn't held to the free tier's daily requests and its scores come from one model."""
    from perry import models

    if not paid:
        return nullcontext()
    if not models.paid_chat_available():
        sys.exit("Not started: --paid needs OPENAI_API_KEY.")
    print(f"Chat model: {models.paid_chat_model()} (paid)", flush=True)
    return models.paid_chat_only()


def tune_margin(turns: list[EvalTurn]) -> int:
    """Print the cascade margin tuned on the eval Turns' expected techniques. Embedding
    calls only (recorded in the eval database's Usage Ledger); no Documents needed."""
    from evals.tuning import report, tune_cascade_margin
    from perry import db, techniques

    _create_database()
    db.init_schema()
    print(report(tune_cascade_margin(turns, list(techniques.TECHNIQUES))))
    print("Record it as DEFAULT_CASCADE_MARGIN in perry/picking.py and in the README.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
