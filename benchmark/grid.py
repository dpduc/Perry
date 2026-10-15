"""The technique grid: every Retrieval Technique forced, and every picking mode, asked the
same benchmark questions, each answer scored for correctness, Citation accuracy and cost.

A *setup* is one grid row group: a technique's name (that technique forced through the
per-request override) or a picking mode (logical, semantic, cascade). Always-hybrid is
hybrid search forced, so it is the `hybrid_search` setup, not a separate one. Any setup
can also run with its Chunks reranked, named with a "+rerank" suffix (e.g.
`cascade+rerank`); the setup without it runs with reranking off.

Perry modules are imported inside functions: the benchmark points DATABASE_URL at its own
database before anything from Perry is imported (see `benchmark.run`).
"""

import contextlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from benchmark.scoring import QUESTION_TYPES, Question, score


@dataclass(frozen=True)
class Answer:
    """One answer from one setup, with what it cost (from the Usage Ledger)."""

    text: str
    found_with: tuple[str, ...]
    cost_usd: float
    tokens: int
    model: str


Ask = Callable[[str, str], Answer]  # (setup, question) -> Answer

# Always-hybrid (Perry before Retrieval Techniques: the Knowledge Agent with hybrid search
# as its one tool) is exactly hybrid search forced, so it shares that setup and its cost.
ALWAYS_HYBRID = "hybrid_search"
_ALIASES = {"always-hybrid": ALWAYS_HYBRID, "always_hybrid": ALWAYS_HYBRID}
# The suffix that runs a setup with its Chunks reranked.
RERANK = "+rerank"
# Ledger kinds that aren't the chat model, left out of a row's "model" ("pick" is semantic
# picking's embeddings).
_NOT_CHAT = ("embedding", "colbert", "pick", "rerank")


def picking_modes() -> tuple[str, ...]:
    """The picking modes, as `perry.picking` names them."""
    from perry import picking

    return picking.MODES


def setups() -> list[str]:
    """Every setup in the grid: each Retrieval Technique forced, then each picking mode."""
    from perry import techniques

    return [*(t.name for t in techniques.TECHNIQUES), *picking_modes()]


def base(setup: str) -> str:
    """The technique or picking mode a setup runs, without its "+rerank"."""
    return setup.removesuffix(RERANK)


def reranked(setup: str) -> bool:
    return setup.endswith(RERANK)


def select(only: list[str] | None) -> list[str]:
    """The setups named in `only` (all of them if None), in grid order, each one's
    reranked run after its plain one. Raises ValueError for a name that is neither a
    technique nor a picking mode, with or without "+rerank"."""
    every = setups()
    if not only:
        return every
    named = {
        _ALIASES.get(base(name), base(name)) + (RERANK if reranked(name) else "")
        for name in only
    }
    unknown = sorted(name for name in named if base(name) not in every)
    if unknown:
        raise ValueError(
            f"Unknown setups {unknown}; choose from {every} or always-hybrid, "
            f"each optionally with {RERANK}"
        )
    return [run for setup in every for run in (setup, setup + RERANK) if run in named]


class NotReady(Exception):
    """The benchmark Documents aren't ready for the setups asked for; nothing was asked."""


def _needs_summaries(setups: list[str]) -> bool:
    """Whether any setup may search Summaries: a summary-based technique, or a picking
    mode (which may pick those once a Document's Summaries are ready)."""
    from perry import techniques

    summary_based = {t.name for t in techniques.TECHNIQUES if t.summary_kind}
    return any(base(s) in summary_based or base(s) in picking_modes() for s in setups)


def _needs_colbert(setups: list[str]) -> bool:
    return any(base(s) == "colbert" or base(s) in picking_modes() for s in setups)


def prepare(setups: list[str], documents: list[Path]) -> None:
    """Make the benchmark Documents ready for `setups`, before any question costs money:
    ingest them (unchanged ones are skipped), bring the ColBERT index in step with
    Postgres, and, if a setup may search Summaries, write every Summary and RAPTOR tree
    on the local model (slow: minutes per Document on a CPU; done once, kept after).

    Raises NotReady if a setup needs ColBERT or the local summary model and it isn't
    there, or if any Document still isn't summarised."""
    from perry import db, models, summaries
    from perry.ingest import ensure_indexes_current, ingest_file

    db.init_schema()
    for path in documents:
        print(ingest_file(path).describe(), flush=True)
    if _needs_colbert(setups) and not models.colbert_supported():
        raise NotReady(
            "ColBERT has no encoder: PERRY_COLBERT_MODEL is empty. Unset it to use "
            f"{models.DEFAULT_COLBERT_MODEL}."
        )
    if any(map(reranked, setups)) and not models.reranking_supported():
        raise NotReady(
            "These setups rerank, but there is no reranker: PERRY_RERANK_MODEL is empty. "
            f"Unset it to use {models.DEFAULT_RERANK_MODEL}."
        )
    ensure_indexes_current()  # e.g. Documents stored before ColBERT or a model change
    if not _needs_summaries(setups):
        return
    if not models.summaries_supported():
        raise NotReady(f"These setups search Summaries. {models.summary_model_hint()}")
    summaries.retry_failed()
    print("Summarising and building RAPTOR trees on the local model (slow)...", flush=True)
    summaries.summarise_pending()
    with db.pool().connection() as conn:
        rows = conn.execute(
            "SELECT name FROM documents WHERE summary_status <> 'ready' "
            "OR raptor_status <> 'ready' ORDER BY name"
        ).fetchall()
    if rows:
        names = ", ".join(row["name"] for row in rows)
        raise NotReady(
            f"Not summarised: {names}. {models.summary_model_hint()} "
            "Rerun to retry; finished Documents are kept."
        )


def ask_perry(setup: str, question: str) -> Answer:
    """Ask Perry's Knowledge Agent `question` in `setup`, as one Turn, and read what the
    Turn cost from the Usage Ledger. A Turn stopped by its token ceiling is an answer too
    (an unanswered one), not a failed run."""
    from perry import models, usage
    from perry.agents import knowledge

    run = base(setup)
    forced, mode = (None, run) if run in picking_modes() else (run, None)
    reranking = contextlib.nullcontext() if reranked(setup) else models.without_reranker()
    with reranking, usage.turn(None, f"benchmark-{setup}") as turn:
        try:
            findings = knowledge.find(question, forced, picking_mode=mode)
            text = knowledge.with_found_with(findings.text, findings.found_with)
            found_with = findings.found_with
        except usage.TurnTokenCeilingReached as stopped:
            text, found_with = f"Stopped: {stopped}", ()
    spent = usage.turn_usage(turn.id)
    chat_models = usage.turn_models(turn.id, excluding_kinds=_NOT_CHAT)
    return Answer(
        text=text,
        found_with=tuple(found_with),
        cost_usd=spent.cost_usd,
        tokens=spent.tokens,
        model=", ".join(chat_models) or models.BENCHMARK_LLM_MODEL,
    )


# What one benchmark Turn is assumed to use, for the estimate shown before a run: the
# Knowledge Agent's prompt with every technique's description, two or three calls, and
# the Chunks they found. Compare with the "tokens" of rows already in grid.jsonl.
TURN_INPUT_TOKENS = 8_000
TURN_OUTPUT_TOKENS = 1_000


def turns_to_ask(questions: list[Question], setups: list[str], path: Path, rerun: bool) -> int:
    """How many (setup, question) Turns `run_grid` would ask: those not already in `path`,
    or all of them with `rerun`."""
    done = set() if rerun else set(load_grid(path))
    return sum((s, q.id) not in done for s in setups for q in questions)


def estimate_usd(turns: int, model: str) -> float:
    """What `turns` benchmark Turns are expected to cost on `model`, from the price table."""
    from perry import usage

    return turns * usage.cost(model, TURN_INPUT_TOKENS, TURN_OUTPUT_TOKENS)


def load_grid(path: Path) -> dict[tuple[str, str], dict]:
    """(setup, question id) -> row. A later row for the same pair replaces an earlier one."""
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return {(row["setup"], row["id"]): row for row in rows}


def run_grid(
    questions: list[Question],
    setups: list[str],
    path: Path,
    ask: Ask,
    *,
    rerun: bool = False,
    today: date | None = None,
) -> None:
    """Ask every question with every setup in `setups`, appending each scored row to
    `path` as soon as it arrives. Rows of other setups are kept.

    By default a pair already in `path` is skipped, so a stopped run (e.g. the quota ran
    out) resumes where it stopped; the first failure stops the run. `rerun` first drops
    the rows of `setups`, so they are asked afresh (e.g. after changing a technique).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_grid(path)
    if rerun:
        done = {key: row for key, row in done.items() if key[0] not in setups}
        lines = [json.dumps(row, ensure_ascii=False) + "\n" for row in done.values()]
        path.write_text("".join(lines), encoding="utf-8")
    for setup in setups:
        for question in questions:
            if (setup, question.id) in done:
                continue
            started = time.monotonic()
            answer = ask(setup, question.question)
            scored = score(question, answer.text)
            row = {
                "setup": setup,
                "id": question.id,
                "type": question.type,
                "question": question.question,
                "answer": answer.text,
                "found_with": list(answer.found_with),
                "correct": scored.correct,
                "fact_recall": scored.fact_recall,
                "citation_recall": scored.citation_recall,
                "citation_precision": scored.citation_precision,
                "cost_usd": answer.cost_usd,
                "tokens": answer.tokens,
                "model": answer.model,
                "date": (today or datetime.now().astimezone().date()).isoformat(),
                "seconds": round(time.monotonic() - started, 1),
            }
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{setup} {question.id}: {row['seconds']}s, ${answer.cost_usd:.4f}", flush=True)


# --- Summary and table ---------------------------------------------------------------


def label(setup: str) -> str:
    if reranked(setup):
        return f"{label(base(setup))} + reranking"
    if setup in picking_modes():
        return f"picking: {setup}"
    if setup == ALWAYS_HYBRID:
        return f"{setup} (= always-hybrid)"
    return setup


@dataclass(frozen=True)
class GridSummary:
    questions: int
    answered: int
    # Means over answered questions; cost_usd is per question.
    correct: float = 0.0
    fact_recall: float = 0.0
    citation_recall: float = 0.0
    citation_precision: float = 0.0
    cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    dates: tuple[str, ...] = ()
    models: tuple[str, ...] = ()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def setups_in(rows: dict[tuple[str, str], dict]) -> list[str]:
    return list(dict.fromkeys(setup for setup, _ in rows))


def summarise_grid(
    questions: list[Question], rows: dict[tuple[str, str], dict]
) -> dict[tuple[str, str], GridSummary]:
    """(setup, question type) -> scores; the type "all" covers every question."""
    summary = {}
    for setup in setups_in(rows):
        for qtype in (*QUESTION_TYPES, "all"):
            group = [q for q in questions if qtype in ("all", q.type)]
            answered = [rows[(setup, q.id)] for q in group if (setup, q.id) in rows]
            if not answered:
                summary[(setup, qtype)] = GridSummary(len(group), 0)
                continue
            summary[(setup, qtype)] = GridSummary(
                questions=len(group),
                answered=len(answered),
                correct=_mean([float(r["correct"]) for r in answered]),
                fact_recall=_mean([r["fact_recall"] for r in answered]),
                citation_recall=_mean([r["citation_recall"] for r in answered]),
                citation_precision=_mean([r["citation_precision"] for r in answered]),
                cost_usd=_mean([r["cost_usd"] for r in answered]),
                total_cost_usd=sum(r["cost_usd"] for r in answered),
                dates=tuple(sorted({r["date"] for r in answered})),
                models=tuple(sorted({r["model"] for r in answered})),
            )
    return summary


def _pct(value: float) -> str:
    return f"{value:.0%}"


def _dollars(value: float) -> str:
    return f"${value:.4f}"


def render_grid(questions: list[Question], rows: dict[tuple[str, str], dict]) -> str:
    """The grid as Markdown: per setup overall, per question type, and per question."""
    summary = summarise_grid(questions, rows)
    setups = setups_in(rows)
    types = (*QUESTION_TYPES, "all")
    lines = [
        "# Retrieval Technique grid",
        "",
        (
            "Generated by `uv run python -m benchmark.run grid`. Each setup is one Retrieval "
            "Technique forced, or one picking mode. Always-hybrid is hybrid search forced. "
            "Scores are means over answered questions. *Correct*: every expected fact "
            "present. *Citation recall*: share of required source pages cited. *Citation "
            "precision*: share of cited pages that are required or related source pages. "
            "*Cost*: from the Usage Ledger, per question (local model calls are free)."
        ),
        "",
        "## Overall",
        "",
        (
            "| Setup | Answered | Correct | Fact recall | Citation recall | Citation precision "
            "| Cost/question | Total cost | Model | Run date |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for setup in setups:
        s = summary[(setup, "all")]
        lines.append(
            f"| {label(setup)} | {s.answered}/{s.questions} | {_pct(s.correct)} "
            f"| {_pct(s.fact_recall)} | {_pct(s.citation_recall)} "
            f"| {_pct(s.citation_precision)} | {_dollars(s.cost_usd)} "
            f"| {_dollars(s.total_cost_usd)} | {', '.join(s.models)} | {', '.join(s.dates)} |"
        )

    measures = [
        ("Correct", lambda s: _pct(s.correct)),
        ("Citation recall", lambda s: _pct(s.citation_recall)),
        ("Citation precision", lambda s: _pct(s.citation_precision)),
        ("Cost per question", lambda s: _dollars(s.cost_usd)),
    ]
    for title, show in measures:
        lines += [
            "",
            f"## {title} per question type",
            "",
            f"| Setup | {' | '.join(types)} |",
            f"|---|{'---|' * len(types)}",
        ]
        for setup in setups:
            cells = [
                show(summary[(setup, t)]) if summary[(setup, t)].answered else "-" for t in types
            ]
            lines.append(f"| {label(setup)} | {' | '.join(cells)} |")

    lines += [
        "",
        "## Per question",
        "",
        "Fact recall, then `c` when every required page is cited. `-`: not asked yet.",
        "",
        f"| Question | Type | {' | '.join(label(s) for s in setups)} |",
        f"|---|---|{'---|' * len(setups)}",
    ]
    for q in questions:
        cells = []
        for setup in setups:
            row = rows.get((setup, q.id))
            if row is None:
                cells.append("-")
            else:
                cited = " c" if row["citation_recall"] == 1.0 else ""
                cells.append(f"{_pct(row['fact_recall'])}{cited}")
        lines.append(f"| {q.id} | {q.type} | {' | '.join(cells)} |")
    return "\n".join(lines) + "\n"
