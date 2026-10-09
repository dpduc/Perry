"""Store Documents as embedded Chunks. Also a CLI for bulk loads:

    uv run python -m perry.ingest path/to/folder-or-file [--yes]

It shows the estimated tokens and cost and asks before ingesting. Ingestion counts
toward the daily cap and is refused, before any model call, if it would exceed it.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from pgvector import Vector

from perry import db, models, raptor, summaries, usage
from perry.documents import (
    Chunk,
    ParsedDocument,
    content_hash,
    find_files,
    page_chunks,
    parse,
    render_page,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    name: str
    status: str  # "ingested" | "unchanged" | "failed"
    chunks: int = 0
    # Pages with no text whose transcription failed, so nothing on them is searchable.
    untranscribed_pages: tuple[int, ...] = ()
    error: str = ""

    def describe(self) -> str:
        if self.status == "unchanged":
            return f"{self.name}: unchanged, skipped"
        if self.status == "failed":
            return f"{self.name}: failed ({self.error})"
        message = f"{self.name}: {self.chunks} chunks"
        if self.untranscribed_pages:
            pages = ", ".join(map(str, self.untranscribed_pages))
            message += f" (couldn't transcribe pages with no text: {pages})"
        return message


@dataclass(frozen=True)
class EstimatedCall:
    """Model calls that ingestion expects to make, priced as the usage ledger will price them."""

    kind: str  # as recorded in the usage ledger, e.g. "embedding"
    model: str
    input_tokens: int
    output_tokens: int = 0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        return usage.cost(self.model, self.input_tokens, self.output_tokens)


@dataclass(frozen=True)
class IngestionEstimate:
    """What ingesting some files would cost, worked out without calling any model."""

    files: int
    unchanged: int
    calls: tuple[EstimatedCall, ...]
    # Calls made after the upload returns (Summaries on the local model, priced at zero).
    background: tuple[EstimatedCall, ...] = ()

    @property
    def tokens(self) -> int:
        """Tokens the upload itself uses."""
        return sum(call.tokens for call in self.calls)

    @property
    def cost_usd(self) -> float:
        """What the upload and the background work after it cost."""
        return sum(call.cost_usd for call in (*self.calls, *self.background))

    def describe(self) -> str:
        files = "1 file" if self.files == 1 else f"{self.files} files"
        message = f"{files}: about {self.tokens:,} tokens, ${self.cost_usd:.4f}"
        if self.unchanged:
            message += f" ({self.unchanged} unchanged, skipped)"
        return message + "."


def estimate_file(path: Path) -> list[EstimatedCall] | None:
    """The model calls ingesting `path` would make, or None if it is unchanged and skipped.

    Every ingestion step that calls a model adds its calls here, so the estimate
    and the budget check cover it.
    """
    if _unchanged(path):
        return None
    parsed = parse(path)
    calls: list[EstimatedCall] = []
    embedding_tokens = models.estimate_tokens([chunk.text for chunk in parsed.chunks])
    if models.vision_supported():
        # One call per page without text; what it reads off the page is embedded too.
        for _ in parsed.pages_without_text:
            calls.append(
                EstimatedCall(
                    "transcription",
                    models.vision_model(),
                    models.PAGE_TRANSCRIPTION_INPUT_TOKENS,
                    models.PAGE_TRANSCRIPTION_OUTPUT_TOKENS,
                )
            )
            embedding_tokens += models.PAGE_TRANSCRIPTION_OUTPUT_TOKENS
    if embedding_tokens:
        calls.append(EstimatedCall("embedding", models.embedding_model(), embedding_tokens))
    return calls


def estimate_background(path: Path) -> list[EstimatedCall]:
    """The model calls made in the background after ingesting `path`: its Summary on
    the local model, then embedding the Summary; its RAPTOR tree's node Summaries on the
    local model, then embedding them."""
    if not models.summaries_supported():
        return []
    chunks = [chunk.text for chunk in parse(path).chunks]
    text = summaries.input_text(chunks)
    tree_in, tree_out, _ = raptor.estimate([models.estimate_tokens([c]) for c in chunks])
    return [
        EstimatedCall(
            "summary",
            models.summary_model_name(),
            summaries.PROMPT_TOKENS + models.estimate_tokens([text]),
            summaries.OUTPUT_TOKENS,
        ),
        EstimatedCall("embedding", models.embedding_model(), summaries.OUTPUT_TOKENS),
        EstimatedCall("raptor", models.summary_model_name(), tree_in, tree_out),
        EstimatedCall("embedding", models.embedding_model(), tree_out),
    ]


def estimate_ingestion(paths: list[Path]) -> IngestionEstimate:
    """What `ingest_files(paths)` would cost. Makes no model calls."""
    calls: list[EstimatedCall] = []
    background: list[EstimatedCall] = []
    unchanged = 0
    for path in paths:
        try:
            file_calls = estimate_file(path)
            if file_calls is not None:
                background.extend(estimate_background(path))
        except Exception:  # noqa: BLE001 - it will fail at ingestion too, before any model call
            file_calls = []
        if file_calls is None:
            unchanged += 1
        else:
            calls.extend(file_calls)
    return IngestionEstimate(len(paths), unchanged, tuple(calls), tuple(background))


def _unchanged(path: Path) -> bool:
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT content_hash FROM documents WHERE name = %s", (path.name,)
        ).fetchone()
    return row is not None and row["content_hash"] == content_hash(path)


def ingest_file(path: Path, *, embed=models.embed_documents) -> IngestResult:
    """Ingest one Document. A Document with the same name is replaced; identical content is skipped.

    This doesn't check the daily cap: bulk ingestion goes through `ingest_files`.
    """
    if _unchanged(path):
        return IngestResult(path.name, "unchanged")

    parsed = parse(path)
    # Call the models before touching the database, so a failed call leaves the old version intact.
    transcribed, untranscribed = _transcribe_pages(path, parsed)
    chunks = sorted([*parsed.chunks, *transcribed], key=lambda chunk: chunk.page)
    embeddings = embed([c.text for c in chunks]) if chunks else []

    with db.pool().connection() as conn, conn.transaction():
        conn.execute("DELETE FROM documents WHERE name = %s", (parsed.name,))
        document_id = conn.execute(
            "INSERT INTO documents (name, content_hash) VALUES (%s, %s) RETURNING id",
            (parsed.name, parsed.content_hash),
        ).fetchone()["id"]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (document_id, page, text, embedding) VALUES (%s, %s, %s, %s)",
                [
                    (document_id, chunk.page, chunk.text, Vector(embedding))
                    for chunk, embedding in zip(chunks, embeddings, strict=True)
                ],
            )

    return IngestResult(parsed.name, "ingested", len(chunks), untranscribed)


def _transcribe_pages(path: Path, parsed: ParsedDocument) -> tuple[list[Chunk], tuple[int, ...]]:
    """Chunks of the pages without text (e.g. scanned pages), read by the vision model,
    and the pages whose transcription failed."""
    chunks: list[Chunk] = []
    failed: list[int] = []
    for page in parsed.pages_without_text:
        try:
            text = models.transcribe_page(render_page(path, page), "image/png")
        except usage.TurnTokenCeilingReached:
            raise
        except Exception:  # the rest of the Document is still worth storing
            log.warning("Couldn't transcribe page %d of %s", page, path.name, exc_info=True)
            failed.append(page)
            continue
        chunks.extend(page_chunks(page, text))
    return chunks, tuple(failed)


class OverBudget(Exception):
    """Ingestion would cost more than what is left of today's budget."""

    def __init__(self, estimate: IngestionEstimate, remaining_usd: float, cap_usd: float):
        super().__init__(
            f"Ingesting these files would cost about ${estimate.cost_usd:.4f}, "
            f"but only ${remaining_usd:.4f} of today's ${cap_usd:.2f} budget is left."
        )
        self.estimate = estimate


def check_budget(estimate: IngestionEstimate) -> None:
    """Raise OverBudget if `estimate` costs more than what is left of today's budget."""
    remaining = usage.remaining_today()
    if estimate.cost_usd > remaining:
        raise OverBudget(estimate, remaining, usage.daily_cap_usd())


def ingest_files(paths: list[Path]) -> list[IngestResult]:
    """Ingest several Documents, after checking they fit in today's remaining budget.

    Raises OverBudget, before any model call, if the estimate exceeds what is left.
    Show the user `estimate_ingestion(paths)` and get their OK before calling this.
    A file that fails is reported in its result, and the rest are still ingested.
    """
    check_budget(estimate_ingestion(paths))
    results = []
    for path in paths:
        try:
            results.append(ingest_file(path))
        except Exception as error:  # noqa: BLE001 - report per file, keep going with the rest
            results.append(IngestResult(path.name, "failed", error=str(error)))
    return results


def main(argv: list[str] | None = None, *, ask=input) -> None:
    parser = argparse.ArgumentParser(prog="python -m perry.ingest", description=__doc__)
    parser.add_argument("target", help="a Document file, or a folder to ingest recursively")
    parser.add_argument("--yes", action="store_true", help="don't ask before ingesting")
    args = parser.parse_args(argv)

    db.init_schema()
    path = Path(args.target)
    files = [path] if path.is_file() else find_files(path)
    if not files:
        sys.exit(f"No supported files found in {path}")

    try:
        estimate = estimate_ingestion(files)
        check_budget(estimate)
        if not args.yes:
            question = f"Ingest {estimate.describe().removesuffix('.')}? [y/N] "
            if ask(question).strip().lower() not in {"y", "yes"}:
                print("Nothing ingested.")
                return
        results = ingest_files(files)
    except OverBudget as refused:
        sys.exit(str(refused))
    for result in results:
        print(result.describe())


if __name__ == "__main__":
    main()
