"""Store Documents as embedded Chunks. Also a CLI for bulk loads:

    uv run python -m perry.ingest path/to/folder-or-file
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from pgvector import Vector

from perry import db, models
from perry.documents import content_hash, find_files, parse


@dataclass(frozen=True)
class IngestResult:
    name: str
    status: str  # "ingested" | "unchanged"
    chunks: int = 0
    pages_without_text: tuple[int, ...] = ()

    def describe(self) -> str:
        if self.status == "unchanged":
            return f"{self.name}: unchanged, skipped"
        message = f"{self.name}: {self.chunks} chunks"
        if self.pages_without_text:
            pages = ", ".join(map(str, self.pages_without_text))
            message += f" (skipped pages with no text: {pages})"
        return message


def ingest_file(path: Path, *, embed=models.embed_documents) -> IngestResult:
    """Ingest one Document. A Document with the same name is replaced; identical content is skipped."""
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT content_hash FROM documents WHERE name = %s", (path.name,)
        ).fetchone()
    if row and row["content_hash"] == content_hash(path):
        return IngestResult(path.name, "unchanged")

    parsed = parse(path)
    # Embed before touching the database, so a failed API call leaves the old version intact.
    embeddings = embed([c.text for c in parsed.chunks]) if parsed.chunks else []

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
                    for chunk, embedding in zip(parsed.chunks, embeddings, strict=True)
                ],
            )

    return IngestResult(
        parsed.name, "ingested", len(parsed.chunks), tuple(parsed.pages_without_text)
    )


def main(target: str) -> None:
    db.init_schema()
    path = Path(target)
    files = [path] if path.is_file() else find_files(path)
    if not files:
        sys.exit(f"No supported files found in {path}")
    for file in files:
        print(ingest_file(file).describe())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m perry.ingest <folder-or-file>")
    main(sys.argv[1])
