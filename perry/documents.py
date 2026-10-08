"""Turn a Document file into page-numbered Chunks, so every answer can cite a page."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pymupdf

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}

# Resolution scanned pages are rendered at for transcription: readable small print,
# without paying for more image tokens than the text needs.
RENDER_DPI = 150

# Roughly 300-400 tokens: small enough to retrieve precisely, big enough to keep context.
CHUNK_CHARS = 1500
OVERLAP_CHARS = 200


@dataclass(frozen=True)
class Chunk:
    page: int  # 1-based; non-paged files are page 1
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    name: str
    content_hash: str
    chunks: list[Chunk]
    pages_without_text: list[int]  # e.g. scanned pages: render them to transcribe them


def citation(document_name: str, page: int) -> str:
    if document_name.lower().endswith(".pdf"):
        return f"{document_name}#page={page}"
    return document_name


def content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with pymupdf.open(path) as pdf:
            pages = [page.get_text() for page in pdf]
    elif suffix in {".md", ".txt"}:
        pages = [path.read_text(encoding="utf-8")]
    else:
        raise ValueError(f"Unsupported file type: {path.name}")

    chunks: list[Chunk] = []
    without_text: list[int] = []
    for number, text in enumerate(pages, start=1):
        if not text.strip():
            without_text.append(number)
            continue
        chunks.extend(page_chunks(number, text))

    return ParsedDocument(
        name=path.name,
        content_hash=content_hash(path),
        chunks=chunks,
        pages_without_text=without_text,
    )


def page_chunks(page: int, text: str) -> list[Chunk]:
    return [Chunk(page=page, text=piece) for piece in split_text(text)]


def render_page(path: Path, page: int) -> bytes:
    """Page `page` (1-based) of the PDF at `path`, as a PNG image."""
    with pymupdf.open(path) as pdf:
        return pdf[page - 1].get_pixmap(dpi=RENDER_DPI).tobytes("png")


def split_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split on paragraph boundaries where possible, hard-splitting oversized paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        while len(paragraph) > size:
            head, paragraph = paragraph[:size], paragraph[size - overlap :]
            if current:
                pieces.append(current)
                current = ""
            pieces.append(head)
        if current and len(current) + 2 + len(paragraph) > size:
            pieces.append(current)
            # Carry the tail forward so a fact split across the boundary stays findable.
            current = current[-overlap:] + "\n\n" + paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        pieces.append(current)
    return pieces


def find_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
