"""The shared Document set: fictional company Documents rendered as multi-page PDFs.

Sources are plain text in benchmark/corpus/, one file per Document, pages separated
by a PAGE_BREAK line. The rendered PDFs in benchmark/documents/ are committed, so
Perry and QA-bot ingest byte-identical files. Rebuild after editing a source:

    uv run python -m benchmark.corpus
"""

from pathlib import Path

import pymupdf

HERE = Path(__file__).parent
SOURCES_DIR = HERE / "corpus"
DOCUMENTS_DIR = HERE / "documents"
PAGE_BREAK = "=== page break ==="

FONT_SIZE = 10.5
MARGIN = 56  # points


def source_pages(source: Path) -> list[str]:
    text = source.read_text(encoding="utf-8")
    return [page.strip() for page in text.split(PAGE_BREAK)]


def render(source: Path, target: Path) -> None:
    pdf = pymupdf.open()
    for text in source_pages(source):
        page = pdf.new_page()  # A4
        box = page.rect + (MARGIN, MARGIN, -MARGIN, -MARGIN)
        if page.insert_textbox(box, text, fontsize=FONT_SIZE, fontname="helv") < 0:
            raise ValueError(f"{source.name}: a page is too long to fit on one PDF page")
    # Fixed metadata keeps the output byte-stable across rebuilds.
    pdf.set_metadata({"title": source.stem, "creationDate": "", "modDate": ""})
    pdf.save(target, garbage=4, deflate=True, no_new_id=True)
    pdf.close()


def build(out_dir: Path = DOCUMENTS_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    built = []
    for source in sorted(SOURCES_DIR.glob("*.txt")):
        target = out_dir / f"{source.stem}.pdf"
        render(source, target)
        built.append(target)
    return built


if __name__ == "__main__":
    for path in build():
        print(path.relative_to(HERE.parent))
