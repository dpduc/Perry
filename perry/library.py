"""Manage the library's Documents from the command line:

    uv run python -m perry.library list
    uv run python -m perry.library delete <name> [--yes]
    uv run python -m perry.library replace <name> <newer file> [--yes]
    uv run python -m perry.library rename <name> <new name>

Works while Perry runs: its searches see the change at once. A name may be typed in
any case, and with its accents typed or pasted any way; an unknown one gets the
closest names suggested. Delete and replace ask first (--yes skips the question).
Anyone who can run commands on this machine is trusted like the Owner, as with
create-owner.
"""

import argparse
import difflib
import re
import sys
import unicodedata
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from perry import db, ingest, models, summaries


class Unknown(LookupError):
    """No Document has the name typed (the message suggests close ones)."""


def documents() -> list[tuple[str, int, datetime]]:
    """Every Document's name, number of Chunks and upload time, by name."""
    with db.pool().connection() as conn:
        rows = conn.execute(
            """SELECT d.name, count(c.id) AS chunks, d.ingested_at FROM documents d
               LEFT JOIN chunks c ON c.document_id = d.id
               GROUP BY d.id ORDER BY lower(d.name)"""
        ).fetchall()
    return [(row["name"], row["chunks"], row["ingested_at"]) for row in rows]


def _key(name: str) -> str:
    """`name` as it is matched: same accents however they were typed, any case."""
    return unicodedata.normalize("NFC", name).casefold().strip()


def _loose(name: str) -> str:
    """`name` as it is compared for suggestions: also spaces, underscores and hyphens alike."""
    return re.sub(r"[\s_-]+", " ", _key(name))


def find(typed: str) -> str:
    """The name of the Document `typed` means: its exact name, or the one name equal to
    it ignoring case and how accents were typed. Raises Unknown, suggesting up to three
    close names."""
    names = [name for name, *_ in documents()]
    if typed in names:
        return typed
    matches = [name for name in names if _key(name) == _key(typed)]
    if len(matches) == 1:
        return matches[0]
    keys = {_loose(name): name for name in names}
    part_of = [name for key, name in keys.items() if _loose(typed) and _loose(typed) in key]
    alike = difflib.get_close_matches(_loose(typed), keys, n=3, cutoff=0.6)
    close = (matches or list(dict.fromkeys([*part_of, *(keys[key] for key in alike)])))[:3]
    message = f"No Document is called {typed!r}."
    if close:
        message += f" Did you mean {' or '.join(map(_quoted, close))}?"
    else:
        message += " See them with `python -m perry.library list`."
    raise Unknown(message)


def _quoted(name: str) -> str:
    """`name` as it would be typed in a command: in quotes if it has spaces."""
    return f'"{name}"' if " " in name else name


def _chunks(count: int) -> str:
    return "1 Chunk" if count == 1 else f"{count} Chunks"


def _describe(name: str) -> str:
    """`name (12 Chunks, uploaded 2026-10-10)`."""
    _, chunks, uploaded = next(d for d in documents() if d[0] == name)
    return f"{name} ({_chunks(chunks)}, uploaded {uploaded:%Y-%m-%d})"


def _confirmed(question: str, yes: bool, ask: Callable[[str], str]) -> bool:
    return yes or ask(f"{question} [y/N] ").strip().lower() in {"y", "yes"}


def _list() -> int:
    rows = documents()
    if not rows:
        print("There are no Documents yet.")
    for name, chunks, uploaded in rows:
        print(f"{name}  ({_chunks(chunks)}, uploaded {uploaded:%Y-%m-%d})")
    return 0


def _delete(typed: str, yes: bool, ask) -> int:
    name = find(typed)
    if not _confirmed(f"Delete {_describe(name)}?", yes, ask):
        print("Nothing was deleted.")
        return 1
    chunks = ingest.delete_document(name)
    print(f"Deleted {name} and its {_chunks(chunks)}.")
    _catch_up()
    return 0


def _replace(typed: str, file: str, yes: bool, ask) -> int:
    name = find(typed)
    path = Path(file)
    if not path.is_file():
        print(f"There is no file {file}. Nothing was replaced.", file=sys.stderr)
        return 1
    estimate = ingest.estimate_ingestion([path])
    ingest.check_budget(estimate)
    cost = estimate.describe().removeprefix("1 file: ").removesuffix(".")
    question = f"Replace {_describe(name)} with {path.name} ({cost})?"
    if not _confirmed(question, yes, ask):
        print("Nothing was replaced.")
        return 1
    result = ingest.ingest_file(path, replaces=name)
    if path.name == name:
        print(result.describe())
    else:
        print(f"{path.name} replaced {name}: {_chunks(result.chunks)}.")
    _catch_up()
    return 0


def _rename(typed: str, new_name: str) -> int:
    name = find(typed)
    try:
        ingest.rename_document(name, new_name)
    except ingest.NameTaken:
        print(f"A Document is already called {new_name}. Nothing was renamed.", file=sys.stderr)
        return 1
    print(f"Renamed {name} to {new_name}. Its Chunks are cited by the new name.")
    return 0


def _catch_up() -> None:
    """Write the pending Summaries and rebuild RAPTOR's cross-Document level now, so a
    running Perry needn't restart for them. On a CPU this takes a minute or more."""
    if not models.summaries_supported() or not (
        summaries.pending_count() or len(documents()) > 1
    ):
        return
    print(
        "Updating Summaries and RAPTOR trees on the local model "
        "(Ctrl+C leaves it to Perry's next start)...",
        flush=True,
    )
    try:
        summaries.summarise_pending()
    except KeyboardInterrupt:
        print("Left for Perry's next start.")


def main(argv: list[str] | None = None, *, ask: Callable[[str], str] = input) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m perry.library",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="every Document, with its Chunks and upload date")
    delete = commands.add_parser("delete", help="delete a Document")
    delete.add_argument("name")
    delete.add_argument("--yes", action="store_true", help="don't ask first")
    replace = commands.add_parser("replace", help="upload a newer version under any name")
    replace.add_argument("name")
    replace.add_argument("file")
    replace.add_argument("--yes", action="store_true", help="don't ask first")
    rename = commands.add_parser("rename", help="give a Document a new name")
    rename.add_argument("name")
    rename.add_argument("new_name")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    db.init_schema()
    try:
        if args.command == "list":
            return _list()
        if args.command == "delete":
            return _delete(args.name, args.yes, ask)
        if args.command == "replace":
            return _replace(args.name, args.file, args.yes, ask)
        return _rename(args.name, args.new_name)
    except Unknown as unknown:
        print(f"{unknown} Nothing was changed.", file=sys.stderr)
    except ingest.OverBudget as refused:
        print(f"{refused} Nothing was replaced.", file=sys.stderr)
    except (KeyboardInterrupt, EOFError):
        print("\nNothing was changed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
