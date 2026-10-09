"""The ColBERT index: late-interaction search over Chunks, with its own on-disk PLAID index
under Perry's app-data folder, kept in step with Postgres (docs/adr/0002).

Postgres stays the source of truth: every Chunk stores its ColBERT token vectors
(`chunks.colbert`), and the on-disk index is built from them. The index isn't
transactional, so it is never edited in place. Ingestion builds a new one in a staging
folder inside its Postgres transaction (`update()`), and swaps it in only after the
commit; a failure on either side leaves both as they were. `ensure_in_step()`, run at
start, repairs an index a crash left behind and encodes Chunks stored without vectors.

fast-plaid (and torch) are imported on first use only: importing them takes tens of seconds.
"""

import functools
import io
import json
import logging
import shutil
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from perry import db, models, paths
from perry.search import SearchHit

log = logging.getLogger(__name__)

# Beside PLAID's files: the model, and the Chunk ids in the order they were added.
_STATE = "state.json"
_PLAID = "plaid"
# PLAID can't train its codec on a handful of token vectors ("Cannot train codec"), so a
# tiny index is padded with a filler entry that is never returned.
_MIN_VECTORS = 32

# One process-wide lock: an index file can't change while it is open or being built.
_lock = threading.RLock()
_open: dict[Path, tuple] = {}  # live folder -> its open PLAID index and Chunk ids


def folder() -> Path:
    """Where the ColBERT index lives."""
    return paths.app_data_dir() / "colbert"


def _live() -> Path:
    return folder() / "live"


def _staging() -> Path:
    return folder() / "staging"


def to_bytes(vectors: np.ndarray) -> bytes:
    """One Chunk's token vectors as stored in `chunks.colbert`."""
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(vectors, np.float32), allow_pickle=False)
    return buffer.getvalue()


def from_bytes(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


class Update:
    """A change to the Chunks under way: `stage()` the index they will have."""

    def __init__(self):
        self.staged = False

    def stage(self, conn, new: Iterable[tuple[int, np.ndarray]]) -> None:
        """Build the index of every Chunk `conn`'s transaction sees, plus `new`
        (Chunk id, token vectors) pairs not inserted yet. Swapped in after the commit."""
        rows = conn.execute(
            "SELECT id, colbert FROM chunks WHERE colbert IS NOT NULL"
        ).fetchall()
        entries = [(row["id"], from_bytes(row["colbert"])) for row in rows]
        _build(_staging(), [*entries, *new])
        self.staged = True


@contextmanager
def update() -> Iterator[Update]:
    """Wrap a Postgres transaction that changes Chunks, and `stage()` the new index inside
    it. Once the block (and so the commit) succeeds, the staged index replaces the live
    one; if anything fails, it is thrown away and the live index is untouched."""
    with _lock:
        change = Update()
        try:
            yield change
        except BaseException:
            shutil.rmtree(_staging(), ignore_errors=True)
            raise
        if change.staged:
            _swap()


def search(question: str, limit: int = 6) -> list[SearchHit]:
    """The Chunks that best match `question` token by token (MaxSim), best first."""
    if not models.colbert_supported():
        return []
    query = models.colbert_encode_query(question)
    with _lock:
        live = _live_index()
        if live is None:
            return []
        index, chunk_ids = live
        results = index.search(_tensors([query]), top_k=limit + 1, show_progress=False)[0]
    # PLAID numbers entries in the order they were added: Chunk ids, then the filler.
    scores = {chunk_ids[i]: float(score) for i, score in results if i < len(chunk_ids)}
    if not scores:
        return []
    with db.pool().connection() as conn:
        rows = conn.execute(
            "SELECT c.id, d.name, c.page, c.text FROM chunks c "
            "JOIN documents d ON d.id = c.document_id WHERE c.id = ANY(%s)",
            (list(scores),),
        ).fetchall()
    hits = [SearchHit(r["name"], r["page"], r["text"], scores[r["id"]]) for r in rows]
    return sorted(hits, key=lambda hit: -hit.score)[:limit]


def ensure_in_step() -> None:
    """Make the index match Postgres: encode Chunks stored without ColBERT vectors (stored
    before ColBERT existed, or by another model), then rebuild the index if its Chunks
    differ from Postgres', e.g. after a crash between a commit and the swap."""
    if not models.colbert_supported():
        return
    model = models.colbert_model()
    with _lock, db.pool().connection() as conn:
        state = _state(_live())
        if state and state["model"] != model:
            conn.execute("UPDATE chunks SET colbert = NULL")
        missing = conn.execute(
            "SELECT id, text FROM chunks WHERE colbert IS NULL ORDER BY id"
        ).fetchall()
        if missing:
            log.info("Encoding %d Chunks for the ColBERT index", len(missing))
            vectors = models.colbert_encode_documents([row["text"] for row in missing])
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE chunks SET colbert = %s WHERE id = %s",
                    [(to_bytes(v), row["id"]) for v, row in zip(vectors, missing, strict=True)],
                )
        ids = [row["id"] for row in conn.execute("SELECT id FROM chunks ORDER BY id")]
        if state and state["model"] == model and state["chunk_ids"] == ids:
            return
        log.info("Rebuilding the ColBERT index (%d Chunks)", len(ids))
        with update() as change:
            change.stage(conn, [])


def close() -> None:
    """Release the open index's files, e.g. before its folder is moved or deleted."""
    with _lock:
        for live in list(_open):
            _open.pop(live)[0].close()


def _build(target: Path, entries: list[tuple[int, np.ndarray]]) -> None:
    """Write a fresh index of `entries` (Chunk id, token vectors) to `target`."""
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    entries = sorted(entries, key=lambda entry: entry[0])
    if entries:
        vectors = [vectors for _, vectors in entries]
        if sum(len(v) for v in vectors) < _MIN_VECTORS:
            filler = np.random.default_rng(0).standard_normal((_MIN_VECTORS, vectors[0].shape[1]))
            vectors.append(filler / np.linalg.norm(filler, axis=1, keepdims=True))
        index = _plaid(target)
        try:
            index.create(_tensors(vectors), seed=42)
        finally:
            index.close()
    state = {"model": models.colbert_model(), "chunk_ids": [chunk_id for chunk_id, _ in entries]}
    (target / _STATE).write_text(json.dumps(state), encoding="utf-8")


def _swap() -> None:
    """Replace the live index with the staged one."""
    live, staging, old = _live(), _staging(), folder() / "old"
    if (opened := _open.pop(live, None)) is not None:
        opened[0].close()  # Windows can't move a folder with memory-mapped files in it
    shutil.rmtree(old, ignore_errors=True)
    if live.exists():
        live.rename(old)
    staging.rename(live)
    shutil.rmtree(old, ignore_errors=True)


def _live_index():
    """The open live index and its Chunk ids, or None if it holds no Chunks yet."""
    live = _live()
    if live not in _open:
        close()  # any index still open is another folder's, e.g. app-data changed
        chunk_ids = _state(live).get("chunk_ids")
        if not chunk_ids:
            return None
        _open[live] = (_plaid(live), chunk_ids)
    return _open[live]


def _state(index_folder: Path) -> dict:
    path = index_folder / _STATE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _plaid(index_folder: Path):
    """The PLAID index in `index_folder` (fast-plaid, on the CPU), created by `create()`.
    Close it to release its memory-mapped files: Windows won't move or delete them."""
    from fast_plaid.search import FastPlaid
    from fast_plaid.search import fast_plaid as module

    # Every open looks for torch's library with recursive globs (seconds on Windows),
    # and the answer never changes: look once.
    if not hasattr(module._load_torch_path, "cache_info"):
        module._load_torch_path = functools.cache(module._load_torch_path)
    return FastPlaid(index=str(index_folder / _PLAID), device="cpu")


def _tensors(vectors: list[np.ndarray]) -> list:
    import torch

    return [torch.from_numpy(np.asarray(v, np.float32)) for v in vectors]
