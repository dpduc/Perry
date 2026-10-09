"""RAPTOR trees, built in the background by the local summary model.

Each Document gets a tree: its Chunks are clustered by embedding, each cluster is
summarised (one node), and the nodes are clustered and summarised again, level by level,
up to one root. Over the trees sits a cross-Document level: the Documents' roots
clustered and summarised the same way, so themes that span Documents have a node.

Nodes are Summaries of kind "raptor" (`summaries.save_summary`), each linked to every
leaf Chunk beneath it. A match at any node returns those Chunks with their page
Citations (`search.summary_search(kind="raptor")`); a node is never cited.

Clustering follows the RAPTOR paper (Sarthi et al., 2024) on a small scale: embeddings
reduced with PCA, a Gaussian mixture whose size is picked by BIC, and soft membership
(a node joins every cluster it is likely enough to belong to). UMAP is left out: it is
slow to load and fails on the handful of Chunks a short Document has.

A Document's tree has its own status (`raptor_status`): `pending` after upload, `ready`
once built, `failed` if the local model failed (retried on the next start). Only ready
trees are searched, and only their roots go into the cross-Document level, which is
rebuilt whenever a tree is built. Re-uploading a Document deletes its tree with it.
"""

import logging
import math
from dataclasses import dataclass

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from perry import db, models, summaries
from perry.documents import citation

log = logging.getLogger(__name__)

PENDING, READY, FAILED = summaries.PENDING, summaries.READY, summaries.FAILED
KIND = summaries.RAPTOR

# At most this many nodes are summarised together: it bounds one summary prompt (six
# Chunks are ~2,300 tokens, minutes on a CPU) and makes every level shrink.
MAX_CLUSTER = 6
# A node joins every cluster whose probability for it is above this (the paper's 0.1).
MEMBERSHIP_THRESHOLD = 0.1
# Embeddings are reduced to this many dimensions before clustering.
PCA_DIMENSIONS = 5
# What one node's summary reads at most, and is expected to cost (for the estimate).
INPUT_CHARS = 10_000
PROMPT_TOKENS = 100
OUTPUT_TOKENS = 150

NODE_PROMPT = """You summarise related excerpts of the user's documents so the summary \
can be found later by its themes. Write 3 to 5 sentences covering the main themes, key \
names, figures and conclusions the excerpts share or differ on. Use only what the \
excerpts say, in their language. Reply with the summary only."""


@dataclass(frozen=True)
class Node:
    """One RAPTOR node: its level (1 is just above the Chunks), its Summary, and the
    Citations of the leaf Chunks beneath it."""

    level: int
    text: str
    citations: tuple[str, ...]


@dataclass(frozen=True)
class _Draft:
    """A node (or a Chunk, as a leaf) while a tree is built: not stored yet."""

    text: str
    embedding: np.ndarray
    chunk_ids: frozenset[int]


# --- Clustering -----------------------------------------------------------------


def cluster(embeddings: np.ndarray) -> list[list[int]]:
    """Group the rows of `embeddings` into clusters of at most MAX_CLUSTER, by index.
    Membership is soft: a row may sit in several clusters, and every row is in one."""
    n = len(embeddings)
    if n <= MAX_CLUSTER:
        return [list(range(n))]
    vectors = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    reduced = PCA(n_components=min(PCA_DIMENSIONS, n - 2), random_state=0).fit_transform(vectors)

    def mixture(k: int) -> GaussianMixture:
        return GaussianMixture(k, covariance_type="diag", random_state=0).fit(reduced)

    # BIC picks the number of clusters; never fewer than keep clusters small enough.
    fewest = math.ceil(n / MAX_CLUSTER)
    best = min(range(2, n // 2 + 1), key=lambda k: mixture(k).bic(reduced))
    probabilities = mixture(max(best, fewest)).predict_proba(reduced)

    members = probabilities > MEMBERSHIP_THRESHOLD
    members[np.arange(n), probabilities.argmax(axis=1)] = True
    clusters = [list(np.flatnonzero(column)) for column in members.T if column.any()]
    # A cluster too big to summarise at once is split in order.
    return [
        [int(i) for i in group[start : start + MAX_CLUSTER]]
        for group in clusters
        for start in range(0, len(group), MAX_CLUSTER)
    ]


def _array(vector) -> np.ndarray:
    """A stored embedding (pgvector's Vector) as a numpy array."""
    return vector.to_numpy() if hasattr(vector, "to_numpy") else np.asarray(vector)


def _summarise(drafts: list[_Draft]) -> str:
    texts = "\n\n".join(draft.text for draft in drafts)[:INPUT_CHARS]
    reply = models.summary_model(kind=KIND).invoke(
        [SystemMessage(NODE_PROMPT), HumanMessage(texts)]
    )
    text, _, _ = summaries.parse_reply(reply.text)  # tolerates a JSON reply
    return text


def _build(level0: list[_Draft]) -> list[tuple[int, _Draft]]:
    """Cluster and summarise `level0` level by level, up to one root (or until a level
    stops shrinking), and return every new node with its level."""
    nodes: list[tuple[int, _Draft]] = []
    current, level = level0, 0
    while current and (level == 0 or len(current) > 1):
        groups = cluster(np.stack([draft.embedding for draft in current]))
        if level > 0 and len(groups) >= len(current):
            break  # no progress: these are the tree's top nodes
        texts = [_summarise([current[i] for i in group]) for group in groups]
        embeddings = models.embed_documents(texts)
        level += 1
        current = [
            _Draft(text, np.asarray(embedding), frozenset().union(*(current[i].chunk_ids for i in group)))
            for text, embedding, group in zip(texts, embeddings, groups, strict=True)
        ]
        nodes += [(level, draft) for draft in current]
    return nodes


def estimate(chunk_tokens: list[int]) -> tuple[int, int, int]:
    """What building the tree of a Document with Chunks of these sizes is expected to
    cost: (summary tokens in, summary tokens out, how many nodes). Each level reads the
    one below it; soft membership and the cross-Document level are left out."""
    nodes, tokens_in = 0, sum(chunk_tokens)
    count = len(chunk_tokens)
    while count:
        count = math.ceil(count / MAX_CLUSTER)
        nodes += count
        if count == 1:
            break
        tokens_in += count * OUTPUT_TOKENS  # the next level reads these nodes
    return tokens_in + nodes * PROMPT_TOKENS, nodes * OUTPUT_TOKENS, nodes


# --- Background job -------------------------------------------------------------


def build_pending() -> int:
    """Build the tree of every Document whose tree is pending, one at a time, then
    rebuild the cross-Document level if any was built. Returns how many were tried."""
    tried = 0
    while (document_id := _next_pending()) is not None:
        tried += 1
        try:
            _build_document(document_id)
        except Exception:
            log.warning("Couldn't build RAPTOR tree %s; will retry", document_id, exc_info=True)
            _set_status(document_id, FAILED)
    if tried or _cross_document_missing():
        try:
            _build_cross_document()
        except Exception:
            log.warning("Couldn't build the RAPTOR cross-Document level", exc_info=True)
    return tried


def _next_pending() -> int | None:
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT id FROM documents WHERE raptor_status = %s ORDER BY id LIMIT 1", (PENDING,)
        ).fetchone()
    return row["id"] if row else None


def _set_status(document_id: int, status: str) -> None:
    with db.pool().connection() as conn:
        conn.execute("UPDATE documents SET raptor_status = %s WHERE id = %s", (status, document_id))


def _build_document(document_id: int) -> None:
    with db.pool().connection() as conn:
        chunks = conn.execute(
            "SELECT id, text, embedding FROM chunks WHERE document_id = %s ORDER BY page, id",
            (document_id,),
        ).fetchall()
    leaves = [_Draft(c["text"], _array(c["embedding"]), frozenset([c["id"]])) for c in chunks]
    nodes = _build(leaves)  # the slow model calls happen outside any transaction

    with db.pool().connection() as conn, conn.transaction():
        still_there = conn.execute(
            "SELECT 1 FROM documents WHERE id = %s FOR UPDATE", (document_id,)
        ).fetchone()
        if still_there is None:  # replaced or deleted meanwhile: its new version is pending
            return
        conn.execute(
            "DELETE FROM summaries WHERE document_id = %s AND kind = %s", (document_id, KIND)
        )
        for level, node in nodes:
            _save(conn, document_id, level, node)
        conn.execute(
            "UPDATE documents SET raptor_status = %s WHERE id = %s", (READY, document_id)
        )


def _roots() -> list[_Draft]:
    """The top nodes of every ready Document's tree."""
    with db.pool().connection() as conn:
        rows = conn.execute(
            """SELECT s.text, s.embedding, array_agg(sc.chunk_id) AS chunk_ids
               FROM summaries s
               JOIN documents d ON d.id = s.document_id AND d.raptor_status = %s
               JOIN summary_chunks sc ON sc.summary_id = s.id
               WHERE s.kind = %s AND s.level = (
                   SELECT max(level) FROM summaries t
                   WHERE t.document_id = s.document_id AND t.kind = %s)
               GROUP BY s.id ORDER BY s.document_id, s.id""",
            (READY, KIND, KIND),
        ).fetchall()
    return [_Draft(r["text"], _array(r["embedding"]), frozenset(r["chunk_ids"])) for r in rows]


def _cross_document_missing() -> bool:
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM summaries WHERE kind = %s AND document_id IS NULL LIMIT 1", (KIND,)
        ).fetchone()
    return row is None and len(_roots()) > 1


def _build_cross_document() -> None:
    """Replace the cross-Document level with one built over the ready trees' roots.
    A single tree needs none: its root is already the top."""
    roots = _roots()
    nodes = _build(roots) if len(roots) > 1 else []
    with db.pool().connection() as conn, conn.transaction():
        conn.execute("DELETE FROM summaries WHERE kind = %s AND document_id IS NULL", (KIND,))
        for level, node in nodes:
            # Skip Chunks deleted (re-uploaded) while the level was being built.
            alive = conn.execute(
                "SELECT array_agg(id) AS ids FROM chunks WHERE id = ANY(%s)",
                (list(node.chunk_ids),),
            ).fetchone()["ids"]
            if alive:
                _save(conn, None, level, _Draft(node.text, node.embedding, frozenset(alive)))


def _save(conn, document_id: int | None, level: int, node: _Draft) -> None:
    summaries.save_summary(
        conn,
        document_id=document_id,
        kind=KIND,
        level=level,
        text=node.text,
        embedding=[float(x) for x in node.embedding],
        chunk_ids=sorted(node.chunk_ids),
    )


# --- Reading trees --------------------------------------------------------------


def status(name: str) -> str | None:
    """The status of the RAPTOR tree of the Document called `name`."""
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT raptor_status FROM documents WHERE name = %s", (name,)
        ).fetchone()
    return row["raptor_status"] if row else None


NODES_SQL = """
SELECT s.level, s.text, array_agg(DISTINCT ARRAY[d.name, c.page::text]) AS pages
FROM summaries s
LEFT JOIN documents sd ON sd.id = s.document_id
JOIN summary_chunks sc ON sc.summary_id = s.id
JOIN chunks c ON c.id = sc.chunk_id
JOIN documents d ON d.id = c.document_id
WHERE s.kind = %(kind)s AND {where}
GROUP BY s.id
ORDER BY s.level, s.id
"""


def tree(name: str) -> list[Node]:
    """The nodes of the Document called `name`'s tree, from level 1 up to the root."""
    return _nodes("sd.name = %(name)s", name=name)


def cross_document() -> list[Node]:
    """The nodes of the cross-Document level, lowest first."""
    return _nodes("s.document_id IS NULL")


def _nodes(where: str, **params) -> list[Node]:
    with db.pool().connection() as conn:
        rows = conn.execute(NODES_SQL.format(where=where), {"kind": KIND, **params}).fetchall()
    return [
        Node(
            row["level"],
            row["text"],
            tuple(
                citation(name, int(page))
                for name, page in sorted(row["pages"], key=lambda p: (p[0], int(p[1])))
            ),
        )
        for row in rows
    ]
