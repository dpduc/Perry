"""Every stored embedding is made by the current embedding model (docs/adr/0003)."""

from dataclasses import replace

from fakes import ScriptedChatModel, fake_embed, fake_embed_one, fake_provider

from perry import db, models, reembed
from perry.ingest import ingest_file
from perry.search import hybrid_search


def reversed_embed_one(text: str) -> list[float]:
    """Another model: the same words, pointing somewhere else entirely."""
    return list(reversed(fake_embed_one(text)))


def another_model(provider):
    return replace(
        provider,
        embedding_model="other-embedding",
        embed_documents=lambda texts: [reversed_embed_one(t) for t in texts],
        embed_query=reversed_embed_one,
    )


def stored_embedding(table: str = "chunks"):
    with db.pool().connection() as conn:
        return conn.execute(f"SELECT embedding FROM {table} LIMIT 1").fetchone()["embedding"].to_list()


def test_nothing_is_re_embedded_while_the_model_stays_the_same(database, tmp_path):
    (tmp_path / "budget.md").write_text("The marketing budget for Q3 is 40,000 dollars.")
    with models.use(fake_provider(ScriptedChatModel())):
        ingest_file(tmp_path / "budget.md", embed=fake_embed)
        assert reembed.ensure_current(report=lambda _: None) == 0
        assert reembed.ensure_current(report=lambda _: None) == 0


def test_a_new_embedding_model_re_embeds_every_chunk_and_summary(database, tmp_path):
    (tmp_path / "budget.md").write_text("The marketing budget for Q3 is 40,000 dollars.")
    provider = fake_provider(ScriptedChatModel())
    with models.use(provider):
        ingest_file(tmp_path / "budget.md", embed=fake_embed)
        reembed.ensure_current(report=lambda _: None)
    with db.pool().connection() as conn:
        conn.execute(
            "INSERT INTO summaries (document_id, kind, text, embedding) "
            "SELECT id, 'document', 'A note on the Q3 budget.', %s FROM documents",
            (stored_embedding(),),
        )

    with models.use(another_model(provider)):
        done = reembed.ensure_current(report=lambda _: None)
        hits = hybrid_search("Q3 marketing budget")

    assert done == 2  # the Chunk and the Summary
    assert stored_embedding("chunks") == reversed_embed_one(
        "The marketing budget for Q3 is 40,000 dollars."
    )
    assert hits[0].citation == "budget.md"


def test_a_new_embedding_size_resizes_the_columns_and_re_embeds_at_start(database, tmp_path):
    (tmp_path / "budget.md").write_text("The marketing budget for Q3 is 40,000 dollars.")
    with models.use(fake_provider(ScriptedChatModel())):
        ingest_file(tmp_path / "budget.md", embed=fake_embed)
    # As if the Chunks were stored by an embedding model of another size.
    with db.pool().connection() as conn:
        conn.execute("DROP INDEX chunks_embedding_idx")
        conn.execute("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(4) USING NULL")

    db.init_schema()
    with models.use(fake_provider(ScriptedChatModel())):
        assert hybrid_search("Q3 marketing budget")  # still found by its words meanwhile
        done = reembed.ensure_current(report=lambda _: None)
        hits = hybrid_search("Q3 marketing budget")

    assert done == 1
    assert len(stored_embedding()) == models.EMBEDDING_DIMENSIONS
    assert hits[0].citation == "budget.md"
