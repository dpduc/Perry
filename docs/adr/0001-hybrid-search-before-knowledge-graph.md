# Hybrid search before a knowledge graph

The Knowledge Agent retrieves Chunks with hybrid search in Postgres (pgvector similarity plus full-text search, merged by reciprocal rank fusion) rather than LightRAG's knowledge graph, even though the predecessor project QA-bot used LightRAG. The graph costs one or more LLM calls per Chunk at ingest, makes retrieval quality depend on opaque LLM extraction, needs the Apache AGE extension that the stock pgvector image lacks, and hands chunking and citation format to LightRAG. Its advantages (multi-hop, thematic and entity-centric questions) are real but unproven on our Documents, so the graph will be added only if it measurably beats hybrid search in a benchmark against QA-bot on the same question set.

## Consequences

The Knowledge Agent's interface (question in, cited Chunks out) must not expose anything hybrid-specific, so a graph can be added behind it without changing the Supervisor.
