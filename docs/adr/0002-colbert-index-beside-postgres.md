# ColBERT keeps its own on-disk index beside Postgres

ColBERT is one of the Knowledge Agent's Retrieval Techniques, and it searches all the Chunks itself, from its own on-disk index (PLAID, written by a local ColBERT model under Perry's app-data folder), rather than only reranking hybrid search's candidates. Late interaction stores a vector per token and scores with MaxSim, which pgvector can't do, so a full late-interaction index can't live in Postgres. We chose the full index over reranking, which needs no index, because the point is to compare three independent Retrieval Techniques (hybrid search, Summaries, ColBERT) on the benchmark, and reranking only reorders hybrid search's results.

## Consequences

This is the one exception to "everything in one Postgres". Ingestion owns keeping the two in step: uploading, replacing or deleting a Document updates Postgres and the ColBERT index together, and if either side fails, the Document is left as it was before.

Perry and a command-line run (`python -m perry.ingest`, `python -m perry.library`) share one index per database, so each new index is built in a folder of its own and a small `current` file names the live one. A swap never moves a folder that another process has open (Windows refuses), and searches open the newer index when `current` changes. A Postgres advisory lock lets one process at a time build and swap.
