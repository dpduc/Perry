# Reranking runs after every Retrieval Technique, on a local cross-encoder

Every search's Chunks are reranked before the Knowledge Agent reads them: the technique gathers a wider pool (20 candidates instead of 6), a local cross-encoder reads the question with each one, and the best 6 go on. We made it a step after every technique, not an eleventh Retrieval Technique, because reranking improves whatever a technique found rather than suiting one kind of question, so as a technique the picker would rarely choose it. ADR-0002 chose a full ColBERT index over "only reranking hybrid search"; that still holds, since ColBERT finds Chunks itself, and reranking now runs after ColBERT too.

## Considered Options

- **Reranking with the chat model** (ask Gemini or OpenAI to order the Chunks): rejected. It would spend free-tier quota or money on every search and add a slow call, where a cross-encoder runs free and private on this machine in under half a second for 20 Chunks.
- **An eleventh Retrieval Technique**: rejected, see above.

## Consequences

- With reranking off (`PERRY_RERANK_MODEL` empty), every technique returns exactly what it did before, so the benchmark compares the two with `+rerank` setups.
- If reranking fails, the technique's own best 6 are used and the Found-with line says "reranking failed"; the answer is never lost.
- Summary-based techniques can widen their pool only to 9 (3 Summaries × 3 Chunks).
- The default model (`cross-encoder/ms-marco-MiniLM-L6-v2`) is English-only; Documents in other languages need a multilingual one such as `BAAI/bge-reranker-v2-m3`, which is much slower on a CPU.
