# Embeddings run locally; chat runs on the free tier with a paid fallback

Every Chunk and Summary is embedded on this machine with Ollama's `nomic-embed-text` (768 dimensions), not with Gemini's `gemini-embedding-001` (1536). Chat stays on Gemini's free tier and falls back to a paid OpenAI model only when the free daily quota is used up or Gemini stays overloaded.

Unlike chat, embeddings can't fall back: vectors from two models can't be compared, so every stored embedding must come from one model. That leaves one choice per installation. We chose local over Gemini because the free embedding quota would otherwise stop uploads mid-way, because it costs nothing, and because Summaries are then never sent anywhere (they're written by the local model too). We chose it over OpenAI's embeddings because those are paid and the paid budget is better spent on chat. The cost is about 1 s per Chunk on this CPU at upload time and a smaller, general-purpose model.

## Consequences

- The vector columns are 768-dimensional. `db.init_schema` resizes them when `PERRY_EMBEDDING_DIMENSIONS` changes, and the start-up step re-embeds every Chunk and Summary whose embedding is missing or was made by another model (the model is recorded in `store_facts`). Until then, those rows are found only by their words.
- Ollama must be running for uploads and for the vector half of every search.
- The cascade margin is tuned per embedding model, so it was re-tuned (0.074 with Gemini, 0.116 with nomic).
- Benchmark rows run on one paid model throughout, never the free one, so no row mixes two models.
