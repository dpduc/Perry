# Perry

A personal multi-agent assistant. A **Supervisor** delegates each request to specialist **Agents** and answers in a chat UI. Vocabulary: [GLOSSARY.md](GLOSSARY.md). Decisions: [docs/adr/](docs/adr/).

Built so far (step 1 of 6): the **Knowledge Agent**. It answers from your uploaded Documents with page Citations, using hybrid search (pgvector + Postgres full-text) in one local Postgres. Conversations are saved as resumable Threads.

## Setup

Needs Docker Desktop running and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
copy .env.example .env               # then fill in GEMINI_API_KEY
git config core.hooksPath .githooks  # ruff + pytest before each commit
docker compose up -d --wait          # Postgres + pgvector on 127.0.0.1:5433
```

## Use

```powershell
uv run python -m perry.app              # chat UI at http://127.0.0.1:7860
uv run python -m perry.ingest some\folder  # bulk-load Documents (.pdf, .md, .txt)
```

Documents can also be uploaded from the UI sidebar. Uploading a file with the same name replaces the old version.

**Datasets** (CSV, or XLSX with one Dataset per sheet) are uploaded from the sidebar's Datasets section, which lists each Dataset with its columns. Uploading a file with an existing Dataset name replaces it. The **Analysis Agent** answers questions about Datasets with SQL and shows the query it ran. That SQL runs as a separate Postgres role (`perry_analyst`) that can only read the `analytics` schema, with a statement timeout and a row limit.

## Notes

- **Speed on the free tier:** Gemini Flash allows 5 requests/minute, and Perry paces itself to 4. A Turn makes 3-6 model calls, so expect about 1-2 minutes per answer.
- **Privacy:** on the free tier, Google may use what you send to improve its products. Don't upload anything confidential.
- **Scanned PDF pages** (no extractable text) are skipped for now and listed after upload. Vision transcription is a later step.
- **Changing the embedding model** or its dimension means re-ingesting every Document.
