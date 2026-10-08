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

## Eval set

```powershell
uv run python -m evals --list            # the eval Turns
uv run python -m evals                   # run them all against the real model
uv run python -m evals knowledge-aphids  # only Turns whose id starts with this
```

Each eval Turn ([evals/turns/](evals/turns/), one file per Agent) says which Agents must be consulted and which Citations the answer must contain; those checks pass or fail the run. An LLM judge also scores answer quality, which is only reported (and logged to the LangSmith project `perry-eval` when tracing is on). The run uses its own `perry_eval` database loaded from [evals/documents/](evals/documents/). It isn't part of `uv run pytest` or the pre-commit hook, because it spends real requests: with pacing, a full run takes about 15 minutes on the free tier.

## Notes

- **Speed on the free tier:** Gemini Flash allows 5 requests/minute, and Perry paces itself to 4. A Turn makes 3-6 model calls, so expect about 1-2 minutes per answer.
- **Privacy:** on the free tier, Google may use what you send to improve its products. Don't upload anything confidential.
- **Scanned PDF pages** (no extractable text) are skipped for now and listed after upload. Vision transcription is a later step.
- **Changing the embedding model** or its dimension means re-ingesting every Document.
