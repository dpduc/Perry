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

Push to talk: record with the microphone under the chat box and stop to send. Perry shows the transcription as your message and reads its answer aloud (text-to-speech model: `PERRY_TTS_MODEL`). Speech counts toward usage like any other model call.

## Knowledge benchmark (ADR-0001)

`benchmark/` runs the same 23 questions against Perry and the predecessor QA-bot (LightRAG) over the Documents in `benchmark/documents/`, and scores correctness and Citation accuracy per question type. See the docstring of `benchmark/run.py` for the commands; results land in `benchmark/results/`.

## Notes

- **Speed on the free tier:** Gemini Flash allows 5 requests/minute, and Perry paces itself to 4. A Turn makes 3-6 model calls, so expect about 1-2 minutes per answer.
- **Privacy:** on the free tier, Google may use what you send to improve its products. Don't upload anything confidential.
- **Scanned PDF pages** (no extractable text) are skipped for now and listed after upload. Vision transcription is a later step.
- **Changing the embedding model** or its dimension means re-ingesting every Document.
- **Usage:** every model call is recorded with its tokens and cost (the sidebar shows today's totals). Prices live in `prices.toml`; edit it to match current pricing. A Turn stops at `PERRY_TURN_TOKEN_CEILING` tokens (default 50,000) and reports what it found so far. Embedding tokens are estimated (about 4 characters per token), since the Gemini API doesn't report them.
