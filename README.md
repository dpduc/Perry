# Perry

A personal multi-agent assistant. A **Supervisor** delegates each request to specialist **Agents** and answers in a chat UI. Vocabulary: [GLOSSARY.md](GLOSSARY.md). Decisions: [docs/adr/](docs/adr/).

Built so far (step 1 of 6): the **Knowledge Agent**. It answers from your uploaded Documents with page Citations, using hybrid search (pgvector + Postgres full-text) in one local Postgres. Conversations are saved as resumable Threads. The **Task Agent** reads your Google Calendar and Gmail (Read Actions only; Write Actions come later).

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

## Connecting Google (Task Agent)

The Task Agent reads your calendar and email (Read Actions only for now). Use a **test Google account containing only fake data**: on the free tier, what Perry reads is sent to Gemini.

1. In [Google Cloud Console](https://console.cloud.google.com/), create a project, enable the **Google Calendar API** and **Gmail API**, and set up the OAuth consent screen (External). Add the scopes `calendar.events`, `gmail.readonly` and `gmail.send`, then **publish the app to production** so sign-ins don't expire after 7 days (an unverified personal app is fine; you click through the warning).
2. Create an OAuth client of type **Desktop app**, download its JSON and save it as `%APPDATA%\Perry\google_client.json` (or point `PERRY_GOOGLE_CLIENT_FILE` at it). Never put it in the repo.
3. Start Perry and click **Connect Google** in the sidebar. A browser tab asks for consent; afterwards the button reads "Google connected".

The sign-in is kept in `%APPDATA%\Perry\google_token.json`, outside the repo, Postgres and the sandbox. If Google refuses it (expired or revoked), Perry answers "Google needs reconnecting" and the button turns into **Reconnect Google**.

Manual check against the test account: add a few events for tomorrow and a few emails (e.g. from "Sam" about a budget), then ask "What's on tomorrow?" and "What did Sam say about the budget?". To check the reconnect path, revoke Perry at <https://myaccount.google.com/permissions> and ask again.

## Knowledge benchmark (ADR-0001)

`benchmark/` runs the same 23 questions against Perry and the predecessor QA-bot (LightRAG) over the Documents in `benchmark/documents/`, and scores correctness and Citation accuracy per question type. See the docstring of `benchmark/run.py` for the commands; results land in `benchmark/results/`.

## Notes

- **Speed on the free tier:** Gemini Flash allows 5 requests/minute, and Perry paces itself to 4. A Turn makes 3-6 model calls, so expect about 1-2 minutes per answer.
- **Privacy:** on the free tier, Google may use what you send to improve its products. Don't upload anything confidential.
- **Scanned PDF pages** (no extractable text) are skipped for now and listed after upload. Vision transcription is a later step.
- **Changing the embedding model** or its dimension means re-ingesting every Document.
- **Usage:** every model call is recorded with its tokens and cost (the sidebar shows today's totals). Prices live in `prices.toml`; edit it to match current pricing. A Turn stops at `PERRY_TURN_TOKEN_CEILING` tokens (default 50,000) and reports what it found so far. Embedding tokens are estimated (about 4 characters per token), since the Gemini API doesn't report them.
