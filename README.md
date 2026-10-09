# Perry

A personal multi-agent assistant that runs on your own machine. You talk to one chat; a **Supervisor** hands each request to the right specialist **Agent** and answers with the results. Vocabulary: [GLOSSARY.md](GLOSSARY.md). Decisions: [docs/adr/](docs/adr/).

![Perry's chat UI: Threads, Documents and Datasets in the sidebar, today's usage, the Google account section, and the chat with a push-to-talk recorder below it](docs/images/perry-ui.png)

## What Perry can do

| Agent | What it does for you |
|---|---|
| **Knowledge Agent** | Answers questions from the Documents you upload (PDF, Markdown, text) and cites the exact page each claim came from. Picks a Retrieval Technique per question (e.g. hybrid search, pgvector + Postgres full-text; or ColBERT, a local late-interaction model with its own on-disk index, for exact names and rare terms) and says which one it used. Scanned PDF pages with no text are read by the vision model at upload. |
| **Analysis Agent** | Answers questions about your Datasets (CSV, or each sheet of an XLSX). It writes read-only SQL and shows you the query it ran. When SQL isn't enough (statistics, forecasts, charts) it writes Python that runs in a locked-down Docker **Sandbox**, and the charts appear in the chat. |
| **Task Agent** | Reads your Google Calendar and Gmail freely. Anything that changes something or reaches another person (send, reply, create, move or delete an event) waits for you on a **Confirmation card**, where you approve, edit the fields, or reject with a note. |

Around the Agents:

- **Chaining:** one request can use several Agents ("find the Q3 notes and email a summary to Sam"). A Turn stops after about 6 steps and says so.
- **Threads:** every conversation is saved and resumable from the sidebar.
- **Voice and images:** hold-to-record push-to-talk (Perry reads its answer aloud), and you can drop images or screenshots into the chat.
- **Guardrails:** every model call is recorded with its tokens and cost. A Turn stops at a token ceiling and reports what it found so far; a daily cap refuses new Turns until midnight; bulk uploads show an estimated cost and wait for your OK. A step that fails is retried once, then reported honestly, and a failed Write Action is never retried.
- **Privacy:** a real Google account is refused unless text generation runs on a local model, and tracing is switched off while one is connected. See [Connecting Google](#connecting-google-task-agent).

## Setup

Needs Docker Desktop running and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
git config core.hooksPath .githooks  # ruff + pytest before each commit
docker compose up -d --wait          # Postgres + pgvector on 127.0.0.1:5433
```

Then create `.env` in the repo root (it's gitignored, never commit it):

```ini
GEMINI_API_KEY=...                   # https://aistudio.google.com/apikey
# Optional LangSmith tracing; the endpoint must match your account's region:
# api. (US), eu.api. or apac.api.smith.langchain.com (a 403 on every call means a region mismatch)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=...
```

Everything else has a default; see [Settings](#settings) for the optional overrides.

## Use

```powershell
uv run python -m perry.app              # chat UI at http://127.0.0.1:7860
uv run python -m perry.ingest some\folder  # bulk-load Documents (.pdf, .md, .txt); shows the estimated cost and asks first (--yes skips the question)
```

Your data lives in the Postgres from `compose.yaml`, in the Docker volume `pgdata`, so it survives restarts. Perry stores each Document as page-numbered Chunks with their embeddings (not the original file), and each Dataset as a table.

### The UI

- **Sidebar:** *New thread* and a *Threads* picker to resume an earlier conversation; *Documents* and *Datasets* upload areas (the Datasets list shows each name with its columns); today's usage (`Today: N tokens · $x of $cap`); and *Calendar and email*, where you pick **Test account** or **Real account** and click **Connect Google**.
- **Chat:** type in the box (the paperclip attaches images). Below it, **Record** is push-to-talk and the player on the right speaks Perry's answer.
- When Perry needs your approval, a **Confirmation card** appears under the chat with the editable fields, **Approve**, **Reject** and a note box.

## Tutorials

The repo ships small sample files you can use to try everything: [evals/documents/](evals/documents/) (four short Documents) and [evals/datasets/sales.csv](evals/datasets/sales.csv) (2025 ad spend, units and revenue for three regions). Answers come from the real model, so expect each one to take a minute or two on the free tier.

### 1. Ask your Documents

1. Start Perry and drop a few files into **Documents** in the sidebar (or run `uv run python -m perry.ingest evals\documents`).
2. Perry shows an estimated token count and cost and asks *Ingest now?* Click **Ingest**. If an upload would exceed today's remaining budget, it is refused before any call is made.
3. Ask, for example: *What's my Q3 marketing budget?* With the sample files you should get 40,000 dollars, with a Citation to `q3-budget.md`.
4. Ask something that isn't in your files, such as *What's my Q4 budget?* Perry should say it can't find it rather than guess.
5. Questions that need two files work too: *How does the offsite compare with the trade fair?*
6. Each answer ends with its Citations, then a line such as `Found with: Hybrid search (picked logically)` naming the Retrieval Technique that found them. To try a specific technique, pick it in the sidebar's **Retrieval Technique** dropdown (Auto by default); it is used for your next questions, reported as `(forced)`, until you set it back to Auto. Perry still decides on its own whether a question needs your Documents at all.

### 2. Replace or update a Document

- **Same file name:** upload it again. Perry replaces the old Document and its Chunks in one step. Identical content is skipped at no cost.
- **Different file name** (`report-v2.pdf` for `report.pdf`): the old one stays, because Documents are matched by name only.
- **There is no delete button yet.** To remove a Document, run:

  ```powershell
  docker compose exec db psql -U perry -d perry -c "DELETE FROM documents WHERE name = 'report.pdf';"
  ```

  Its Chunks go with it. Your Threads are untouched.
- Scanned PDF pages with no text are transcribed by the vision model and cited by page like any other. The upload estimate includes them, and a page that fails to transcribe is listed after the upload.

### 3. Analyse a Dataset

1. Drop `sales.csv` (or your own CSV/XLSX; each sheet becomes its own Dataset) into **Datasets**. It appears in the list with its columns. Uploading a file with the same name replaces that Dataset.
2. Ask: *What was total 2025 revenue per region?* Perry runs a read-only SQL query and shows the query it used, so you can check the logic. With the sample file the totals are North 181,300, South 156,350 and West 220,850.
3. Ask for something SQL is bad at: *Is ad spend correlated with revenue?* or *Forecast the next three months of revenue.* Perry writes Python, runs it in the Sandbox and shows the code it used.
4. Ask for a picture: *Plot monthly revenue per region as a chart.* The PNG appears in the chat above the answer.
5. A bad or slow query is cut off by a timeout and a row limit, and Perry tells you what failed so you can rephrase.

The first Python question builds the Sandbox image, which takes about 2 minutes and needs Docker running.

### 4. Calendar and email (test account first)

1. Follow [Connecting Google](#connecting-google-task-agent) once. In the sidebar choose **Test account (fake data only)**, then **Connect Google**.
2. Read-only questions run straight away: *What's on tomorrow?*, *What did Sam say about the budget last week?*, *Summarise my latest email thread with Sam.*
3. Ask for a change: *Email Sam a short summary of the Q3 budget.* Perry pauses on a **Confirmation card** showing exactly what will happen (To, Subject, Message).
4. Choose one:
   - **Approve** performs it exactly once.
   - **Edit** any field first (recipients, subject, body, start and end times, invitees), then **Approve**. What you edited is what gets performed.
   - **Reject** with a note in the box (for example "make it shorter and friendlier") and Perry revises and asks again.
5. If you type a new message while a card is waiting, the waiting action is cancelled and Perry says so. Nothing is sent hours later by surprise.
6. If a Write Action fails, Perry reports it and never retries, so you never get duplicate emails or events. If it says the outcome is unknown, check your account before asking again.

### 5. Chain several Agents

Ask for one result that needs more than one Agent: *Find my Q3 budget notes and email Sam a summary.* The Knowledge Agent finds the notes with Citations, then the Task Agent drafts the email and pauses on a Confirmation card before anything is sent. A chain never skips your approval.

### 6. Talk to Perry

Click **Record**, speak, and click stop. Your words appear as your message so you can catch mishearings, then Perry answers in text and reads the answer aloud. Speech counts toward your usage limits. If the daily cap is already reached, Perry says so instead of transcribing.

### 7. Ask about an image

Click the paperclip (or drop a file) to attach a screenshot, chart or receipt, and ask: *What does this error mean?* The image stays in the Thread, so follow-up questions about it work.

### 8. Keep an eye on cost

- The sidebar line `Today: N tokens · $x of $1.00` is read from the usage ledger.
- A single Turn stops at `PERRY_TURN_TOKEN_CEILING` tokens (default 50,000) and reports what it found so far.
- Once `PERRY_DAILY_CAP_USD` (default $1) is reached, new Turns are refused until local midnight. A Turn already running always finishes.
- Prices live in `prices.toml`; edit it to match current pricing. All limits are set in `.env`.

## More about the features

**Datasets** are stored as real tables in an `analytics` schema. The Analysis Agent's SQL runs as a separate Postgres role (`perry_analyst`) that can only read that schema, one statement at a time, in a read-only transaction with a statement timeout and a row limit, so a bad query can't touch your Documents, Threads or credentials.

The **Sandbox** is a throwaway Docker container from the image in `sandbox/Dockerfile` (pandas, numpy, scipy, matplotlib), built on first use. It has no network, CPU, memory and time limits, no environment from the host, and sees only a temp folder holding the input CSVs Perry wrote from read-only queries. Charts it saves are kept under `%APPDATA%\Perry\charts`. Sandbox tests skip when Docker isn't running.

Push to talk uses the chat model for transcription and `PERRY_TTS_MODEL` for speech.

## Eval set

```powershell
uv run python -m evals --list            # the eval Turns
uv run python -m evals                   # run them all against the real model
uv run python -m evals knowledge-aphids  # only Turns whose id starts with this
```

Each eval Turn ([evals/turns/](evals/turns/), one file per Agent) says which Agents must be consulted, which Citations the answer must contain and which Write Actions must wait for a Confirmation (the run approves them, against the fake); those checks pass or fail the run. An LLM judge also scores answer quality, which is only reported (and logged to the LangSmith project `perry-eval` when tracing is on). The run uses its own `perry_eval` database loaded from [evals/documents/](evals/documents/) and [evals/datasets/](evals/datasets/) (the `analysis-python-*` Turns run Python in the Sandbox, so they need Docker), and a fake calendar and mailbox ([evals/workspace.py](evals/workspace.py)) instead of Google. It isn't part of `uv run pytest` or the pre-commit hook, because it spends real requests: with pacing, a full run takes about 15 minutes on the free tier.

## Connecting Google (Task Agent)

The Task Agent reads your calendar and email freely, and performs Write Actions (send, reply, create, move or delete an event) only after you approve their Confirmation card. A new message cancels a waiting Confirmation; a failed Write Action is reported, never retried. Use a **test Google account containing only fake data**: on the free tier, what Perry reads is sent to Gemini.

1. In [Google Cloud Console](https://console.cloud.google.com/), create a project, enable the **Google Calendar API** and **Gmail API**, and set up the OAuth consent screen (External). Add the scopes `calendar.events`, `gmail.readonly` and `gmail.send`, then **publish the app to production** so sign-ins don't expire after 7 days (an unverified personal app is fine; you click through the warning).
2. Create an OAuth client of type **Desktop app**, download its JSON and save it as `%APPDATA%\Perry\google_client.json` (or point `PERRY_GOOGLE_CLIENT_FILE` at it). Never put it in the repo.
3. Start Perry, choose **Test account** or **Real account** in the sidebar, and click **Connect Google**. A browser tab asks for consent; afterwards the button reads "Google connected".

Privacy switches, enforced in code at connect time and at startup:
- A **real account** is refused unless text generation runs on a local model (for now it's Gemini, so only test accounts connect). If a real account is already connected while a cloud model is configured, the Task Agent is off and says why.
- While a real account is connected, LangSmith tracing is switched off, whatever `.env` says.
- A sign-in kept before Perry asked for the kind counts as real: reconnect and choose **Test account**.

The sign-in is kept in `%APPDATA%\Perry\google_token.json`, outside the repo, Postgres and the sandbox. If Google refuses it (expired or revoked), Perry answers "Google needs reconnecting" and the button turns into **Reconnect Google**.

Manual check against the test account: add a few events for tomorrow and a few emails (e.g. from "Sam" about a budget), then ask "What's on tomorrow?" and "What did Sam say about the budget?". To check the reconnect path, revoke Perry at <https://myaccount.google.com/permissions> and ask again.

## Knowledge benchmark (ADR-0001)

`benchmark/` runs the same 23 questions against Perry and the predecessor QA-bot (LightRAG) over the Documents in `benchmark/documents/`, and scores correctness and Citation accuracy per question type. See the docstring of `benchmark/run.py` for the commands; results land in `benchmark/results/`.

## Settings

All optional; add any of these to `.env` to override the default.

| Setting | Default | What it does |
|---|---|---|
| `PERRY_LLM_MODEL` | `gemini-3.8-flash` | Chat model; also used for speech-to-text and image reading |
| `PERRY_EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model (changing it means re-ingesting every Document) |
| `PERRY_TTS_MODEL` | `gemini-2.5-flash-preview-tts` | Reads voice answers aloud |
| `PERRY_TTS_VOICE` | `Kore` | Voice for read-aloud answers |
| `PERRY_REQUESTS_PER_SECOND` | `0.0667` | Request pacing (4/min; the free tier allows 5/min) |
| `DATABASE_URL` | `postgresql://perry:perry@127.0.0.1:5433/perry` | The Postgres Perry uses |
| `PERRY_RETRY_DELAY_SECONDS` | `2` | Wait before a failed step's one retry |
| `PERRY_TURN_TOKEN_CEILING` | `50000` | A Turn stops and reports its findings past this many tokens |
| `PERRY_DAILY_CAP_USD` | `1.00` | New Turns and ingestion are refused past this until local midnight |
| `PERRY_COLBERT_MODEL` | `answerdotai/answerai-colbert-small-v1` | Local model for the ColBERT Retrieval Technique (downloaded on first use, runs on the CPU); changing it re-encodes every Chunk at next start; empty turns ColBERT off |
| `PERRY_SHOW_FOUND_WITH` | `true` | End answers from your Documents with the "Found with" line naming the Retrieval Technique; `false` turns it off |
| `PERRY_PRICES_FILE` | `prices.toml` | Price table, USD per million tokens per model |
| `PERRY_ANALYST_PASSWORD` | `perry_analyst` | Password of the read-only `perry_analyst` role |
| `PERRY_ANALYSIS_TIMEOUT_MS` | `10000` | Statement timeout for the Analysis Agent's SQL |
| `PERRY_ANALYSIS_ROW_LIMIT` | `200` | Rows returned per SQL query |
| `PERRY_SANDBOX_TIME_LIMIT_S` | `60` | Sandbox time limit per script |
| `PERRY_SANDBOX_MEMORY` | `1g` | Sandbox memory limit |
| `PERRY_SANDBOX_CPUS` | `1` | Sandbox CPU limit |
| `PERRY_SANDBOX_INPUT_ROW_LIMIT` | `100000` | Rows per input file written for a script |
| `PERRY_APP_DATA_DIR` | `%APPDATA%\Perry` | Holds `google_token.json`, saved charts and the ColBERT index (`colbert\`) |
| `PERRY_GOOGLE_CLIENT_FILE` | `<app-data>\google_client.json` | Google OAuth client file |

## Notes

- **Speed on the free tier:** Gemini Flash allows 5 requests/minute, and Perry paces itself to 4. A Turn makes 3-6 model calls, so expect about 1-2 minutes per answer.
- **Privacy:** on the free tier, Google may use what you send to improve its products. Don't upload anything confidential.
- **Scanned PDF pages** (no extractable text) are rendered to images and transcribed by the vision model (the chat model, `PERRY_LLM_MODEL`), then cited by page like any other. The estimate shown before ingesting includes one transcription per such page, and the calls count toward usage. A page whose transcription fails is listed after upload.
- **ColBERT** keeps its own index under `<app-data>\colbert`, beside Postgres (docs/adr/0002). Every upload updates both before it returns, and a failed upload leaves both as they were. Uploads take a little longer while ColBERT encodes the text on the CPU (roughly 7 Chunks a second), and the first one downloads the model (~130 MB). At start, Perry encodes any Chunks stored before ColBERT existed and rebuilds the index if it doesn't match Postgres; deleting the folder is safe.
- **Changing the embedding model** or its dimension means re-ingesting every Document.
- **Usage:** every model call is recorded with its tokens and cost (the sidebar shows today's totals). Prices live in `prices.toml`; edit it to match current pricing. A Turn stops at `PERRY_TURN_TOKEN_CEILING` tokens (default 50,000) and reports what it found so far. Embedding tokens are estimated (about 4 characters per token), since the Gemini API doesn't report them. Once today's cost reaches `PERRY_DAILY_CAP_USD` (default $1), new Turns are refused until local midnight; a Turn already running always finishes. Uploads count toward the cap too: they show an estimate and wait for your OK, and are refused up front if they would go over.
