# Perry

- **Vocabulary**: use the terms in `GLOSSARY.md` (Supervisor, Agent, Turn, Thread, Document, Chunk, Citation, Write Action, Confirmation, …) in code, prompts and docs.
- **Model provider**: `perry/models.py` is the only module that knows the model providers: Gemini free tier for chat, a paid OpenAI fallback when its quota runs out, local Ollama for embeddings and Summaries, and ColBERT (`docs/adr/0003`). Switching a model is a change there only.
- **Database**: one Postgres (`compose.yaml`, port 5433). The one exception is ColBERT's on-disk index, which ingestion keeps in step with Postgres (`docs/adr/0002`). Connect via `127.0.0.1`, not `localhost`: on Windows, `localhost` tries IPv6 first and stalls. The pool prepares every statement (the LangGraph checkpointer requires it), so multi-statement SQL needs `prepare=False`.
- **Knowledge Agent interface** stays retrieval-agnostic (question in, cited findings out). See `docs/adr/0001`.
- **Library APIs**: langchain 1.x / langgraph 1.x / gradio 6.x differ a lot from older tutorials. Verify signatures against `.venv/Lib/site-packages`.
- **Checks**: `uv run pytest` (DB tests need `docker compose up -d`; they use a separate `perry_test` database) and `uv run ruff check .`

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues for `dpduc/Perry` via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
