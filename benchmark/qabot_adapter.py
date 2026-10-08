"""Drive QA-bot for the benchmark. Runs inside QA-bot's own virtualenv (see run.py):

    python qabot_adapter.py ingest <folder>   # build the LightRAG graph
    python qabot_adapter.py ask <mode>        # question on stdin, JSON answer on stdout

QA-bot's code is used unchanged. The only addition is request pacing, because
QA-bot has none and LightRAG's graph extraction makes many calls in a burst.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from lightrag import QueryParam
from lightrag.base import DocStatus
from qa_bot import models
from qa_bot.ingest import ingest
from qa_bot.rag import open_rag

# 15 s = 4 requests per minute, under the free tier's 5. Lower it on a paid tier.
SECONDS_PER_REQUEST = float(os.getenv("QABOT_SECONDS_PER_REQUEST", "15"))

_unpaced = models.llm_func
_lock = asyncio.Lock()
_last_call = 0.0


async def paced_llm(*args, **kwargs):
    global _last_call
    async with _lock:
        wait = _last_call + SECONDS_PER_REQUEST - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call = time.monotonic()
    return await _unpaced(*args, **kwargs)


models.llm_func = paced_llm


async def check_ingested() -> None:
    rag = await open_rag()
    try:
        counts = await rag.doc_status.get_status_counts()
        failed = await rag.doc_status.get_docs_by_statuses([DocStatus.FAILED])
    finally:
        await rag.finalize_storages()
    print(f"QA-bot document status: {counts}", file=sys.stderr)
    if failed:
        sys.exit(f"QA-bot failed to ingest {len(failed)} pages; rerun to retry them.")


async def ask(question: str, mode: str) -> dict:
    rag = await open_rag()
    try:
        result = await rag.aquery_llm(
            question, param=QueryParam(mode=mode, include_references=True)
        )
    finally:
        await rag.finalize_storages()
    # On "no relevant context" LightRAG returns a canned refusal in "message".
    answer = (result.get("llm_response") or {}).get("content") or result.get("message", "")
    references = (result.get("data") or {}).get("references", [])
    return {"answer": answer, "references": references}


def main() -> None:
    command = sys.argv[1]
    if command == "ingest":
        asyncio.run(ingest(Path(sys.argv[2])))
        asyncio.run(check_ingested())
    elif command == "ask":
        question = sys.stdin.read().strip()
        print(json.dumps(asyncio.run(ask(question, sys.argv[2])), ensure_ascii=False))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
