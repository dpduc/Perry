"""Perry's chat UI. Run with: uv run python -m perry.app"""

from pathlib import Path

import gradio as gr

from perry import db, threads, usage
from perry.ingest import ingest_file
from perry.supervisor import run_turn, thread_history

EMPTY = {"text": "", "files": []}


def show_user_message(message: dict, history: list[dict]):
    """Show the message (text and any dropped images) in the chat and clear the box."""
    text, images = message.get("text", ""), message.get("files", [])
    if not text.strip() and not images:
        return EMPTY, history, EMPTY
    shown = [{"role": "user", "content": {"path": image}} for image in images]
    if text.strip():
        shown.append({"role": "user", "content": text})
    return {"text": text, "files": images}, [*history, *shown], EMPTY


def answer(message: dict, history: list[dict], thread_id: str | None):
    text, images = message.get("text", ""), message.get("files", [])
    if not text.strip() and not images:
        return history, thread_id, gr.skip()
    if thread_id is None:
        thread_id = threads.create(text)
    reply = run_turn(thread_id, text, images=images)
    history = [*history, {"role": "assistant", "content": reply}]
    return history, thread_id, gr.update(choices=threads.list_recent(), value=thread_id)


def open_thread(thread_id: str | None):
    if not thread_id:
        return [], None
    return thread_history(thread_id), thread_id


def new_thread():
    return [], None, gr.update(value=None)


def upload(files: list[str] | None) -> str:
    if not files:
        return gr.skip()
    lines = []
    for file in files:
        try:
            lines.append(ingest_file(Path(file)).describe())
        except Exception as error:  # noqa: BLE001 - report per file, keep going with the rest
            lines.append(f"{Path(file).name}: failed ({error})")
    return "\n".join(f"- {line}" for line in lines)


def todays_usage() -> str:
    today = usage.today()
    return f"Today: {today.tokens:,} tokens · ${today.cost_usd:.4f}"


def build() -> gr.Blocks:
    with gr.Blocks(title="Perry") as demo:
        thread_id = gr.State(None)
        pending = gr.State(EMPTY)

        with gr.Sidebar():
            gr.Markdown("## Perry")
            new_button = gr.Button("New thread", variant="primary")
            thread_picker = gr.Dropdown(
                choices=threads.list_recent(), label="Threads", value=None
            )
            gr.Markdown("### Documents")
            uploader = gr.File(
                file_count="multiple", file_types=[".pdf", ".md", ".txt"], label="Upload"
            )
            upload_status = gr.Markdown()
            usage_status = gr.Markdown()

        chatbot = gr.Chatbot(height="75vh", label="Perry")
        # Images (screenshots, photos) can be dropped or attached alongside the text.
        box = gr.MultimodalTextbox(
            placeholder="Ask Perry…",
            show_label=False,
            submit_btn=True,
            file_types=["image"],
            file_count="multiple",
        )

        box.submit(show_user_message, [box, chatbot], [pending, chatbot, box]).then(
            answer, [pending, chatbot, thread_id], [chatbot, thread_id, thread_picker]
        )
        thread_picker.input(open_thread, thread_picker, [chatbot, thread_id])
        new_button.click(new_thread, None, [chatbot, thread_id, thread_picker])
        uploader.upload(upload, uploader, upload_status).then(todays_usage, None, usage_status)
        # Refreshed whenever Perry finishes an answer, and on page load.
        chatbot.change(todays_usage, None, usage_status)
        demo.load(todays_usage, None, usage_status)
    return demo


if __name__ == "__main__":
    db.init_schema()
    build().launch(server_name="127.0.0.1")
