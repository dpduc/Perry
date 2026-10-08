"""Perry's chat UI. Run with: uv run python -m perry.app"""

from pathlib import Path

import gradio as gr

from perry import db, google_auth, ingest, threads, usage, workspace
from perry.supervisor import run_turn, thread_history


def show_user_message(message: str, history: list[dict]):
    if not message.strip():
        return gr.skip(), history, ""
    return message, [*history, {"role": "user", "content": message}], ""


def answer(message: str, history: list[dict], thread_id: str | None):
    if not message.strip():
        return history, thread_id, gr.skip()
    if thread_id is None:
        thread_id = threads.create(message)
    reply = run_turn(thread_id, message)
    history = [*history, {"role": "assistant", "content": reply}]
    return history, thread_id, gr.update(choices=threads.list_recent(), value=thread_id)


def open_thread(thread_id: str | None):
    if not thread_id:
        return [], None
    return thread_history(thread_id), thread_id


def new_thread():
    return [], None, gr.update(value=None)


def propose_upload(files: list[str] | None):
    """Show what ingesting the uploaded files would cost, and wait for the user's OK."""
    if not files:
        return gr.skip(), gr.skip(), gr.skip()
    estimate = ingest.estimate_ingestion([Path(file) for file in files])
    try:
        ingest.check_budget(estimate)
    except ingest.OverBudget as refused:
        return [], str(refused), gr.update(visible=False)
    if not estimate.tokens:
        # Nothing to pay for (e.g. every file is unchanged): no need to ask.
        return [], *ingest_upload(files)
    return files, f"{estimate.describe()} Ingest now?", gr.update(visible=True)


def ingest_upload(files: list[str]):
    try:
        results = ingest.ingest_files([Path(file) for file in files])
    except ingest.OverBudget as refused:
        return str(refused), gr.update(visible=False)
    return "\n".join(f"- {r.describe()}" for r in results), gr.update(visible=False)


def cancel_upload():
    return [], "Nothing ingested.", gr.update(visible=False)


def todays_usage() -> str:
    today = usage.today()
    return (
        f"Today: {today.tokens:,} tokens · ${today.cost_usd:.4f} "
        f"of ${usage.daily_cap_usd():.2f}"
    )


GOOGLE_BUTTON = {
    "connected": ("Google connected", "secondary"),
    "not connected": ("Connect Google", "primary"),
    "needs reconnecting": ("Reconnect Google", "stop"),
}


def google_button():
    """The Google button as the current Workspace status calls for."""
    label, variant = GOOGLE_BUTTON[workspace.current().status()]
    return gr.update(value=label, variant=variant)


def connect_google():
    try:
        google_auth.connect()  # opens the browser and waits for consent
        message = "Google connected."
    except Exception as error:  # noqa: BLE001 - show it next to the button, keep the UI up
        message = f"Couldn't connect Google: {error}"
    return google_button(), message


def build() -> gr.Blocks:
    with gr.Blocks(title="Perry") as demo:
        thread_id = gr.State(None)
        pending = gr.State("")

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
            uploaded = gr.State([])  # files waiting for the user's OK to ingest
            with gr.Row(visible=False) as upload_choice:
                ingest_button = gr.Button("Ingest", variant="primary", size="sm")
                cancel_button = gr.Button("Cancel", size="sm")
            usage_status = gr.Markdown()
            gr.Markdown("### Calendar and email")
            google = gr.Button()
            google_status = gr.Markdown()

        chatbot = gr.Chatbot(height="75vh", label="Perry")
        box = gr.Textbox(placeholder="Ask Perry…", show_label=False, submit_btn=True)

        box.submit(show_user_message, [box, chatbot], [pending, chatbot, box]).then(
            answer, [pending, chatbot, thread_id], [chatbot, thread_id, thread_picker]
        ).then(google_button, None, google)
        thread_picker.input(open_thread, thread_picker, [chatbot, thread_id])
        new_button.click(new_thread, None, [chatbot, thread_id, thread_picker])
        uploader.upload(
            propose_upload, uploader, [uploaded, upload_status, upload_choice]
        ).then(todays_usage, None, usage_status)
        ingest_button.click(
            ingest_upload, uploaded, [upload_status, upload_choice]
        ).then(todays_usage, None, usage_status)
        cancel_button.click(cancel_upload, None, [uploaded, upload_status, upload_choice])
        # Refreshed whenever Perry finishes an answer, and on page load.
        chatbot.change(todays_usage, None, usage_status)
        demo.load(todays_usage, None, usage_status)
        demo.load(google_button, None, google)
        google.click(connect_google, None, [google, google_status])
    return demo


if __name__ == "__main__":
    db.init_schema()
    build().launch(server_name="127.0.0.1")
