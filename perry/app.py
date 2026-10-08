"""Perry's chat UI. Run with: uv run python -m perry.app"""

from pathlib import Path

import gradio as gr

from perry import db, google_auth, privacy, threads, usage, workspace
from perry.ingest import ingest_file
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


GOOGLE_BUTTON = {
    "connected": ("Google connected", "secondary"),
    "not connected": ("Connect Google", "primary"),
    "needs reconnecting": ("Reconnect Google", "stop"),
}


def google_button():
    """The Google button as the current Workspace status calls for."""
    label, variant = GOOGLE_BUTTON[workspace.current().status()]
    return gr.update(value=label, variant=variant)


ACCOUNT_KINDS = [("Test account (fake data only)", "test"), ("Real account", "real")]


def connect_google(kind: str | None):
    if kind is None:
        return gr.skip(), "Choose whether this is a test or a real account first."
    try:
        google_auth.connect(kind)  # opens the browser and waits for consent
        message = f"Google connected ({kind} account)."
    except privacy.Refused as refusal:
        message = str(refusal)
    except Exception as error:  # noqa: BLE001 - show it next to the button, keep the UI up
        message = f"Couldn't connect Google: {error}"
    return google_button(), message


def privacy_notice() -> str:
    return privacy.startup() or ""


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
            usage_status = gr.Markdown()
            gr.Markdown("### Calendar and email")
            account_kind = gr.Radio(ACCOUNT_KINDS, value=None, label="Account")
            google = gr.Button()
            google_status = gr.Markdown()

        chatbot = gr.Chatbot(height="75vh", label="Perry")
        box = gr.Textbox(placeholder="Ask Perry…", show_label=False, submit_btn=True)

        box.submit(show_user_message, [box, chatbot], [pending, chatbot, box]).then(
            answer, [pending, chatbot, thread_id], [chatbot, thread_id, thread_picker]
        ).then(google_button, None, google)
        thread_picker.input(open_thread, thread_picker, [chatbot, thread_id])
        new_button.click(new_thread, None, [chatbot, thread_id, thread_picker])
        uploader.upload(upload, uploader, upload_status).then(todays_usage, None, usage_status)
        # Refreshed whenever Perry finishes an answer, and on page load.
        chatbot.change(todays_usage, None, usage_status)
        demo.load(todays_usage, None, usage_status)
        demo.load(google_button, None, google)
        demo.load(privacy_notice, None, google_status)
        google.click(connect_google, account_kind, [google, google_status])
    return demo


if __name__ == "__main__":
    # Before anything runs: switches tracing off if a real account is connected.
    if notice := privacy.startup():
        print(notice)
    db.init_schema()
    build().launch(server_name="127.0.0.1")
