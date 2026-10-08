"""Perry's chat UI. Run with: uv run python -m perry.app"""

from pathlib import Path

import gradio as gr

from perry import (
    charts,
    datasets,
    db,
    google_auth,
    ingest,
    models,
    privacy,
    threads,
    usage,
    workspace,
)
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
    history = [*history, *charts.chat_messages(reply)]  # any charts show as images
    return history, thread_id, gr.update(choices=threads.list_recent(), value=thread_id)


def hear(recording: str | None, history: list[dict]):
    """Push-to-talk: transcribe the recording and show it as the user's message."""
    if not recording:
        return gr.skip(), history, None, False
    try:
        usage.check_daily_cap()  # transcribing is a model call, so it waits for the cap too
    except usage.DailyCapReached as refused:
        return EMPTY, [*history, {"role": "assistant", "content": str(refused)}], None, False
    try:
        transcript = models.transcribe(Path(recording).read_bytes(), "audio/wav")
    except Exception as error:  # noqa: BLE001 - tell the user, keep the UI usable
        gr.Warning(f"Couldn't transcribe the recording: {error}")
        return EMPTY, history, None, False
    if not transcript.strip():
        gr.Warning("I didn't catch anything. Hold the button and speak, then release.")
        return EMPTY, history, None, False
    message = {"text": transcript, "files": []}
    return message, [*history, {"role": "user", "content": transcript}], None, True


def read_aloud(spoken: bool, history: list[dict]):
    """Speak the latest answer, only when the request came by voice."""
    if not spoken or not history or history[-1]["role"] != "assistant":
        return None
    text = _text(history[-1]["content"])
    if not text:
        return None
    try:
        return models.synthesize(text)
    except Exception as error:  # noqa: BLE001 - the text answer is already shown
        gr.Warning(f"Couldn't read the answer aloud: {error}")
        return None


def _text(content) -> str:
    """A chat message's text: Gradio passes content as a str or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def open_thread(thread_id: str | None):
    if not thread_id:
        return [], None
    history = []
    for message in thread_history(thread_id):
        is_answer = message["role"] == "assistant"
        history += charts.chat_messages(message["content"]) if is_answer else [message]
    return history, thread_id


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


def upload_datasets(files: list[str] | None) -> tuple[str, str]:
    if not files:
        return gr.skip(), gr.skip()
    lines = []
    for file in files:
        try:
            lines += [result.describe() for result in datasets.upload(Path(file))]
        except Exception as error:  # noqa: BLE001 - report per file, keep going with the rest
            lines.append(f"{Path(file).name}: failed ({error})")
    return "\n".join(f"- {line}" for line in lines), dataset_list()


def dataset_list() -> str:
    found = datasets.list_datasets()
    if not found:
        return "_No Datasets yet._"
    return "\n".join(f"- **{d.name}** ({d.row_count} rows): {', '.join(d.columns)}" for d in found)


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
        pending = gr.State(EMPTY)
        spoken = gr.State(False)  # whether the latest request came by voice

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
            gr.Markdown("### Datasets")
            dataset_uploader = gr.File(
                file_count="multiple", file_types=[".csv", ".xlsx"], label="Upload data"
            )
            dataset_status = gr.Markdown()
            datasets_shown = gr.Markdown(dataset_list)
            usage_status = gr.Markdown()
            gr.Markdown("### Calendar and email")
            account_kind = gr.Radio(ACCOUNT_KINDS, value=None, label="Account")
            google = gr.Button()
            google_status = gr.Markdown()

        chatbot = gr.Chatbot(height="75vh", label="Perry")
        # Images (screenshots, photos) can be dropped or attached alongside the text.
        box = gr.MultimodalTextbox(
            placeholder="Ask Perry…",
            show_label=False,
            submit_btn=True,
            file_types=["image"],
            file_count="multiple",
        )
        with gr.Row(visible=models.speech_supported()):
            mic = gr.Audio(
                sources=["microphone"],
                type="filepath",
                format="wav",
                label="Push to talk: record, then stop to send",
            )
            speaker = gr.Audio(label="Perry's answer", autoplay=True, interactive=False)

        box.submit(show_user_message, [box, chatbot], [pending, chatbot, box]).then(
            answer, [pending, chatbot, thread_id], [chatbot, thread_id, thread_picker]
        ).then(google_button, None, google)
        box.submit(lambda: False, None, spoken)
        mic.stop_recording(hear, [mic, chatbot], [pending, chatbot, mic, spoken]).then(
            answer, [pending, chatbot, thread_id], [chatbot, thread_id, thread_picker]
        ).then(read_aloud, [spoken, chatbot], speaker).then(
            todays_usage, None, usage_status
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
        dataset_uploader.upload(
            upload_datasets, dataset_uploader, [dataset_status, datasets_shown]
        )
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
    build().launch(server_name="127.0.0.1", allowed_paths=[str(charts.directory())])
