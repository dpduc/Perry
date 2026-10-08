"""Perry's chat UI. Run with: uv run python -m perry.app"""

from pathlib import Path

import gradio as gr

from perry import db, models, threads, usage
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


def hear(recording: str | None, history: list[dict]):
    """Push-to-talk: transcribe the recording and show it as the user's message."""
    if not recording:
        return gr.skip(), history, None, False
    try:
        transcript = models.transcribe(Path(recording).read_bytes(), "audio/wav")
    except Exception as error:  # noqa: BLE001 - tell the user, keep the UI usable
        gr.Warning(f"Couldn't transcribe the recording: {error}")
        return "", history, None, False
    if not transcript.strip():
        gr.Warning("I didn't catch anything. Hold the button and speak, then release.")
        return "", history, None, False
    return transcript, [*history, {"role": "user", "content": transcript}], None, True


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
        pending = gr.State("")
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
            usage_status = gr.Markdown()

        chatbot = gr.Chatbot(height="75vh", label="Perry")
        box = gr.Textbox(placeholder="Ask Perry…", show_label=False, submit_btn=True)
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
        )
        box.submit(lambda: False, None, spoken)
        mic.stop_recording(hear, [mic, chatbot], [pending, chatbot, mic, spoken]).then(
            answer, [pending, chatbot, thread_id], [chatbot, thread_id, thread_picker]
        ).then(read_aloud, [spoken, chatbot], speaker).then(todays_usage, None, usage_status)
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
