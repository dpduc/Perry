"""Perry's chat UI, served behind the login pages by perry.web. Run with:
uv run python -m perry.web

Handlers that act for the User take Gradio's request first and get the User with
`user_of`, never from anything the browser holds."""

from pathlib import Path

import gradio as gr

from perry import (
    charts,
    datasets,
    google_auth,
    ingest,
    models,
    privacy,
    summaries,
    techniques,
    threads,
    usage,
    users,
    voice,
    workspace,
)
from perry.supervisor import (
    Approve,
    Confirmation,
    Reject,
    pending_confirmation,
    resume_turn,
    run_turn,
    thread_history,
)

EMPTY = {"text": "", "files": []}

LOG_IN_AGAIN = "You're not logged in. Reload the page to log in again."


def user_of(request: gr.Request | None) -> users.User:
    """The User Gradio's request belongs to (perry.web gives Gradio their id as the
    request's username). Refused if there is none or they are disabled."""
    username = getattr(request, "username", None)
    user = users.get(int(username)) if username and str(username).isdigit() else None
    if user is None or user.disabled:
        raise gr.Error(LOG_IN_AGAIN)
    return user


OWNER_ONLY = "Only the Owner can do that."


def owner_of(request: gr.Request | None) -> users.User:
    """The Owner, if Gradio's request is theirs; refused for anyone else (ADR-0004)."""
    user = user_of(request)
    if not user.is_owner:
        raise gr.Error(OWNER_ONLY)
    return user


# The sidebar's Owner-only controls: Document upload, Dataset upload, Workspace connect.
OWNER_CONTROLS = ("documents", "datasets", "workspace")


def owner_controls(request: gr.Request) -> list:
    """Show the Owner-only controls (hidden until this says otherwise) to the Owner only."""
    shown = user_of(request).is_owner
    return [gr.update(visible=shown) for _ in OWNER_CONTROLS]


def logged_in_as(request: gr.Request) -> str:
    """The sidebar's note on who is logged in, with a link to the Owner panel for the
    Owner."""
    user = user_of(request)
    note = f"Logged in as **{user.profile.display_name}**"
    return f"{note} · [Owner panel](/owner)" if user.is_owner else note


# Every field a Write Action can have: (label, lines, editable). The Confirmation card
# shows the ones its Write Action takes.
CARD_FIELDS = {
    "to": ("To (comma-separated)", 1, True),
    "subject": ("Subject", 1, True),
    "body": ("Message", 6, True),
    "title": ("Title", 1, True),
    "start": ("Start (YYYY-MM-DDTHH:MM)", 1, True),
    "end": ("End (YYYY-MM-DDTHH:MM)", 1, True),
    "location": ("Location", 1, True),
    "attendees": ("Invite (comma-separated emails)", 1, True),
    "email_thread_id": ("Email thread", 1, False),
    "event_id": ("Event", 1, False),
}


def show_user_message(message: dict, history: list[dict]):
    """Show the message (text and any dropped images) in the chat and clear the box."""
    text, images = message.get("text", ""), message.get("files", [])
    if not text.strip() and not images:
        return EMPTY, history, EMPTY
    shown = [{"role": "user", "content": {"path": image}} for image in images]
    if text.strip():
        shown.append({"role": "user", "content": text})
    return {"text": text, "files": images}, [*history, *shown], EMPTY


AUTO = "auto"  # the dropdown's default: the Knowledge Agent picks the Retrieval Technique


def technique_choices() -> list[tuple[str, str]]:
    """The Retrieval Technique dropdown: Auto, then every registered technique."""
    return [("Auto", AUTO), *((t.label, t.name) for t in techniques.TECHNIQUES)]


def answer(
    request: gr.Request,
    message: dict,
    history: list[dict],
    thread_id: str | None,
    technique: str = AUTO,
    voice_turn: str | None = None,
):
    """Run the Turn (`voice_turn` when the request came by voice), forcing `technique`
    unless it is Auto. Outputs: chat, Thread, Thread picker, then `show()`'s card outputs."""
    user = user_of(request)
    text, images = message.get("text", ""), message.get("files", [])
    if not text.strip() and not images:
        return history, thread_id, gr.skip(), *[gr.skip() for _ in confirmation_card(None)]
    if thread_id is None:
        thread_id = threads.create(text)
    # A pending Confirmation is cancelled by this message; the reply says so.
    forced = None if technique in (None, AUTO) else technique
    outcome = run_turn(user, thread_id, text, images=images, turn_id=voice_turn, technique=forced)
    history, *card = show(outcome, history)
    picker = gr.update(choices=threads.list_recent(), value=thread_id)
    return history, thread_id, picker, *card


def show(outcome: str | Confirmation, history: list[dict]):
    """What a Turn's outcome looks like: the chat, then the Confirmation card's outputs
    (waiting Confirmation, card, summary, each field, note)."""
    if isinstance(outcome, Confirmation):
        note = "This needs your Confirmation: check it below, edit if needed, then approve or reject."
        return [*history, {"role": "assistant", "content": note}], *confirmation_card(outcome)
    # Any charts in the answer show as images.
    return [*history, *charts.chat_messages(outcome)], *confirmation_card(None)


def confirmation_card(confirmation: Confirmation | None) -> list:
    """Outputs: waiting Confirmation, card, summary, each of CARD_FIELDS, note."""
    if confirmation is None:
        hidden = [gr.update(visible=False, value="") for _ in CARD_FIELDS]
        return [None, gr.update(visible=False), "", *hidden, ""]
    fields = confirmation.fields
    boxes = [
        gr.update(visible=name in fields, value=str(fields.get(name, ""))) for name in CARD_FIELDS
    ]
    summary = f"**Perry will do this only if you approve:**\n\n```\n{confirmation.summary}\n```"
    return [confirmation, gr.update(visible=True), summary, *boxes, ""]


def approve(
    request: gr.Request,
    thread_id: str | None,
    waiting: Confirmation | None,
    history: list[dict],
    *values,
):
    """Approve the waiting Write Action with the card's fields (edited or not)."""
    user = user_of(request)
    if thread_id is None or waiting is None:
        return history, *confirmation_card(None)
    edits = {
        name: value
        for name, value in zip(CARD_FIELDS, values, strict=True)
        if name in waiting.fields
    }
    return show(resume_turn(user, thread_id, Approve(edits)), history)


def reject(
    request: gr.Request,
    thread_id: str | None,
    waiting: Confirmation | None,
    history: list[dict],
    note: str,
):
    """Reject the waiting Write Action; the note goes to Perry to revise."""
    user = user_of(request)
    if thread_id is None or waiting is None:
        return history, *confirmation_card(None)
    shown = f"Rejected: {note}" if note.strip() else "Rejected."
    history = [*history, {"role": "user", "content": shown}]
    return show(resume_turn(user, thread_id, Reject(note.strip())), history)


def hear(recording: str | None, history: list[dict], thread_id: str | None):
    """Push-to-talk: transcribe the recording and show it as the user's message.
    Outputs: message, chat, recorder, and the voice Turn (None if nothing is sent)."""
    if not recording:
        return gr.skip(), history, None, None
    turn_id = voice.new_turn()
    try:
        transcript = voice.transcribe(Path(recording).read_bytes(), thread_id, turn_id)
    except usage.DailyCapReached as refused:
        return EMPTY, [*history, {"role": "assistant", "content": str(refused)}], None, None
    except Exception as error:  # noqa: BLE001 - tell the user, keep the UI usable
        gr.Warning(f"Couldn't transcribe the recording: {error}")
        return EMPTY, history, None, None
    if not transcript.strip():
        gr.Warning("I didn't catch anything. Hold the button and speak, then release.")
        return EMPTY, history, None, None
    message = {"text": transcript, "files": []}
    return message, [*history, {"role": "user", "content": transcript}], None, turn_id


def read_aloud(turn_id: str | None, history: list[dict], thread_id: str | None):
    """Speak the latest answer, only when the request came by voice (as Turn `turn_id`)."""
    if turn_id is None or not history or history[-1]["role"] != "assistant":
        return None
    text = _text(history[-1]["content"])
    if not text:
        return None
    try:
        speech = voice.read_aloud(text, thread_id, turn_id)
    except Exception as error:  # noqa: BLE001 - the text answer is already shown
        gr.Warning(f"Couldn't read the answer aloud: {error}")
        return None
    if speech.note:
        gr.Warning(speech.note)
    return speech.audio


def _text(content) -> str:
    """A chat message's text: Gradio passes content as a str or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def open_thread(thread_id: str | None):
    if not thread_id:
        return [], None, *confirmation_card(None)
    history = []
    for message in thread_history(thread_id):
        is_answer = message["role"] == "assistant"
        history += charts.chat_messages(message["content"]) if is_answer else [message]
    return history, thread_id, *confirmation_card(pending_confirmation(thread_id))


def new_thread():
    return [], None, gr.update(value=None), *confirmation_card(None)


def thread_choices():
    """The Thread picker's list as it is now (the page may have been built long ago)."""
    return gr.update(choices=threads.list_recent())


def lock_threads():
    """While a Turn runs, the New thread button and Thread picker are off: a Turn that
    finished after a switch would otherwise write its Thread over the one being shown."""
    return gr.update(interactive=False), gr.update(interactive=False)


def unlock_threads():
    return gr.update(interactive=True), gr.update(interactive=True)


def propose_upload(request: gr.Request, files: list[str] | None):
    """Show what ingesting the uploaded files would cost, and wait for the Owner's OK."""
    owner_of(request)
    if not files:
        return gr.skip(), gr.skip(), gr.skip()
    estimate = ingest.estimate_ingestion([Path(file) for file in files])
    try:
        ingest.check_budget(estimate)
    except ingest.OverBudget as refused:
        return [], str(refused), gr.update(visible=False)
    if not estimate.tokens:
        # Nothing to pay for (e.g. every file is unchanged): no need to ask.
        return [], *ingest_upload(request, files)
    return files, f"{estimate.describe()} Ingest now?", gr.update(visible=True)


def ingest_upload(request: gr.Request, files: list[str]):
    owner_of(request)
    try:
        results = ingest.ingest_files([Path(file) for file in files])
    except ingest.OverBudget as refused:
        return str(refused), gr.update(visible=False)
    # The Documents are searchable now; Summaries follow without holding up the upload.
    summaries.start_background()
    return "\n".join(f"- {r.describe()}" for r in results), gr.update(visible=False)


def cancel_upload():
    return [], "Nothing ingested.", gr.update(visible=False)


def upload_datasets(request: gr.Request, files: list[str] | None) -> tuple[str, str]:
    owner_of(request)
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


def summarising_status() -> str:
    """The sidebar's note on Documents still waiting for their Summary; empty if none.
    Without a local summary model nothing is being summarised, so it says what's needed."""
    pending = summaries.pending_count()
    if not pending:
        return ""
    if not models.summaries_supported():
        return models.summary_model_hint()
    return f"Summarising {pending} Document{'' if pending == 1 else 's'}…"


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


def connect_google(request: gr.Request, kind: str | None):
    owner_of(request)
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


def sidebar_status():
    """Outputs: the Google button and today's usage, both as they are now."""
    return google_button(), todays_usage()


def build(privacy_notice: str = "") -> gr.Blocks:
    """The chat UI. `privacy_notice` (from `privacy.startup()`) shows under the Google button."""
    with gr.Blocks(title="Perry") as demo:
        thread_id = gr.State(None)
        pending = gr.State(EMPTY)
        voice_turn = gr.State(None)  # the Turn id when the latest request came by voice

        with gr.Sidebar():
            gr.Markdown("## Perry")
            who = gr.Markdown()
            gr.Button("Log out", link="/logout", size="sm")
            new_button = gr.Button("New thread", variant="primary")
            thread_picker = gr.Dropdown(
                choices=threads.list_recent(), label="Threads", value=None
            )
            gr.Markdown("### Documents")
            technique = gr.Dropdown(
                choices=technique_choices(),
                value=AUTO,
                label="Retrieval Technique",
                info="Auto lets Perry pick; choose one to force it for the next questions.",
            )
            # Owner-only controls start hidden; owner_controls shows them to the Owner.
            with gr.Column(visible=False) as document_controls:
                uploader = gr.File(
                    file_count="multiple", file_types=[".pdf", ".md", ".txt"], label="Upload"
                )
                upload_status = gr.Markdown()
                with gr.Row(visible=False) as upload_choice:
                    ingest_button = gr.Button("Ingest", variant="primary", size="sm")
                    cancel_button = gr.Button("Cancel", size="sm")
            summarising = gr.Markdown()
            # Summaries are written in the background, so the count is polled.
            summarising_timer = gr.Timer(10)
            uploaded = gr.State([])  # files waiting for the Owner's OK to ingest
            gr.Markdown("### Datasets")
            with gr.Column(visible=False) as dataset_controls:
                dataset_uploader = gr.File(
                    file_count="multiple", file_types=[".csv", ".xlsx"], label="Upload data"
                )
                dataset_status = gr.Markdown()
            datasets_shown = gr.Markdown(dataset_list)
            usage_status = gr.Markdown()
            with gr.Column(visible=False) as workspace_controls:
                gr.Markdown("### Calendar and email")
                account_kind = gr.Radio(ACCOUNT_KINDS, value=None, label="Account")
                google = gr.Button()
                google_status = gr.Markdown()

        chatbot = gr.Chatbot(height="75vh", label="Perry")
        # The Confirmation card: one Write Action, shown exactly as it will be performed.
        waiting = gr.State(None)  # the Confirmation the card shows
        with gr.Group(visible=False) as card:
            card_summary = gr.Markdown()
            card_fields = [
                gr.Textbox(label=label, lines=lines, interactive=editable, visible=False)
                for label, lines, editable in CARD_FIELDS.values()
            ]
            card_note = gr.Textbox(
                label="Note for Perry if you reject (what to change)", lines=2
            )
            with gr.Row():
                approve_button = gr.Button("Approve", variant="primary")
                reject_button = gr.Button("Reject", variant="stop")
        card_outputs = [waiting, card, card_summary, *card_fields, card_note]
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

        status_outputs = [google, usage_status]

        def then_refresh_status(event):
            """After anything that may call a model or the Workspace, refresh the sidebar."""
            return event.then(sidebar_status, None, status_outputs)

        switching = [new_button, thread_picker]

        def during_turn(event):
            """Lock Thread switching from `event` until the Turn's chain ends, then refresh
            the sidebar. `.then` (not `.success`) so a failed Turn still unlocks."""
            return then_refresh_status(event.then(unlock_threads, None, switching))

        answer_outputs = [chatbot, thread_id, thread_picker, *card_outputs]
        during_turn(
            box.submit(lock_threads, None, switching)
            .then(show_user_message, [box, chatbot], [pending, chatbot, box])
            .then(answer, [pending, chatbot, thread_id, technique], answer_outputs)
        )
        box.submit(lambda: None, None, voice_turn)
        during_turn(
            mic.stop_recording(lock_threads, None, switching)
            .then(hear, [mic, chatbot, thread_id], [pending, chatbot, mic, voice_turn])
            .then(answer, [pending, chatbot, thread_id, technique, voice_turn], answer_outputs)
            .then(read_aloud, [voice_turn, chatbot, thread_id], speaker)
        )
        during_turn(
            approve_button.click(lock_threads, None, switching).then(
                approve, [thread_id, waiting, chatbot, *card_fields], [chatbot, *card_outputs]
            )
        )
        during_turn(
            reject_button.click(lock_threads, None, switching).then(
                reject, [thread_id, waiting, chatbot, card_note], [chatbot, *card_outputs]
            )
        )
        thread_picker.input(open_thread, thread_picker, [chatbot, thread_id, *card_outputs])
        new_button.click(new_thread, None, [chatbot, thread_id, thread_picker, *card_outputs])
        then_refresh_status(
            uploader.upload(
                propose_upload, uploader, [uploaded, upload_status, upload_choice]
            ).then(summarising_status, None, summarising)
        )
        then_refresh_status(
            ingest_button.click(
                ingest_upload, uploaded, [upload_status, upload_choice]
            ).then(summarising_status, None, summarising)
        )
        cancel_button.click(cancel_upload, None, [uploaded, upload_status, upload_choice])
        dataset_uploader.upload(
            upload_datasets, dataset_uploader, [dataset_status, datasets_shown]
        )
        demo.load(logged_in_as, None, who, api_name="logged_in_as")
        demo.load(
            owner_controls,
            None,
            [document_controls, dataset_controls, workspace_controls],
            api_name="owner_controls",
        )
        demo.load(sidebar_status, None, status_outputs)
        demo.load(summarising_status, None, summarising)
        summarising_timer.tick(summarising_status, None, summarising, show_progress="hidden")
        demo.load(thread_choices, None, thread_picker)
        demo.load(lambda: privacy_notice, None, google_status)
        google.click(connect_google, account_kind, [google, google_status])
    return demo


if __name__ == "__main__":
    # The chat UI is only served behind the login pages.
    from perry import web

    web.main()
