"""Turns with Write Actions: each one waits for a Confirmation, against the Workspace fake."""

from datetime import datetime, timedelta

import pytest
from fakes import call, last_message, with_usage
from logins import request_as

from perry.supervisor import Approve, Confirmation, Reject
from perry.workspace import InMemoryWorkspace


def propose_email_to_sam(perry):
    perry.script(
        call("ask_task_agent", request="Email Sam that I'm running 10 minutes late."),
        call(
            "send_email",
            to="sam@example.com",
            subject="Running late",
            body="Running 10 minutes late, start without me.",
        ),
    )


def test_email_waits_for_a_confirmation_and_approving_sends_it_once(perry, workspace):
    propose_email_to_sam(perry)

    pending = perry.send("Tell Sam I'm running 10 minutes late")

    assert isinstance(pending, Confirmation)
    assert pending.action == "send_email"
    assert pending.fields == {
        "to": "sam@example.com",
        "subject": "Running late",
        "body": "Running 10 minutes late, start without me.",
    }
    assert "sam@example.com" in pending.summary and "Running late" in pending.summary
    assert workspace.sent == []  # nothing happens before the Confirmation

    perry.script(last_message, last_message)  # Task Agent reports, Supervisor passes it on
    reply = perry.resume(Approve())

    assert [(e.to, e.subject) for e in workspace.sent] == [(("sam@example.com",), "Running late")]
    assert "Sent" in reply


def test_approving_with_edits_performs_the_edited_fields(perry, workspace):
    propose_email_to_sam(perry)
    perry.send("Tell Sam I'm running 10 minutes late")

    perry.script(last_message, last_message)
    perry.resume(
        Approve(edits={"to": "sam@example.com, priya@example.com", "body": "15 minutes late!"})
    )

    [sent] = workspace.sent
    assert sent.to == ("sam@example.com", "priya@example.com")
    assert sent.subject == "Running late"
    assert sent.body == "15 minutes late!"


def test_rejecting_with_a_note_performs_nothing_and_the_supervisor_revises(perry, workspace):
    propose_email_to_sam(perry)
    perry.send("Tell Sam I'm running 10 minutes late")

    def revise_with_the_note(messages):
        assert "say I'll bring the slides" in messages[-1].text  # the note reached the Supervisor
        return call("ask_task_agent", request="Email Sam: running late, I'll bring the slides.")

    perry.script(
        last_message,  # Task Agent reports the rejection and the note
        revise_with_the_note,  # Supervisor asks again, revised
        call(
            "send_email",
            to="sam@example.com",
            subject="Running late",
            body="Running 10 minutes late. I'll bring the slides.",
        ),
    )
    revised = perry.resume(Reject("Also say I'll bring the slides"))

    assert workspace.sent == []
    assert isinstance(revised, Confirmation)
    assert revised.fields["body"] == "Running 10 minutes late. I'll bring the slides."
    assert perry.consulted() == ["ask_task_agent"]  # the first ask, closed by the rejection


def every_tool_call_answered(messages) -> bool:
    answered = {m.tool_call_id for m in messages if m.type == "tool"}
    return all(c["id"] in answered for m in messages if m.type == "ai" for c in m.tool_calls)


def test_new_message_cancels_the_pending_confirmation_and_perry_says_so(perry, workspace):
    propose_email_to_sam(perry)
    perry.send("Tell Sam I'm running 10 minutes late")

    def answer_the_new_message(messages):
        # The model is never shown a tool call without its result (Gemini refuses that).
        assert every_tool_call_answered(messages)
        return "Hello! How can I help?"

    perry.script(answer_the_new_message)
    reply = perry.send("Actually, forget that. Hello!")

    assert "cancelled" in reply.lower() and "Send an email" in reply
    assert reply.endswith("Hello! How can I help?")
    assert workspace.sent == []

    later = perry.resume(Approve())  # e.g. an old Confirmation card clicked hours later

    assert workspace.sent == []
    assert "nothing waiting" in later.lower()


class TimingOutWorkspace(InMemoryWorkspace):
    """Every send times out (a failure that is retried for Read Actions)."""

    attempts: int = 0

    def send_email(self, to, subject, body):
        self.attempts += 1
        raise TimeoutError("Gmail did not answer in time")


def test_failing_write_action_is_not_retried_and_is_reported(perry):
    from perry import workspace

    failing = TimingOutWorkspace()
    propose_email_to_sam(perry)
    with workspace.use(failing):
        perry.send("Tell Sam I'm running 10 minutes late")
        perry.script(last_message, last_message)
        reply = perry.resume(Approve())

    assert failing.attempts == 1
    assert "FAILED" in reply and "Gmail did not answer in time" in reply
    assert "not retried" in reply


def test_task_agent_failing_after_a_write_action_is_not_rerun(perry, workspace):
    from perry import models

    propose_email_to_sam(perry)
    perry.send("Tell Sam I'm running 10 minutes late")
    perry.script(
        models.RateLimitError("429 RESOURCE_EXHAUSTED"),  # Task Agent, after sending
        last_message,  # Supervisor passes on the failure report
    )

    reply = perry.resume(Approve())

    assert len(workspace.sent) == 1  # a retry would have re-run the Task Agent
    assert "failed once" in reply
    assert "no result" not in reply  # the email was in fact sent
    assert "outcome is unknown" in reply
    assert "may already have been performed" in reply


def test_chained_turn_pauses_at_the_write_action_in_the_middle(perry, workspace):
    perry.add_document("q3-notes.md", "Q3 notes: revenue grew 12%. Hiring freeze until January.")

    def email_the_summary(messages):
        assert "revenue grew 12%" in messages[-1].text
        return call(
            "ask_task_agent",
            request="Email sam@example.com a summary: revenue grew 12%, hiring freeze until January.",
        )

    perry.script(
        call("ask_knowledge_agent", question="What do the Q3 notes say?"),  # Supervisor
        call("hybrid_search", question="Q3 notes"),  # Knowledge Agent
        last_message,  # Knowledge Agent reports
        email_the_summary,  # Supervisor chains to the Task Agent
        call(
            "send_email",
            to="sam@example.com",
            subject="Q3 summary",
            body="Revenue grew 12%. Hiring freeze until January. [q3-notes.md]",
        ),
    )

    pending = perry.send("Find my Q3 notes and email a summary to Sam")

    assert isinstance(pending, Confirmation)
    assert "Revenue grew 12%" in pending.summary
    assert workspace.sent == []

    # The Supervisor's answer drops the Citation; it comes back all the same, at the end.
    perry.script(last_message, "Done: I emailed Sam the Q3 summary.")
    reply = perry.resume(Approve())

    assert [e.subject for e in workspace.sent] == ["Q3 summary"]
    assert perry.consulted() == ["ask_knowledge_agent", "ask_task_agent"]
    assert reply == (
        "Done: I emailed Sam the Q3 summary.\n\nCitations: [q3-notes.md]\n"
        "Found with: Hybrid search (picked logically)"
    )
    assert perry.history()[-1] == {"role": "assistant", "content": reply}


def test_confirmation_card_shows_the_write_actions_fields_and_approves_the_edited_ones(
    perry, workspace
):
    from perry import app

    app.build()  # the card and its wiring construct
    propose_email_to_sam(perry)
    message = {"text": "Tell Sam I'm running 10 minutes late", "files": []}
    owner = request_as(perry.user)
    history, thread_id, _, waiting, card, summary, *rest = app.answer(owner, message, [], None)
    boxes = dict(zip(app.CARD_FIELDS, rest[:-1], strict=True))

    assert card["visible"] and "Running late" in summary
    assert [name for name, box in boxes.items() if box["visible"]] == ["to", "subject", "body"]
    assert boxes["subject"]["value"] == "Running late"

    values = [box["value"] for box in boxes.values()]
    values[list(app.CARD_FIELDS).index("subject")] = "Late, sorry"
    perry.script(last_message, "Sent it.")
    history, waiting, card, *_ = app.approve(owner, thread_id, waiting, history, *values)

    assert [e.subject for e in workspace.sent] == ["Late, sorry"]
    assert waiting is None and not card["visible"]
    assert history[-1] == {"role": "assistant", "content": "Sent it."}


def test_moving_an_event_shows_which_event_and_moves_it_once_approved(perry, workspace):
    from perry.workspace import Event

    start = datetime(2026, 10, 9, 10).astimezone()
    workspace.events.append(Event("Dentist", start, start + timedelta(hours=1), id="e1"))
    perry.script(
        call("ask_task_agent", request="Move my dentist appointment tomorrow to 15:00."),
        call("move_event", event_id="e1", start="2026-10-09T15:00", end="2026-10-09T16:00"),
    )

    pending = perry.send("Move the dentist to 3pm")

    assert "Dentist" in pending.summary and "2026-10-09T15:00" in pending.summary
    assert workspace.events[0].start.hour == 10

    perry.script(last_message, last_message)
    perry.resume(Approve())

    [moved] = workspace.events
    assert (moved.title, moved.start.hour, moved.end.hour) == ("Dentist", 15, 16)


def test_replying_and_creating_and_deleting_each_wait_for_their_own_confirmation(
    perry, workspace
):
    from perry.workspace import Email, Event

    start = datetime(2026, 10, 9, 18).astimezone()
    workspace.events.append(Event("Yoga", start, start + timedelta(hours=1), id="e1"))
    workspace.emails.append(
        Email("Priya <priya@example.com>", "Lunch?", "Noodles?", start, ("me@example.com",), "m1", "t1")
    )
    perry.script(
        call("ask_task_agent", request="Reply yes to Priya, add lunch, cancel yoga."),
        call("reply_email", email_thread_id="t1", body="Yes, noodles!"),
    )
    actions = [perry.send("Say yes to Priya's lunch, put it in my calendar, and drop yoga")]
    perry.script(
        call(
            "create_event",
            title="Lunch with Priya",
            start="2026-10-09T12:30",
            end="2026-10-09T13:30",
            attendees="priya@example.com",
        )
    )
    actions.append(perry.resume(Approve()))
    perry.script(call("delete_event", event_id="e1"))
    actions.append(perry.resume(Approve()))
    perry.script(last_message, last_message)
    perry.resume(Approve())

    assert [a.action for a in actions] == ["reply_email", "create_event", "delete_event"]
    assert "'Lunch?' from Priya" in actions[0].summary
    assert "Invite: priya@example.com" in actions[1].summary
    assert "Yoga" in actions[2].summary
    assert [(e.subject, e.to) for e in workspace.sent] == [("Re: Lunch?", ("Priya <priya@example.com>",))]
    assert [(e.title, e.attendees) for e in workspace.events] == [
        ("Lunch with Priya", ("priya@example.com",))
    ]


@pytest.mark.parametrize(
    ("steps_after", "answered"), [(1, True), (2, False)], ids=["within_cap", "past_cap"]
)
def test_steps_before_a_confirmation_count_toward_the_turns_step_cap(
    perry, workspace, steps_after, answered
):
    from perry.supervisor import MAX_STEPS_PER_TURN

    steps_before = MAX_STEPS_PER_TURN - 1
    for _ in range(steps_before - 1):
        perry.script(call("ask_knowledge_agent", question="Anything on Sam?"), "Nothing.")
    propose_email_to_sam(perry)  # the last step before the Confirmation
    assert isinstance(perry.send("Look around, then tell Sam I'm late"), Confirmation)

    perry.script(last_message)  # Task Agent reports the email was sent
    for _ in range(steps_after):
        perry.script(call("ask_knowledge_agent", question="Anything else?"), "Nothing.")
    perry.script("Done.")
    reply = perry.resume(Approve())

    if answered:
        assert reply == "Done."
    else:
        assert reply.startswith(f"I stopped after {MAX_STEPS_PER_TURN} steps without finishing.")
    assert len(workspace.sent) == 1


def test_tokens_before_a_confirmation_count_toward_the_turns_token_ceiling(
    perry, workspace, monkeypatch
):
    monkeypatch.setenv("PERRY_TURN_TOKEN_CEILING", "1000")
    perry.script(
        with_usage(call("ask_task_agent", request="Email Sam that I'm late."), 900),
        call("send_email", to="sam@example.com", subject="Late", body="Running late."),
    )
    assert isinstance(perry.send("Tell Sam I'm late"), Confirmation)

    perry.script(with_usage(last_message, 200), "Sent.")  # 1,100 tokens in this Turn
    reply = perry.resume(Approve())

    assert reply.startswith("I stopped because this Turn reached its limit of 1,000 tokens.")
    assert len(workspace.sent) == 1
