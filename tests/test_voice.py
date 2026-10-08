"""Voice: push-to-talk Turns, with fake speech-to-text and text-to-speech."""

import pytest
from fakes import spoken, with_usage


def test_voice_request_is_shown_as_the_users_message_and_answered_in_text_and_audio(perry):
    perry.script("Hi! I'm Perry. How can I help?")

    reply = perry.say(spoken("Hello there"))

    assert reply.transcript == "Hello there"
    assert reply.text == "Hi! I'm Perry. How can I help?"
    assert reply.audio == spoken("Hi! I'm Perry. How can I help?")
    assert perry.history() == [
        {"role": "user", "content": "Hello there"},
        {"role": "assistant", "content": "Hi! I'm Perry. How can I help?"},
    ]


def test_speech_to_text_and_text_to_speech_count_in_todays_usage(perry, prices):
    from perry import usage

    prices(
        {
            "scripted": {"input": 1.0, "output": 2.0},
            "fake-stt": {"input": 3.0, "output": 4.0},
            "fake-tts": {"input": 5.0, "output": 6.0},
        }
    )
    perry.script(with_usage("Hi!", 100, 10))

    # The fakes report 40 + 10 tokens per transcription and 10 + 200 per synthesis.
    perry.say(spoken("Hello"))

    today = usage.today()
    assert today.tokens == 110 + 50 + 210
    assert today.cost_usd == pytest.approx(
        (100 * 1 + 10 * 2 + 40 * 3 + 10 * 4 + 10 * 5 + 200 * 6) * 1e-6
    )


def test_a_provider_without_speech_refuses_voice_requests(database):
    from fakes import ScriptedChatModel, fake_provider

    from perry import models

    with models.use(fake_provider(ScriptedChatModel(), speech=False)):
        assert not models.speech_supported()
        with pytest.raises(models.SpeechUnavailable):
            models.transcribe(spoken("Hello"), "audio/wav")


def test_voice_request_after_the_daily_cap_is_refused_without_transcribing(
    perry, prices, monkeypatch, tmp_path
):
    from perry import app, usage

    prices({"scripted": {"input": 1.0, "output": 1.0}, "fake-stt": {"input": 1.0}})
    monkeypatch.setenv("PERRY_DAILY_CAP_USD", "0.01")
    perry.script(with_usage("Hi! How can I help?", 10_000))  # $0.01: the whole cap
    perry.send("Hello")
    used = usage.today().tokens
    recording = tmp_path / "request.wav"
    recording.write_bytes(spoken("How big is my Q3 marketing budget?"))

    message, history, _, read_aloud = app.hear(str(recording), [])

    assert message == {"text": "", "files": []}  # nothing is sent as a Turn
    assert history == [
        {
            "role": "assistant",
            "content": "Perry has used today's budget of $0.01, so it can't take new "
            "requests until the budget resets at midnight.",
        }
    ]
    assert not read_aloud
    assert usage.today().tokens == used  # no speech-to-text call was made
