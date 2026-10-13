"""Voice: push-to-talk Turns, with fake speech-to-text and text-to-speech."""

import pytest
from fakes import call, spoken, with_usage
from logins import request_as


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

    message, history, _, voice_turn = app.hear(request_as(perry.user), str(recording), [], perry.thread_id)

    assert message == {"text": "", "files": []}  # nothing is sent as a Turn
    assert history == [
        {
            "role": "assistant",
            "content": "Perry has used today's budget of $0.01, so it can't take new "
            "requests until the budget resets at midnight.",
        }
    ]
    assert voice_turn is None  # so the refusal isn't read aloud
    assert usage.today().tokens == used  # no speech-to-text call was made


def test_speech_to_text_counts_toward_the_voice_turns_token_ceiling(perry, monkeypatch):
    monkeypatch.setenv("PERRY_TURN_TOKEN_CEILING", "1000")
    # 50 tokens to transcribe, then 960 here: the Knowledge Agent's call is over the ceiling.
    perry.script(with_usage(call("ask_knowledge_agent", question="Q3 budget?"), 960))

    reply = perry.say(spoken("How big is my Q3 budget?"))

    assert reply.text.startswith("I stopped because this Turn reached its limit of 1,000 tokens.")


def test_answer_is_not_read_aloud_past_the_voice_turns_token_ceiling(perry, monkeypatch):
    monkeypatch.setenv("PERRY_TURN_TOKEN_CEILING", "1000")
    perry.script(with_usage("Hi! How can I help?", 960))  # with transcribing: 1,010 tokens

    reply = perry.say(spoken("Hello"))

    assert reply.text == "Hi! How can I help?"
    assert reply.audio is None
    assert "limit of 1,000 tokens" in reply.note


def test_answer_is_not_read_aloud_once_the_turn_reaches_the_daily_cap(
    perry, prices, monkeypatch
):
    from perry import usage

    prices({"scripted": {"input": 1.0, "output": 1.0}})
    monkeypatch.setenv("PERRY_DAILY_CAP_USD", "0.01")
    perry.script(with_usage("Hi! How can I help?", 10_000))  # $0.01: the whole cap

    reply = perry.say(spoken("Hello"))

    assert reply.text == "Hi! How can I help?"  # the text answer is kept
    assert reply.audio is None
    assert "today's budget" in reply.note
    assert usage.today("text_to_speech").tokens == 0


def test_the_gemini_text_to_speech_request_carries_the_configured_voice():
    """Builds the real library request offline: the fakes elsewhere can't see whether
    langchain-google-genai accepts how the voice is passed."""
    from langchain_core.messages import HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    from perry import models

    tts = ChatGoogleGenerativeAI(model=models.TTS_MODEL, api_key="offline")
    request = tts._prepare_request([HumanMessage("Hello.")], **models.tts_options())

    voice = request["config"].speech_config.voice_config.prebuilt_voice_config
    assert voice.voice_name == models.TTS_VOICE
