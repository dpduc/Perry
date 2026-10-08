"""Push-to-talk: a voice request is one Turn, from transcribing it to reading the answer aloud.

The UI makes a Turn id with `new_turn()` when the recording arrives, then passes it to
`transcribe`, `supervisor.run_turn` and `read_aloud`. So speech-to-text and
text-to-speech are attributed to that Turn in the usage ledger and count toward its
token ceiling. Both wait for the daily cap: transcribing is refused over it, and the
answer is then shown but not read aloud.
"""

from dataclasses import dataclass

from perry import models, usage

MIME_TYPE = "audio/wav"


@dataclass(frozen=True)
class Speech:
    """The answer read aloud, or None with a note saying why it wasn't."""

    audio: bytes | None
    note: str = ""


def new_turn() -> str:
    """The id of a new voice Turn."""
    return usage.new_turn_id()


def transcribe(audio: bytes, thread_id: str | None, turn_id: str) -> str:
    """What the user said in `audio`. Raises usage.DailyCapReached, before any model call,
    once today's usage has reached the daily cap."""
    usage.check_daily_cap()
    with usage.turn(thread_id, turn_id):
        return models.transcribe(audio, MIME_TYPE)


def read_aloud(text: str, thread_id: str | None, turn_id: str) -> Speech:
    """`text` as speech, unless the daily cap or the Turn's token ceiling stops it."""
    if refused := usage.daily_cap_refusal():
        return Speech(None, f"I didn't read the answer aloud. {refused}")
    with usage.turn(thread_id, turn_id):
        try:
            return Speech(models.synthesize(text))
        except usage.TurnTokenCeilingReached as ceiling:
            return Speech(
                None,
                "I didn't read the answer aloud: this Turn reached its limit of "
                f"{ceiling.ceiling:,} tokens.",
            )
