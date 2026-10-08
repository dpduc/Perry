"""Images in chat: dropped into a Turn, they reach the model as multimodal input."""

import base64

import pytest
from fakes import with_usage

# A 1x1 PNG.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def screenshot(tmp_path):
    path = tmp_path / "error.png"
    path.write_bytes(PNG)
    return path


def image_blocks(messages) -> list[dict]:
    """The image content blocks in the messages a model call was given."""
    return [
        block
        for message in messages
        if isinstance(message.content, list)
        for block in message.content
        if isinstance(block, dict) and block.get("type") == "image"
    ]


def test_image_dropped_into_the_chat_reaches_the_model_as_multimodal_input(perry, screenshot):
    perry.script("The screenshot shows a single pixel.")

    reply = perry.send("What does this screenshot show?", images=[screenshot])

    assert reply == "The screenshot shows a single pixel."
    (image,) = image_blocks(perry.model.calls[-1])
    assert image["mime_type"] == "image/png"
    assert base64.b64decode(image["base64"]) == PNG
    assert perry.model.calls[-1][-1].text == "What does this screenshot show?"


def test_model_calls_given_an_image_are_recorded_as_vision_in_the_usage_ledger(
    perry, screenshot
):
    from perry import usage

    perry.script(with_usage("Hello!", 50, 5))
    perry.send("Hi")  # no image: not a vision call
    perry.script(with_usage("A single pixel.", 1000, 10))

    perry.send("What does this screenshot show?", images=[screenshot])

    assert usage.today(kind="vision").tokens == 1010


def test_image_stays_in_the_thread_so_follow_up_questions_can_use_it(perry, screenshot):
    perry.script("A single pixel.")
    perry.send("What does this screenshot show?", images=[screenshot])
    perry.script("It is black.")

    reply = perry.send("What colour is it?")

    assert reply == "It is black."
    (image,) = image_blocks(perry.model.calls[-1])
    assert base64.b64decode(image["base64"]) == PNG
    assert perry.history() == [
        {"role": "user", "content": "What does this screenshot show?"},
        {"role": "assistant", "content": "A single pixel."},
        {"role": "user", "content": "What colour is it?"},
        {"role": "assistant", "content": "It is black."},
    ]
