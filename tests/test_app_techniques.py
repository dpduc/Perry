"""The sidebar's Retrieval Technique dropdown: Auto, or force one technique for the next Turns."""

from fakes import call, last_message
from logins import request_as


def test_the_dropdown_lists_auto_then_every_registered_technique():
    from perry import app
    from perry.techniques import TECHNIQUES

    labels = [label for label, _value in app.technique_choices()]

    assert labels[0] == "Auto"
    assert "Hybrid search" in labels
    assert len(labels) == 1 + len(TECHNIQUES)


def test_the_technique_chosen_in_the_dropdown_is_forced_for_the_next_turns(perry, owner):
    from perry import app

    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    chosen = dict(app.technique_choices())["Hybrid search"]
    history, thread_id = [], None
    for question in ("What is my Q3 budget?", "And the Q3 marketing budget?"):
        perry.script(
            call("ask_knowledge_agent", question=question),
            call("hybrid_search", question="Q3 marketing budget"),
            last_message,
            last_message,
        )
        history, thread_id, *_ = app.answer(
            request_as(owner), {"text": question, "files": []}, history, thread_id, chosen
        )

        assert history[-1]["content"].endswith("Found with: Hybrid search (forced)")


def test_auto_lets_the_knowledge_agent_pick(perry, owner):
    from perry import app

    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    perry.script(
        call("ask_knowledge_agent", question="What is my Q3 budget?"),
        call("hybrid_search", question="Q3 marketing budget"),
        last_message,
        last_message,
    )

    history, *_ = app.answer(
        request_as(owner), {"text": "What is my Q3 budget?", "files": []}, [], None, app.AUTO
    )

    assert history[-1]["content"].endswith("Found with: Hybrid search (picked logically)")


def test_the_dropdown_is_in_the_ui_and_feeds_every_typed_and_spoken_turn():
    from perry import app

    demo = app.build()
    dropdowns = [b for b in demo.blocks.values() if getattr(b, "label", None) == "Retrieval Technique"]

    assert len(dropdowns) == 1
    feeds = [fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == "answer"]
    assert len(feeds) == 2  # typed and push-to-talk
    assert all(dropdowns[0]._id in [b._id for b in fn.inputs] for fn in feeds)
