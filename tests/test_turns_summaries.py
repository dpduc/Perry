"""The Document Summaries technique seen through Turns."""

from fakes import FakeSummaryModel, call, fake_provider, last_message

from perry import models, summaries


def test_document_summaries_answers_with_page_citations_and_names_itself(perry):
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    perry.add_document("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")
    with models.use(fake_provider(perry.model, summary=FakeSummaryModel())):
        summaries.summarise_pending()
    perry.script(
        call("ask_knowledge_agent", question="Which document has the Q3 marketing budget?"),
        call("document_summaries", question="Q3 marketing budget"),
        last_message,  # Knowledge Agent reports the Chunks it found
        last_message,  # Supervisor passes the findings on
    )

    reply = perry.send("Which file has my Q3 marketing budget?")

    assert reply.startswith("The marketing budget for Q3 is 40,000 dollars.")
    assert "Summary:" not in reply  # a Summary is never shown or cited
    *_, citations, found_with = reply.splitlines()
    assert citations.startswith("Citations: [budget.md]")
    assert found_with == "Found with: Document Summaries (picked logically)"


def test_document_summaries_finds_nothing_while_summaries_are_pending(perry):
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    perry.script(
        call("ask_knowledge_agent", question="Which document has the Q3 marketing budget?"),
        call("document_summaries", question="Q3 marketing budget"),
        last_message,
        last_message,
    )

    reply = perry.send("Which file has my Q3 marketing budget?", technique="document_summaries")

    assert "40,000" not in reply
    assert "No passages found" in reply


def test_forced_raptor_answers_with_page_citations_and_names_itself(perry):
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    perry.add_document("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")
    with models.use(fake_provider(perry.model, summary=FakeSummaryModel())):
        summaries.summarise_pending()  # Summaries, then the RAPTOR trees
    perry.script(
        call("ask_knowledge_agent", question="What do my documents say about the marketing budget?"),
        call("raptor", question="marketing budget overall"),
        last_message,  # Knowledge Agent reports the Chunks it found
        last_message,  # Supervisor passes the findings on
    )

    reply = perry.send("Overall, what about the marketing budget?", technique="raptor")

    assert reply.startswith("The marketing budget for Q3 is 40,000 dollars.")
    assert "Summary:" not in reply  # a node's Summary is never shown or cited
    *_, citations, found_with = reply.splitlines()
    assert citations.startswith("Citations: [budget.md]")
    assert found_with == "Found with: RAPTOR (forced)"
