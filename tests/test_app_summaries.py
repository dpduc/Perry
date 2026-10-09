"""The sidebar's note on Documents still being summarised in the background."""

from fakes import FakeSummaryModel, fake_provider

from perry import models


def test_the_sidebar_shows_how_many_documents_are_still_being_summarised(perry):
    from perry import app, summaries

    assert app.summarising_status() == ""

    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    assert app.summarising_status() == "Summarising 1 Document…"
    perry.add_document("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")
    assert app.summarising_status() == "Summarising 2 Documents…"

    with models.use(fake_provider(perry.model, summary=FakeSummaryModel())):
        summaries.summarise_pending()

    assert app.summarising_status() == ""


def test_without_a_local_summary_model_documents_stay_pending(perry):
    from perry import summaries

    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")

    summaries.summarise_pending()  # the fake provider has no summary model

    assert summaries.document("budget.md").status == summaries.PENDING
