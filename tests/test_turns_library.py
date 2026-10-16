"""The Supervisor is told what the library holds (each Document with its topic, each
Dataset with its columns), so it can tell a question about the user's notes from one
about their calendar or their data."""

from fakes import FakeSummaryModel, fake_provider

from perry import models


def supervisor_prompt(perry) -> str:
    first = perry.model.calls[0][0]
    assert first.type == "system"
    return first.text


def test_the_supervisor_is_given_each_document_with_its_topic(perry):
    from perry import summaries

    topic = '{"summary": "Plans for the team offsite.", "topic": "Team offsite in Annecy"}'
    with models.use(fake_provider(perry.model, summary=FakeSummaryModel(replies=[topic]))):
        perry.add_document("team-offsite.md", "The team offsite is at Lakeview Lodge.")
        summaries.summarise_pending()
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    perry.script("Hi!")

    perry.send("Hello")

    prompt = supervisor_prompt(perry)
    assert "team-offsite.md: Team offsite in Annecy" in prompt
    assert "budget.md" in prompt  # not summarised yet: its name alone


def test_the_supervisor_is_given_each_dataset_with_its_columns(perry, tmp_path):
    from perry import datasets

    sales = tmp_path / "sales.csv"
    sales.write_text("month,region,revenue\n2025-01,North,100\n", encoding="utf-8")
    datasets.upload(sales)
    perry.script("Hi!")

    perry.send("Hello")

    prompt = supervisor_prompt(perry)
    assert "sales" in prompt
    assert "month, region, revenue" in prompt


def test_an_empty_library_is_said_to_be_empty(perry):
    perry.script("Hi!")

    perry.send("Hello")

    prompt = supervisor_prompt(perry)
    assert "no Documents yet" in prompt
    assert "no Datasets yet" in prompt


def test_a_large_library_lists_the_first_documents_and_counts_the_rest(perry, monkeypatch):
    from perry import supervisor

    monkeypatch.setattr(supervisor, "LIBRARY_LISTED", 2)
    for name in ("a.md", "b.md", "c.md"):
        perry.add_document(name, f"Notes {name}")
    perry.script("Hi!")

    perry.send("Hello")

    prompt = supervisor_prompt(perry)
    assert "a.md" in prompt and "b.md" in prompt
    assert "c.md" not in prompt
    assert "and 1 more Document" in prompt
