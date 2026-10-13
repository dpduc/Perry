"""Only the Owner uploads Documents and Datasets and connects the Workspace (ADR-0004):
the controls show only for the Owner, and the handlers refuse anyone else."""

from pathlib import Path

import gradio as gr
import pytest
from logins import request_as

from perry import app, datasets, google_auth, ingest


@pytest.fixture
def sam(database):
    from perry import users

    return users.create_user("sam", "sam-password-1", "Sam", "UTC")


@pytest.fixture
def notes(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("The marketing budget for Q3 is 40,000 dollars.", encoding="utf-8")
    return [str(path)]


@pytest.fixture
def sales(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text("region,total\nNorth,10\nSouth,20\n", encoding="utf-8")
    return [str(path)]


@pytest.fixture
def connects(monkeypatch):
    """Records each Google connect instead of opening the browser for consent."""
    calls = []
    monkeypatch.setattr(google_auth, "connect", calls.append)
    return calls


def visible(updates) -> list[bool]:
    return [update["visible"] for update in updates]


def test_the_owner_sees_the_upload_and_workspace_controls(owner):
    assert visible(app.owner_controls(request_as(owner))) == [True] * len(app.OWNER_CONTROLS)


def test_other_users_dont_see_the_upload_and_workspace_controls(sam):
    assert visible(app.owner_controls(request_as(sam))) == [False] * len(app.OWNER_CONTROLS)


def test_a_non_owner_cant_propose_or_ingest_a_document_upload(sam, notes):
    with pytest.raises(gr.Error, match="Only the Owner"):
        app.propose_upload(request_as(sam), notes)
    with pytest.raises(gr.Error, match="Only the Owner"):
        app.ingest_upload(request_as(sam), notes)

    # Still new to Perry: nothing was ingested.
    assert ingest.estimate_ingestion([Path(file) for file in notes]).tokens > 0


def test_a_non_owner_cant_upload_a_dataset(sam, sales):
    with pytest.raises(gr.Error, match="Only the Owner"):
        app.upload_datasets(request_as(sam), sales)

    assert datasets.list_datasets() == []


def test_the_owner_uploads_a_dataset(owner, sales):
    _, shown = app.upload_datasets(request_as(owner), sales)

    assert "sales" in shown
    assert [d.name for d in datasets.list_datasets()] == ["sales"]


def test_a_non_owner_cant_connect_the_workspace(sam, connects):
    with pytest.raises(gr.Error, match="Only the Owner"):
        app.connect_google(request_as(sam), "test")

    assert connects == []


def test_the_owner_connects_the_workspace(owner, connects):
    _, message = app.connect_google(request_as(owner), "test")

    assert connects == ["test"]
    assert message == "Google connected (test account)."
