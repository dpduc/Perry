"""`python -m perry.library`: list, delete, replace and rename Documents from the command
line. Names are matched forgivingly, and nothing changes until the user says yes."""

import unicodedata

import pytest
from fakes import ScriptedChatModel, fake_embed, fake_provider

from perry import library, models
from perry.ingest import ingest_file
from perry.search import hybrid_search


@pytest.fixture
def documents(database, tmp_path):
    folder = tmp_path / "documents"
    folder.mkdir()

    def write(name: str, text: str, *, ingest: bool = True):
        path = folder / name
        path.write_text(text, encoding="utf-8")
        if ingest:
            ingest_file(path, embed=fake_embed)
        return path

    with models.use(fake_provider(ScriptedChatModel())):
        write("report.md", "The Kestrel launch is in March.")
        write("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")
        yield write


def run(*argv: str, answer: str = "y", asked: list[str] | None = None) -> int:
    def ask(question: str) -> str:
        if asked is not None:
            asked.append(question)
        return answer

    return library.main(list(argv), ask=ask)


def names() -> set[str]:
    return {name for name, *_ in library.documents()}


def cited(question: str) -> str:
    from fakes import fake_embed_one

    return hybrid_search(question, embed=fake_embed_one)[0].citation


def test_list_shows_each_document_with_its_chunks(documents, capsys):
    assert run("list") == 0

    out = capsys.readouterr().out
    assert "report.md" in out and "hiking.md" in out
    assert "1 Chunk" in out


def test_delete_asks_first_then_deletes(documents, capsys):
    asked = []

    assert run("delete", "report.md", asked=asked) == 0

    assert "Delete report.md (1 Chunk" in asked[0]
    assert "Deleted report.md" in capsys.readouterr().out
    assert names() == {"hiking.md"}


def test_answering_no_deletes_nothing(documents, capsys):
    assert run("delete", "report.md", answer="n") == 1

    assert "Nothing was deleted." in capsys.readouterr().out
    assert names() == {"report.md", "hiking.md"}


def test_yes_skips_the_question(documents):
    asked = []

    assert run("delete", "report.md", "--yes", asked=asked) == 0

    assert asked == []
    assert names() == {"hiking.md"}


def test_a_name_is_matched_ignoring_case_and_how_its_accents_were_typed(documents):
    documents("Báo cáo quý 3.md", "Doanh thu quý 3 tăng mười phần trăm.")
    # Typed in another case, with the accents as separate marks (as some keyboards do).
    typed = unicodedata.normalize("NFD", "BÁO CÁO QUÝ 3.md")

    assert run("delete", typed) == 0

    assert names() == {"report.md", "hiking.md"}


def test_an_unknown_name_suggests_close_ones_and_changes_nothing(documents, capsys):
    assert run("delete", "reprot.md") == 1

    err = capsys.readouterr().err
    assert "No Document is called 'reprot.md'" in err
    assert "Did you mean report.md?" in err
    assert names() == {"report.md", "hiking.md"}


def test_replace_puts_a_newer_version_in_place_of_the_old_one(documents, capsys):
    newer = documents("report-v2.md", "The Kestrel launch moved to June.", ingest=False)

    with models.use(fake_provider(ScriptedChatModel())):
        assert run("replace", "report.md", str(newer), "--yes") == 0

    assert "report-v2.md replaced report.md" in capsys.readouterr().out
    assert names() == {"report-v2.md", "hiking.md"}
    assert cited("When is the Kestrel launch?") == "report-v2.md"


def test_replace_with_a_file_that_is_not_there_changes_nothing(documents, capsys, tmp_path):
    assert run("replace", "report.md", str(tmp_path / "missing.md"), "--yes") == 1

    assert "missing.md" in capsys.readouterr().err
    assert names() == {"report.md", "hiking.md"}


def test_rename_changes_how_the_document_is_cited(documents, capsys):
    assert run("rename", "report.md", "kestrel-plan.md") == 0

    assert "Renamed report.md to kestrel-plan.md" in capsys.readouterr().out
    assert cited("When is the Kestrel launch?") == "kestrel-plan.md"


def test_rename_onto_a_name_in_use_is_refused(documents, capsys):
    assert run("rename", "report.md", "hiking.md") == 1

    assert "already called hiking.md" in capsys.readouterr().err
    assert names() == {"report.md", "hiking.md"}


def test_part_of_a_name_suggests_the_documents_it_is_part_of_quoted(documents, capsys):
    documents("Báo cáo quý 3.md", "Doanh thu quý 3 tăng mười phần trăm.")

    assert run("delete", "BÁO_CÁO") == 1

    assert 'Did you mean "Báo cáo quý 3.md"?' in capsys.readouterr().err
    assert "Báo cáo quý 3.md" in names()


def test_a_name_that_differs_by_more_than_case_is_only_suggested(documents, capsys):
    documents("report-v2.md", "The Kestrel launch moved to June.")

    assert run("delete", "report_v2.md", "--yes") == 1

    assert "Did you mean report-v2.md" in capsys.readouterr().err  # the closest first
    assert "report-v2.md" in names()
