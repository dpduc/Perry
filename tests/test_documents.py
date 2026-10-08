from perry.documents import citation, parse, split_text


def test_pdf_citations_point_at_a_page():
    assert citation("report.pdf", 4) == "report.pdf#page=4"
    assert citation("Report.PDF", 1) == "Report.PDF#page=1"


def test_non_paged_citations_are_the_file_name():
    assert citation("notes.md", 1) == "notes.md"


def test_short_text_is_one_chunk():
    assert split_text("one\n\ntwo", size=100) == ["one\n\ntwo"]


def test_chunks_respect_size_and_overlap():
    paragraphs = [f"paragraph {i} " + "x" * 80 for i in range(10)]
    pieces = split_text("\n\n".join(paragraphs), size=300, overlap=50)

    assert len(pieces) > 1
    assert all(len(p) <= 300 + 50 + 2 for p in pieces)
    # Every paragraph survives somewhere.
    for paragraph in paragraphs:
        assert any(paragraph in p for p in pieces)


def test_oversized_paragraph_is_hard_split():
    pieces = split_text("y" * 1000, size=300, overlap=50)
    assert all(len(p) <= 300 for p in pieces)
    assert "".join(p[:250] for p in pieces[:-1]) + pieces[-1] == "y" * 1000


def test_parse_markdown(tmp_path):
    file = tmp_path / "notes.md"
    file.write_text("# Title\n\nSome content.", encoding="utf-8")

    parsed = parse(file)

    assert parsed.name == "notes.md"
    assert [c.page for c in parsed.chunks] == [1]
    assert parsed.pages_without_text == []
