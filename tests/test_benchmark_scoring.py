"""Benchmark scoring: how an answer is judged for correctness and Citation accuracy."""

from benchmark.scoring import Question, score

REFUNDS = Question(
    id="q1",
    type="lookup",
    question="How long do refunds take and who approves them?",
    facts=(("14 days", "fourteen days"), ("Dana Reyes",)),
    pages=("policy.pdf#page=2",),
    related_pages=("policy.pdf#page=3",),
)


def test_answer_with_every_fact_and_the_right_page_scores_full_marks():
    answer = "Refunds take 14 days and Dana Reyes approves them [policy.pdf#page=2]."

    result = score(REFUNDS, answer)

    assert result.correct
    assert result.fact_recall == 1.0
    assert result.citation_recall == 1.0
    assert result.citation_precision == 1.0


def test_facts_match_any_alternative_ignoring_case_and_number_formatting():
    answer = "They take FOURTEEN DAYS; approval sits with dana reyes."

    assert score(REFUNDS, answer).fact_recall == 1.0


def test_a_missing_fact_makes_the_answer_partially_correct():
    result = score(REFUNDS, "Refunds take 14 days [policy.pdf#page=2].")

    assert not result.correct
    assert result.fact_recall == 0.5


def test_citing_a_wrong_page_lowers_precision_but_related_pages_count_as_right():
    answer = (
        "Refunds take 14 days, approved by Dana Reyes "
        "[policy.pdf#page=2] [policy.pdf#page=3] [policy.pdf#page=9]."
    )

    result = score(REFUNDS, answer)

    assert result.citation_recall == 1.0
    assert result.citation_precision == 2 / 3


def test_an_answer_without_citations_has_zero_citation_scores():
    result = score(REFUNDS, "Refunds take 14 days, approved by Dana Reyes.")

    assert result.citation_recall == 0.0
    assert result.citation_precision == 0.0


def test_citations_are_read_from_a_references_section_too():
    answer = (
        "Refunds take 14 days [1], approved by Dana Reyes [1].\n\n"
        "### References\n\n* [1] policy.pdf#page=2\n"
    )

    assert score(REFUNDS, answer).cited == ("policy.pdf#page=2",)


def test_thousands_separators_do_not_break_number_facts():
    question = Question(
        id="q2",
        type="lookup",
        question="Capacity?",
        facts=(("1200 units",),),
        pages=("a.pdf#page=1",),
    )

    assert score(question, "About 1,200 units a year.").correct
