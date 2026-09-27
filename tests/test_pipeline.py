import pytest

from civiclens.config import CATEGORY_MODEL_PATH, URGENCY_MODEL_PATH
from civiclens.pipeline import ComplaintAnalyzer

pytestmark = pytest.mark.skipif(
    not (CATEGORY_MODEL_PATH.exists() and URGENCY_MODEL_PATH.exists()),
    reason="Trained models not found; run `python -m civiclens.train` first.",
)


@pytest.fixture(scope="module")
def analyzer():
    return ComplaintAnalyzer.load()


def test_analyze_returns_a_fully_populated_ticket(analyzer):
    ticket = analyzer.analyze(
        "There is a massive pothole on MG Road, a bike already crashed into it, "
        "someone could get seriously injured, please fix this immediately!",
        register_in_duplicate_index=False,
    )
    assert ticket.category == "Roads & Infrastructure"
    assert ticket.final_urgency_level in {"High", "Critical"}
    assert ticket.department
    assert 0.0 <= ticket.category_confidence <= 1.0
    assert ticket.sla_hours > 0


def test_analyze_rejects_empty_text(analyzer):
    with pytest.raises(ValueError):
        analyzer.analyze("   ", register_in_duplicate_index=False)


def test_analyze_generates_complaint_id_when_not_given(analyzer):
    ticket = analyzer.analyze("The park bench near the lake is broken.", register_in_duplicate_index=False)
    assert ticket.complaint_id.startswith("CMP")


def test_duplicate_detection_flags_repeated_complaint(analyzer):
    text = "The community toilet near Harbor Road has been unusable for over a week due to no cleaning."
    first = analyzer.analyze(text, complaint_id="TESTDUP1")
    second = analyzer.analyze(text, complaint_id="TESTDUP2", register_in_duplicate_index=False)
    assert second.duplicate_count >= 1
    assert first.complaint_id in second.similar_complaint_ids
