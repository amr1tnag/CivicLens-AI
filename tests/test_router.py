from datetime import datetime, timedelta, timezone

from civiclens.config import CATEGORY_TO_DEPARTMENT, CRITICAL_ESCALATION_DEPARTMENT, URGENCY_SLA_HOURS
from civiclens.nlp.duplicates import SimilarComplaint
from civiclens.routing.router import DUPLICATE_BOOST_POINTS, DUPLICATE_BOOST_THRESHOLD, route


def _dupe(cid: str, sim: float = 0.7) -> SimilarComplaint:
    return SimilarComplaint(complaint_id=cid, similarity=sim, category="Roads & Infrastructure", text="...")


def test_critical_prediction_is_escalated_and_gets_shortest_sla():
    ticket = route(
        complaint_id="CMP1",
        category="Electricity & Power",
        category_confidence=0.9,
        ml_urgency="Critical",
        lexicon_urgency_score_normalized=0.95,
    )
    assert ticket.final_urgency_level == "Critical"
    assert ticket.escalated_department == CRITICAL_ESCALATION_DEPARTMENT
    assert ticket.sla_hours == URGENCY_SLA_HOURS["Critical"]


def test_low_prediction_has_no_escalation_and_longest_sla():
    ticket = route(
        complaint_id="CMP2",
        category="Parks & Public Property",
        category_confidence=0.9,
        ml_urgency="Low",
        lexicon_urgency_score_normalized=0.05,
    )
    assert ticket.final_urgency_level == "Low"
    assert ticket.escalated_department is None
    assert ticket.sla_hours == URGENCY_SLA_HOURS["Low"]


def test_department_routing_table_is_respected():
    for category, expected_department in CATEGORY_TO_DEPARTMENT.items():
        ticket = route(
            complaint_id="CMP3",
            category=category,
            category_confidence=0.9,
            ml_urgency="Medium",
            lexicon_urgency_score_normalized=0.5,
        )
        assert ticket.department == expected_department


def test_duplicate_boost_applied_at_threshold():
    duplicates = [_dupe(f"CMP{i}") for i in range(DUPLICATE_BOOST_THRESHOLD)]
    boosted = route(
        complaint_id="CMP4",
        category="Roads & Infrastructure",
        category_confidence=0.9,
        ml_urgency="Medium",
        lexicon_urgency_score_normalized=0.5,
        duplicates=duplicates,
    )
    baseline = route(
        complaint_id="CMP5",
        category="Roads & Infrastructure",
        category_confidence=0.9,
        ml_urgency="Medium",
        lexicon_urgency_score_normalized=0.5,
        duplicates=[],
    )
    assert boosted.priority_score >= baseline.priority_score + DUPLICATE_BOOST_POINTS - 1e-6
    assert "duplicate_boost_applied" in boosted.explanation
    assert "duplicate_boost_applied" not in baseline.explanation


def test_priority_score_is_capped_at_100():
    duplicates = [_dupe(f"CMP{i}") for i in range(10)]
    ticket = route(
        complaint_id="CMP6",
        category="Roads & Infrastructure",
        category_confidence=0.9,
        ml_urgency="Critical",
        lexicon_urgency_score_normalized=1.0,
        duplicates=duplicates,
    )
    assert ticket.priority_score <= 100.0


def test_sla_deadline_computed_from_submitted_at():
    submitted = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ticket = route(
        complaint_id="CMP7",
        category="Water Supply",
        category_confidence=0.9,
        ml_urgency="High",
        lexicon_urgency_score_normalized=0.6,
        submitted_at=submitted,
    )
    expected = submitted + timedelta(hours=URGENCY_SLA_HOURS["High"])
    assert ticket.sla_deadline == expected.isoformat()


def test_sla_deadline_is_none_without_submitted_at():
    ticket = route(
        complaint_id="CMP8",
        category="Water Supply",
        category_confidence=0.9,
        ml_urgency="High",
        lexicon_urgency_score_normalized=0.6,
    )
    assert ticket.sla_deadline is None
