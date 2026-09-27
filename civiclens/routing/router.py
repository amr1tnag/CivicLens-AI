"""Routing engine: turns model predictions into a routed, prioritized ticket.

Combines three independent signals into one final decision:
  1. the trained urgency classifier's predicted class,
  2. the transparent lexicon's normalized urgency score, and
  3. how many similar complaints already exist (repeat-report boost),
rather than trusting the ML classifier alone. This is deliberate: a
single mis-classified "Low" on a genuinely dangerous complaint is a much
worse failure mode for a civic system than a few points of accuracy, so the
final decision is an ensemble the classifier can't unilaterally get wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from civiclens.config import (
    CATEGORY_TO_DEPARTMENT,
    CRITICAL_ESCALATION_DEPARTMENT,
    URGENCY_SLA_HOURS,
    URGENCY_WEIGHT,
)
from civiclens.nlp.duplicates import SimilarComplaint

# Score bands a final blended 0-100 priority score is mapped back onto,
# centered on URGENCY_WEIGHT's Low/Medium/High/Critical anchors (25/50/75/100).
_LEVEL_BANDS = [
    (87.5, "Critical"),
    (62.5, "High"),
    (37.5, "Medium"),
    (0.0, "Low"),
]

# A duplicate index this large or larger bumps priority (many people
# reporting the same issue is itself evidence it matters).
DUPLICATE_BOOST_THRESHOLD = 2
DUPLICATE_BOOST_POINTS = 12

ML_WEIGHT = 0.7
LEXICON_WEIGHT = 0.3


def _score_to_level(score: float) -> str:
    for threshold, level in _LEVEL_BANDS:
        if score >= threshold:
            return level
    return "Low"  # pragma: no cover - unreachable, _LEVEL_BANDS covers [0, inf)


@dataclass
class Ticket:
    complaint_id: str
    category: str
    category_confidence: float
    department: str
    ml_predicted_urgency: str
    lexicon_urgency_score: float
    priority_score: float
    final_urgency_level: str
    sla_hours: int
    sla_deadline: str | None
    duplicate_count: int
    similar_complaint_ids: list[str] = field(default_factory=list)
    escalated_department: str | None = None
    explanation: dict = field(default_factory=dict)


def route(
    complaint_id: str,
    category: str,
    category_confidence: float,
    ml_urgency: str,
    lexicon_urgency_score_normalized: float,
    duplicates: list[SimilarComplaint] | None = None,
    submitted_at: datetime | None = None,
    lexicon_explanation: dict | None = None,
) -> Ticket:
    duplicates = duplicates or []

    base_score = URGENCY_WEIGHT[ml_urgency]
    lexicon_score_scaled = lexicon_urgency_score_normalized * 100
    blended = ML_WEIGHT * base_score + LEXICON_WEIGHT * lexicon_score_scaled

    duplicate_count = len(duplicates)
    if duplicate_count >= DUPLICATE_BOOST_THRESHOLD:
        blended = min(100.0, blended + DUPLICATE_BOOST_POINTS)

    blended = max(0.0, min(100.0, blended))
    final_level = _score_to_level(blended)

    department = CATEGORY_TO_DEPARTMENT.get(category, "General Administration / Citizen Helpdesk")
    escalated_department = CRITICAL_ESCALATION_DEPARTMENT if final_level == "Critical" else None

    sla_hours = URGENCY_SLA_HOURS[final_level]
    sla_deadline = (submitted_at + timedelta(hours=sla_hours)).isoformat() if submitted_at else None

    explanation = dict(lexicon_explanation or {})
    explanation["ml_predicted_urgency"] = ml_urgency
    explanation["ml_category_confidence"] = round(category_confidence, 4)
    explanation["blended_priority_score"] = round(blended, 2)
    if duplicate_count >= DUPLICATE_BOOST_THRESHOLD:
        explanation["duplicate_boost_applied"] = f"+{DUPLICATE_BOOST_POINTS} (found {duplicate_count} similar reports)"

    return Ticket(
        complaint_id=complaint_id,
        category=category,
        category_confidence=round(category_confidence, 4),
        department=department,
        ml_predicted_urgency=ml_urgency,
        lexicon_urgency_score=lexicon_urgency_score_normalized,
        priority_score=round(blended, 2),
        final_urgency_level=final_level,
        sla_hours=sla_hours,
        sla_deadline=sla_deadline,
        duplicate_count=duplicate_count,
        similar_complaint_ids=[d.complaint_id for d in duplicates],
        escalated_department=escalated_department,
        explanation=explanation,
    )
