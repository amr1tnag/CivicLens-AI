"""End-to-end orchestration: raw complaint text -> a routed Ticket.

This is the single entry point both the API and the CLI demo call, so
neither needs to know about vectorizers, classifiers, lexicons or the
duplicate index individually - just `ComplaintAnalyzer.load().analyze(text)`.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from civiclens.config import CATEGORY_MODEL_PATH, DUPLICATE_INDEX_PATH, URGENCY_MODEL_PATH
from civiclens.nlp import lexicons
from civiclens.nlp.category_model import CategoryModel
from civiclens.nlp.duplicates import DuplicateIndex
from civiclens.nlp.urgency_model import UrgencyModel
from civiclens.routing.router import Ticket, route


class ComplaintAnalyzer:
    def __init__(
        self,
        category_model: CategoryModel,
        urgency_model: UrgencyModel,
        duplicate_index: DuplicateIndex | None = None,
    ):
        self.category_model = category_model
        self.urgency_model = urgency_model
        self.duplicate_index = duplicate_index

    @classmethod
    def load(
        cls,
        category_path=CATEGORY_MODEL_PATH,
        urgency_path=URGENCY_MODEL_PATH,
        duplicate_path=DUPLICATE_INDEX_PATH,
        load_duplicate_index: bool = True,
    ) -> "ComplaintAnalyzer":
        category_model = CategoryModel.load(category_path)
        urgency_model = UrgencyModel.load(urgency_path)
        duplicate_index = None
        if load_duplicate_index and duplicate_path.exists():
            duplicate_index = DuplicateIndex.load(duplicate_path)
        return cls(category_model, urgency_model, duplicate_index)

    def analyze(
        self,
        text: str,
        complaint_id: str | None = None,
        submitted_at: datetime | None = None,
        register_in_duplicate_index: bool = True,
    ) -> Ticket:
        if not text or not text.strip():
            raise ValueError("Complaint text must not be empty")

        complaint_id = complaint_id or f"CMP{uuid.uuid4().hex[:10].upper()}"
        submitted_at = submitted_at or datetime.now(timezone.utc)

        category, category_confidence = self.category_model.predict_one(text)
        ml_urgency, urgency_confidence = self.urgency_model.predict_one(text)
        lex = lexicons.analyze(text)

        duplicates = []
        if self.duplicate_index is not None:
            duplicates = self.duplicate_index.find_similar(text, category=category)

        ticket = route(
            complaint_id=complaint_id,
            category=category,
            category_confidence=category_confidence,
            ml_urgency=ml_urgency,
            lexicon_urgency_score_normalized=lex.urgency_score_normalized,
            duplicates=duplicates,
            submitted_at=submitted_at,
            lexicon_explanation=lexicons.explain(text),
        )
        ticket.explanation["ml_urgency_confidence"] = round(urgency_confidence, 4)

        if self.duplicate_index is not None and register_in_duplicate_index:
            self.duplicate_index.add(text, complaint_id, category)

        return ticket
