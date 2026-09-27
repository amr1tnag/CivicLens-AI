"""SQLAlchemy ORM model for a routed complaint ticket."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from civiclens.db.database import Base


class ComplaintRecord(Base):
    __tablename__ = "complaints"

    complaint_id: Mapped[str] = mapped_column(String, primary_key=True)
    text: Mapped[str] = mapped_column(String, nullable=False)
    location: Mapped[str | None] = mapped_column(String, nullable=True)

    category: Mapped[str] = mapped_column(String, nullable=False)
    category_confidence: Mapped[float] = mapped_column(Float, nullable=False)

    ml_predicted_urgency: Mapped[str] = mapped_column(String, nullable=False)
    lexicon_urgency_score: Mapped[float] = mapped_column(Float, nullable=False)
    priority_score: Mapped[float] = mapped_column(Float, nullable=False)
    final_urgency_level: Mapped[str] = mapped_column(String, nullable=False)

    department: Mapped[str] = mapped_column(String, nullable=False)
    escalated_department: Mapped[str | None] = mapped_column(String, nullable=True)
    sla_hours: Mapped[int] = mapped_column(Integer, nullable=False)
    sla_deadline: Mapped[str | None] = mapped_column(String, nullable=True)

    duplicate_count: Mapped[int] = mapped_column(Integer, default=0)
    similar_complaint_ids: Mapped[list] = mapped_column(JSON, default=list)
    explanation: Mapped[dict] = mapped_column(JSON, default=dict)

    status: Mapped[str] = mapped_column(String, default="Open")
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
