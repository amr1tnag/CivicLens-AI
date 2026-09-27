"""Pydantic request/response schemas for the CivicLens API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ComplaintCreate(BaseModel):
    text: str = Field(..., min_length=10, max_length=4000, description="The citizen's complaint text.")
    location: str | None = Field(None, max_length=200, description="Free-text location/ward, if known.")


class ComplaintResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    complaint_id: str
    text: str
    location: str | None

    category: str
    category_confidence: float

    ml_predicted_urgency: str
    lexicon_urgency_score: float
    priority_score: float
    final_urgency_level: str

    department: str
    escalated_department: str | None
    sla_hours: int
    sla_deadline: str | None

    duplicate_count: int
    similar_complaint_ids: list[str]
    explanation: dict

    status: str
    submitted_at: datetime


class StatusUpdate(BaseModel):
    status: str = Field(..., description="One of: Open, In Progress, Resolved")


class CategoryCount(BaseModel):
    category: str
    count: int


class UrgencyCount(BaseModel):
    urgency: str
    count: int


class DepartmentCount(BaseModel):
    department: str
    count: int


class DashboardStats(BaseModel):
    total_complaints: int
    open_count: int
    in_progress_count: int
    resolved_count: int
    average_priority_score: float
    by_category: list[CategoryCount]
    by_urgency: list[UrgencyCount]
    by_department: list[DepartmentCount]
