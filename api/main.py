"""CivicLens AI - FastAPI backend.

Run with:
    uvicorn api.main:app --reload

Interactive API docs are then served at /docs (Swagger UI) and /redoc.
"""

from __future__ import annotations

import threading
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.schemas import (
    CategoryCount,
    ComplaintCreate,
    ComplaintResponse,
    DashboardStats,
    DepartmentCount,
    StatusUpdate,
    UrgencyCount,
)
from civiclens.config import CATEGORY_MODEL_PATH, URGENCY_MODEL_PATH
from civiclens.db.database import get_db, init_db
from civiclens.db.models import ComplaintRecord
from civiclens.pipeline import ComplaintAnalyzer

VALID_STATUSES = ["Open", "In Progress", "Resolved"]

# Guards the analyzer's in-memory duplicate index, which is mutated on every
# submission (see DuplicateIndex.add): FastAPI runs sync `def` endpoints in a
# thread pool, so concurrent requests could otherwise race on it.
_analyzer_lock = threading.Lock()
_analyzer: ComplaintAnalyzer | None = None


def get_analyzer() -> ComplaintAnalyzer:
    if _analyzer is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Models are not loaded. Run `python -m civiclens.train` first "
                f"to produce {CATEGORY_MODEL_PATH.name} and {URGENCY_MODEL_PATH.name}."
            ),
        )
    return _analyzer


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _analyzer
    init_db()
    if CATEGORY_MODEL_PATH.exists() and URGENCY_MODEL_PATH.exists():
        _analyzer = ComplaintAnalyzer.load()
    else:
        _analyzer = None  # /complaints will 503 with a clear message until trained
    yield


app = FastAPI(
    title="CivicLens AI",
    description="Understands, prioritizes, and routes citizen complaints using NLP.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": _analyzer is not None}


@app.post("/complaints", response_model=ComplaintResponse, status_code=201)
def submit_complaint(payload: ComplaintCreate, db: Session = Depends(get_db)):
    analyzer = get_analyzer()

    with _analyzer_lock:
        ticket = analyzer.analyze(payload.text)

    record = ComplaintRecord(
        complaint_id=ticket.complaint_id,
        text=payload.text,
        location=payload.location,
        category=ticket.category,
        category_confidence=ticket.category_confidence,
        ml_predicted_urgency=ticket.ml_predicted_urgency,
        lexicon_urgency_score=ticket.lexicon_urgency_score,
        priority_score=ticket.priority_score,
        final_urgency_level=ticket.final_urgency_level,
        department=ticket.department,
        escalated_department=ticket.escalated_department,
        sla_hours=ticket.sla_hours,
        sla_deadline=ticket.sla_deadline,
        duplicate_count=ticket.duplicate_count,
        similar_complaint_ids=ticket.similar_complaint_ids,
        explanation=ticket.explanation,
        status="Open",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@app.get("/complaints", response_model=list[ComplaintResponse])
def list_complaints(
    category: str | None = None,
    urgency: str | None = Query(None, description="Filter by final_urgency_level"),
    department: str | None = None,
    status: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    stmt = select(ComplaintRecord)
    if category:
        stmt = stmt.where(ComplaintRecord.category == category)
    if urgency:
        stmt = stmt.where(ComplaintRecord.final_urgency_level == urgency)
    if department:
        stmt = stmt.where(ComplaintRecord.department == department)
    if status:
        stmt = stmt.where(ComplaintRecord.status == status)
    stmt = stmt.order_by(ComplaintRecord.priority_score.desc()).offset(offset).limit(limit)
    return db.execute(stmt).scalars().all()


@app.get("/complaints/{complaint_id}", response_model=ComplaintResponse)
def get_complaint(complaint_id: str, db: Session = Depends(get_db)):
    record = db.get(ComplaintRecord, complaint_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Complaint {complaint_id} not found")
    return record


@app.patch("/complaints/{complaint_id}/status", response_model=ComplaintResponse)
def update_status(complaint_id: str, payload: StatusUpdate, db: Session = Depends(get_db)):
    if payload.status not in VALID_STATUSES:
        raise HTTPException(status_code=422, detail=f"status must be one of {VALID_STATUSES}")
    record = db.get(ComplaintRecord, complaint_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Complaint {complaint_id} not found")
    record.status = payload.status
    db.commit()
    db.refresh(record)
    return record


@app.get("/stats/dashboard", response_model=DashboardStats)
def dashboard_stats(db: Session = Depends(get_db)):
    records = db.execute(select(ComplaintRecord)).scalars().all()
    total = len(records)
    avg_priority = round(sum(r.priority_score for r in records) / total, 2) if total else 0.0

    return DashboardStats(
        total_complaints=total,
        open_count=sum(1 for r in records if r.status == "Open"),
        in_progress_count=sum(1 for r in records if r.status == "In Progress"),
        resolved_count=sum(1 for r in records if r.status == "Resolved"),
        average_priority_score=avg_priority,
        by_category=[
            CategoryCount(category=k, count=v)
            for k, v in Counter(r.category for r in records).most_common()
        ],
        by_urgency=[
            UrgencyCount(urgency=k, count=v)
            for k, v in Counter(r.final_urgency_level for r in records).most_common()
        ],
        by_department=[
            DepartmentCount(department=k, count=v)
            for k, v in Counter(r.department for r in records).most_common()
        ],
    )
