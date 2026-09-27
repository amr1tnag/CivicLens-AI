"""API integration tests.

Uses an in-memory SQLite database (overriding the app's `get_db` dependency)
so tests never touch the real civiclens.db file used by the running app.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from civiclens.config import CATEGORY_MODEL_PATH, URGENCY_MODEL_PATH
from civiclens.db.database import Base

pytestmark = pytest.mark.skipif(
    not (CATEGORY_MODEL_PATH.exists() and URGENCY_MODEL_PATH.exists()),
    reason="Trained models not found; run `python -m civiclens.train` first.",
)


@pytest.fixture()
def client(monkeypatch):
    import api.main as main_module

    test_engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    TestSession = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr(main_module, "init_db", lambda: None)
    main_module.app.dependency_overrides[main_module.get_db] = override_get_db

    with TestClient(main_module.app) as test_client:
        yield test_client

    main_module.app.dependency_overrides.clear()


SAMPLE_TEXT = (
    "There is a massive pothole on MG Road, a bike already crashed into it, "
    "someone could get seriously injured, please fix this immediately!"
)


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["models_loaded"] is True


def test_submit_complaint_returns_routed_ticket(client):
    r = client.post("/complaints", json={"text": SAMPLE_TEXT, "location": "Ward 3 - Lakeview"})
    assert r.status_code == 201
    body = r.json()
    assert body["category"] == "Roads & Infrastructure"
    assert body["final_urgency_level"] in {"High", "Critical"}
    assert body["status"] == "Open"
    assert body["complaint_id"]


def test_submit_complaint_rejects_too_short_text(client):
    r = client.post("/complaints", json={"text": "short"})
    assert r.status_code == 422


def test_get_complaint_by_id(client):
    created = client.post("/complaints", json={"text": SAMPLE_TEXT}).json()
    r = client.get(f"/complaints/{created['complaint_id']}")
    assert r.status_code == 200
    assert r.json()["complaint_id"] == created["complaint_id"]


def test_get_unknown_complaint_returns_404(client):
    r = client.get("/complaints/does-not-exist")
    assert r.status_code == 404


def test_update_status(client):
    created = client.post("/complaints", json={"text": SAMPLE_TEXT}).json()
    r = client.patch(f"/complaints/{created['complaint_id']}/status", json={"status": "Resolved"})
    assert r.status_code == 200
    assert r.json()["status"] == "Resolved"


def test_update_status_rejects_invalid_value(client):
    created = client.post("/complaints", json={"text": SAMPLE_TEXT}).json()
    r = client.patch(f"/complaints/{created['complaint_id']}/status", json={"status": "Not A Real Status"})
    assert r.status_code == 422


def test_list_complaints_filters_by_category(client):
    client.post("/complaints", json={"text": SAMPLE_TEXT})
    client.post("/complaints", json={
        "text": "The park in Ward 6 has overgrown grass, no rush at all, just a suggestion whenever convenient."
    })
    r = client.get("/complaints", params={"category": "Roads & Infrastructure"})
    assert r.status_code == 200
    assert all(c["category"] == "Roads & Infrastructure" for c in r.json())


def test_dashboard_stats_reflect_submitted_complaints(client):
    client.post("/complaints", json={"text": SAMPLE_TEXT})
    r = client.get("/stats/dashboard")
    assert r.status_code == 200
    stats = r.json()
    assert stats["total_complaints"] >= 1
    assert stats["open_count"] >= 1
