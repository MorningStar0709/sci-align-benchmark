"""API smoke tests — verify endpoints mount and respond without crashing.

Uses FastAPI TestClient so no uvicorn process is needed.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mineru_api.app import app

# Env overrides to unlock cloud-demo endpoints in tests (no real keys needed).
_TEST_ENV = {
    "DEMO_REQUIRE_ACCESS_CODE": "0",
    "DEMO_ALLOW_DRY_RUN_WITHOUT_CODE": "1",
}


@pytest.fixture
def client():
    with patch.dict(os.environ, _TEST_ENV):
        yield TestClient(app)


class TestApiSmoke:
    def test_root_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_demo_health(self, client):
        """May return 'degraded' if external services are down — that's fine."""
        resp = client.get("/api/demo/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert "components" in data

    def test_demo_quota(self, client):
        resp = client.get("/api/demo/quota")
        assert resp.status_code == 200

    def test_openapi_schema(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        # Expect at least the 3 health endpoints + quota
        health_paths = [p for p in schema["paths"] if "health" in p or "quota" in p]
        assert len(health_paths) >= 2


class TestCloudDemoSmoke:
    """Cloud demo POST/GET endpoints — happy path and validation."""

    def test_cloud_demo_index(self, client):
        resp = client.get("/api/cloud-demo/")
        assert resp.status_code == 200
        assert resp.json()["name"]

    def test_cloud_demo_health(self, client):
        resp = client.get("/api/cloud-demo/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ready", "degraded")
        assert "components" in data

    def test_cloud_demo_runtime(self, client):
        resp = client.get("/api/cloud-demo/runtime")
        assert resp.status_code == 200
        assert resp.json()["mode"] == "pure_cloud_only"

    def test_cloud_demo_quota(self, client):
        resp = client.get("/api/cloud-demo/quota")
        assert resp.status_code == 200

    def test_cloud_demo_tracks(self, client):
        resp = client.get("/api/cloud-demo/tracks")
        assert resp.status_code == 200
        tracks = resp.json()["tracks"]
        assert len(tracks) >= 3

    def test_agent_plan_missing_goal(self, client):
        """POST with empty goal should return 400."""
        resp = client.post(
            "/api/cloud-demo/agent/plan",
            json={"goal": ""},
        )
        assert resp.status_code == 400

    def test_agent_plan_happy(self, client):
        """POST with valid goal should succeed (dry_run / no key required)."""
        resp = client.post(
            "/api/cloud-demo/agent/plan",
            json={"goal": "test RAG pipeline"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "dag" in data

    def test_medical_rag_ask_empty_question(self, client):
        """POST with empty question should return 400."""
        resp = client.post(
            "/api/cloud-demo/medical-rag/ask",
            json={"question": ""},
        )
        assert resp.status_code == 400

    def test_quiz_generate_short_source(self, client):
        """source_text < 20 chars should return 400."""
        resp = client.post(
            "/api/cloud-demo/quiz/generate",
            json={"source_text": "short", "topic": "test"},
        )
        assert resp.status_code == 400

    def test_quiz_grade_missing_fields(self, client):
        """Missing question/student_answer should return 400."""
        resp = client.post(
            "/api/cloud-demo/quiz/grade",
            json={"question": "", "student_answer": ""},
        )
        assert resp.status_code == 400
