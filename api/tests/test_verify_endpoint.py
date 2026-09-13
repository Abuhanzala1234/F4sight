"""The verification endpoint (§8). The one that has to convince a sceptic."""

from __future__ import annotations

import json

from drishti_worker.evidence import assemble, evidence_hash


def genuine_document() -> dict:
    doc = assemble(
        alert_id="a-1",
        site={"code": "BOP-03"},
        camera={"code": "CAM-01"},
        detection={"track_id": 7, "class": "person", "evqm_profile": "night"},
        risk={"score": 65.0, "severity": "high"},
        items=[{"kind": "snapshot", "sha256": "a" * 64, "enhanced": False}],
        config_version="c" * 64,
        spec_version="1.0.0",
        worker_version="1.0.0",
        created_at="2026-09-12T22:14:00+00:00",
    )
    doc["evidence_hash"] = evidence_hash(doc)
    return doc


class TestVerifyDocument:
    def test_genuine_document_verifies(self, client):
        doc = genuine_document()
        body = client.post("/api/v1/verify/document", json={"document": doc}).json()
        assert body["verdict"] == "VERIFIED"
        assert body["hash_match"] is True
        assert all(c["passed"] for c in body["checks"])

    def test_it_returns_every_intermediate_value(self, client):
        """A sceptical evaluator must be able to recompute each number."""
        doc = genuine_document()
        body = client.post("/api/v1/verify/document", json={"document": doc}).json()
        assert body["canonical_length"] > 0
        assert len(body["recomputed_hash"]) == 64
        assert body["canonical_bytes_sha256"] == body["recomputed_hash"]

    def test_one_altered_field_is_caught(self, client):
        doc = genuine_document()
        tampered = json.loads(json.dumps(doc))
        tampered["detection"]["track_id"] = 999
        body = client.post("/api/v1/verify/document", json={"document": tampered}).json()
        assert body["verdict"] == "TAMPERED"
        assert body["hash_match"] is False

    def test_altered_media_digest_is_caught(self, client):
        """file bytes -> item digest -> evidence doc -> hash. Break any link."""
        doc = genuine_document()
        tampered = json.loads(json.dumps(doc))
        tampered["items"][0]["sha256"] = "b" * 64
        body = client.post("/api/v1/verify/document", json={"document": tampered}).json()
        assert body["verdict"] == "TAMPERED"

    def test_key_reordering_does_not_break_verification(self, client):
        """The whole point of RFC 8785: semantically identical JSON verifies."""
        doc = genuine_document()
        shuffled = dict(reversed(list(doc.items())))
        body = client.post("/api/v1/verify/document", json={"document": shuffled}).json()
        assert body["verdict"] == "VERIFIED"

    def test_missing_hash_is_a_400(self, client):
        doc = genuine_document()
        del doc["evidence_hash"]
        response = client.post("/api/v1/verify/document", json={"document": doc})
        assert response.status_code == 400

    def test_explicit_expected_hash_overrides_the_document(self, client):
        doc = genuine_document()
        response = client.post(
            "/api/v1/verify/document", json={"document": doc, "expected_hash": "0" * 64}
        )
        assert response.json()["verdict"] == "TAMPERED"

    def test_no_authentication_required(self, client):
        """P5: tamper-evidence is a property of the record. A third party must
        be able to check it without an account on this system."""
        response = client.post("/api/v1/verify/document", json={"document": genuine_document()})
        assert response.status_code == 200


class TestAccessControl:
    def test_alert_feed_requires_a_token(self, client):
        assert client.get("/api/v1/alerts").status_code == 401

    def test_garbage_token_is_rejected(self, client):
        response = client.get("/api/v1/alerts", headers={"Authorization": "Bearer not-a-jwt"})
        assert response.status_code == 401

    def test_refresh_token_cannot_call_the_api(self, client):
        from drishti_api.security import create_token
        from drishti_api.settings import get_settings

        refresh, _ = create_token("u", "admin", get_settings(), refresh=True)
        response = client.get("/api/v1/alerts", headers={"Authorization": f"Bearer {refresh}"})
        assert response.status_code == 401
        assert "refresh" in response.json()["detail"].lower()


class TestPublicSurface:
    def test_root_identifies_the_system(self, client):
        body = client.get("/").json()
        assert body["name"] == "DRISHTI-BOP"
        assert "26187" in body["problem_statement"]

    def test_liveness_needs_no_database(self, client):
        assert client.get("/api/v1/health/live").json()["status"] == "alive"

    def test_openapi_is_served(self, client):
        spec = client.get("/api/openapi.json").json()
        assert "/api/v1/alerts/{alert_id}/verify" in spec["paths"]
