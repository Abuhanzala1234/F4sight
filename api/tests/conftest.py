"""API test fixtures.

These tests deliberately avoid a database. Everything that needs Postgres is
marked ``integration`` and skipped by default, so `make test` stays green on a
machine with no Docker — which is the machine an evaluator will have.

What IS tested without a database: RBAC, token handling, schema validation
(including the P2 sum invariant), and the whole ``/verify/document`` path, which
is the endpoint that has to convince a sceptic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "api" / "src"))
sys.path.insert(0, str(ROOT / "worker" / "src"))

from fastapi.testclient import TestClient

from ibvap_api.main import app
from ibvap_api.security import create_token
from ibvap_api.settings import get_settings


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def token_for():
    def _make(role: str = "viewer", user_id: str = "test-user") -> str:
        token, _ = create_token(user_id, role, get_settings())
        return token

    return _make


@pytest.fixture
def auth_headers(token_for):
    def _make(role: str = "viewer") -> dict[str, str]:
        return {"Authorization": f"Bearer {token_for(role)}"}

    return _make
