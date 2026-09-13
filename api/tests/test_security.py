"""Auth and RBAC (§8, §12)."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from drishti_api.security import (
    ROLE_ORDER,
    LoginRateLimiter,
    Principal,
    create_token,
    decode_token,
    hash_password,
    verify_password,
)
from drishti_api.settings import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(jwt_secret="x" * 48, access_token_ttl_s=60)


class TestPasswords:
    def test_round_trip(self):
        stored = hash_password("correct horse battery staple")
        assert verify_password("correct horse battery staple", stored)
        assert not verify_password("wrong", stored)

    def test_hashes_are_salted(self):
        """Two identical passwords must not produce the same hash."""
        assert hash_password("same") != hash_password("same")

    def test_malformed_hash_returns_false_rather_than_raising(self):
        assert not verify_password("anything", "not-a-hash")


class TestTokens:
    def test_round_trip(self, settings):
        token, ttl = create_token("user-1", "operator", settings)
        payload = decode_token(token, settings)
        assert payload["sub"] == "user-1"
        assert payload["role"] == "operator"
        assert payload["typ"] == "access"
        assert ttl == 60

    def test_refresh_tokens_are_marked(self, settings):
        token, _ = create_token("user-1", "viewer", settings, refresh=True)
        assert decode_token(token, settings)["typ"] == "refresh"

    def test_a_token_from_another_secret_is_rejected(self, settings):
        token, _ = create_token("user-1", "admin", settings)
        other = Settings(jwt_secret="y" * 48)
        with pytest.raises(HTTPException) as exc:
            decode_token(token, other)
        assert exc.value.status_code == 401

    def test_expired_token_is_rejected(self):
        settings = Settings(jwt_secret="z" * 48, access_token_ttl_s=-10)
        token, _ = create_token("user-1", "viewer", settings)
        with pytest.raises(HTTPException):
            decode_token(token, settings)

    def test_tampered_token_is_rejected(self, settings):
        token, _ = create_token("user-1", "viewer", settings)
        forged = token[:-4] + ("aaaa" if not token.endswith("aaaa") else "bbbb")
        with pytest.raises(HTTPException):
            decode_token(forged, settings)


class TestRoleLadder:
    @pytest.mark.parametrize("role", ROLE_ORDER)
    def test_role_implies_everything_below_it(self, role):
        principal = Principal("u", role)
        index = ROLE_ORDER.index(role)
        for i, other in enumerate(ROLE_ORDER):
            assert principal.at_least(other) is (index >= i)

    def test_viewer_cannot_do_admin_things(self):
        assert not Principal("u", "viewer").at_least("admin")

    def test_admin_can_do_everything(self):
        assert all(Principal("u", "admin").at_least(r) for r in ROLE_ORDER)

    def test_unknown_role_grants_nothing(self):
        """A typo in a role name must fail closed, never open."""
        assert not Principal("u", "superuser").at_least("viewer")


class TestLoginRateLimit:
    def test_blocks_after_the_limit(self):
        limiter = LoginRateLimiter(per_minute=3)
        for _ in range(3):
            limiter.check("10.0.0.1")
        with pytest.raises(HTTPException) as exc:
            limiter.check("10.0.0.1")
        assert exc.value.status_code == 429

    def test_addresses_are_independent(self):
        limiter = LoginRateLimiter(per_minute=1)
        limiter.check("10.0.0.1")
        limiter.check("10.0.0.2")  # must not raise

    def test_successful_login_resets_the_counter(self):
        limiter = LoginRateLimiter(per_minute=2)
        limiter.check("10.0.0.1")
        limiter.reset("10.0.0.1")
        limiter.check("10.0.0.1")
        limiter.check("10.0.0.1")  # still under the limit
