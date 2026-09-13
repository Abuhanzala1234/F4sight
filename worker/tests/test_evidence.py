"""Evidence canonicalisation and hashing (§7.11). Blocker #7.

The RFC 8785 number vectors are the important part of this file. Every naive
JCS implementation passes the string tests and fails on 1e21, 1e-7 or -0, and
the failure is invisible until verification breaks months later.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from drishti_worker.evidence import (
    EXCLUDED_FIELDS,
    assemble,
    canonicalise,
    es_number_to_string,
    evidence_hash,
    strip_excluded,
    verify,
)


class TestEcmaScriptNumbers:
    """RFC 8785 §3.2.2.3. These are the values that break naive code."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.0, "0"),
            (-0.0, "0"),  # JCS normalises minus zero
            (1.0, "1"),  # integral floats print WITHOUT a decimal point
            (100.0, "100"),
            (-1.5, "-1.5"),
            (0.1, "0.1"),
            (1e20, "100000000000000000000"),  # still positional at 1e20
            (1e21, "1e+21"),  # exponential from 1e21
            (1e-6, "0.000001"),  # still positional at 1e-6
            (1e-7, "1e-7"),  # exponential below; note: NO zero padding
            (5e-324, "5e-324"),  # smallest subnormal
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (333333333.3333332, "333333333.3333332"),
            (9.999999999999997e22, "9.999999999999997e+22"),
        ],
    )
    def test_vectors(self, value, expected):
        assert es_number_to_string(value) == expected

    def test_integers_pass_through(self):
        assert es_number_to_string(42) == "42"
        assert es_number_to_string(-7) == "-7"

    def test_bool_is_not_a_number(self):
        with pytest.raises(TypeError):
            es_number_to_string(True)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_refused(self, bad):
        """Refusing to hash NaN is the point: JSON cannot represent it, so a
        silent substitution would make the document unverifiable."""
        with pytest.raises(ValueError):
            es_number_to_string(bad)

    @given(x=st.floats(allow_nan=False, allow_infinity=False, width=64))
    @settings(max_examples=500)
    def test_round_trips_through_json(self, x):
        """Whatever we emit must parse back to the identical double."""
        assert (
            json.loads(es_number_to_string(x)) == pytest.approx(x, rel=0, abs=0)
            or x == 0
        )


class TestCanonicalisation:
    def test_no_whitespace(self):
        assert canonicalise({"a": 1, "b": [1, 2]}) == b'{"a":1,"b":[1,2]}'

    def test_keys_are_sorted(self):
        assert canonicalise({"b": 1, "a": 2, "c": 3}) == b'{"a":2,"b":1,"c":3}'

    def test_key_order_does_not_change_the_hash(self):
        a = {"alpha": 1, "beta": {"x": 1, "y": 2}}
        b = {"beta": {"y": 2, "x": 1}, "alpha": 1}
        assert evidence_hash(a) == evidence_hash(b)

    def test_utf16_code_unit_ordering(self):
        """Above the BMP, UTF-16 order differs from code-point order. Sorting
        by code point here would be invisible until an evidence note contained
        an emoji."""
        out = canonicalise({"\U0001f600": 1, "￿": 2}).decode()
        assert out.index("\U0001f600") < out.index("￿")

    def test_non_ascii_stays_literal(self):
        assert canonicalise({"k": "café"}) == '{"k":"café"}'.encode()

    def test_control_characters_escaped(self):
        assert canonicalise({"k": "a\nb\tc\x01"}) == b'{"k":"a\\nb\\tc\\u0001"}'

    def test_quotes_and_backslashes(self):
        assert canonicalise({"k": 'a"b\\c'}) == b'{"k":"a\\"b\\\\c"}'

    def test_nested_structures(self):
        doc = {"z": [{"b": 2, "a": 1}], "a": None, "t": True, "f": False}
        assert canonicalise(doc) == b'{"a":null,"f":false,"t":true,"z":[{"a":1,"b":2}]}'

    def test_datetime_is_refused(self):
        """Forces the caller to choose an explicit ISO-8601 serialisation
        rather than inheriting whatever a library does this month."""
        from datetime import UTC, datetime

        with pytest.raises(TypeError):
            canonicalise({"t": datetime.now(UTC)})

    def test_unsupported_type_is_refused(self):
        with pytest.raises(TypeError):
            canonicalise({"k": {1, 2, 3}})


class TestExclusion:
    def test_excluded_fields(self):
        doc = {"a": 1, "evidence_hash": "x", "ledger": {"tx": "y"}}
        assert strip_excluded(doc) == {"a": 1}

    def test_hash_ignores_excluded_fields(self):
        base = {"alert_id": "x", "n": 1}
        assert evidence_hash(base) == evidence_hash(
            {**base, "evidence_hash": "anything", "ledger": {"tx_id": "z"}}
        )

    def test_exclusion_is_top_level_only(self):
        """A nested field called 'ledger' is real content and must be covered."""
        a = {"note": {"ledger": "the accounting kind"}}
        b = {"note": {"ledger": "a different value"}}
        assert evidence_hash(a) != evidence_hash(b)

    def test_excluded_set_is_exactly_the_spec(self):
        assert frozenset({"evidence_hash", "ledger"}) == EXCLUDED_FIELDS


class TestAssembleAndVerify:
    def _doc(self, **overrides):
        base = dict(
            alert_id="a-1",
            site={"code": "BOP-03"},
            camera={"code": "CAM-01"},
            detection={"track_id": 7, "class": "person"},
            risk={"score": 60.0, "severity": "high"},
            items=[{"kind": "snapshot", "sha256": "ab" * 32, "enhanced": False}],
            config_version="c" * 64,
            spec_version="1.0.0",
            worker_version="1.0.0",
            created_at="2026-09-12T22:00:00+00:00",
        )
        base.update(overrides)
        return assemble(**base)

    def test_round_trip(self):
        doc = self._doc()
        digest = evidence_hash(doc)
        result = verify(doc, digest)
        assert result.ok
        assert result.computed_hash == digest
        assert all(passed for _, passed, _ in result.checks)

    def test_item_without_digest_is_refused(self):
        """Each file's digest must be INSIDE the hashed document, else altering
        a JPEG would not invalidate the alert."""
        with pytest.raises(ValueError, match="sha256"):
            self._doc(items=[{"kind": "snapshot"}])

    def test_enhanced_evidence_is_refused(self):
        """P4, enforced in code as well as in the database."""
        with pytest.raises(ValueError, match="enhanced"):
            self._doc(
                items=[{"kind": "snapshot", "sha256": "ab" * 32, "enhanced": True}]
            )

    def test_tampering_is_detected_with_a_diff(self):
        doc = self._doc()
        digest = evidence_hash(doc)
        tampered = json.loads(json.dumps(doc))
        tampered["detection"]["track_id"] = 8

        result = verify(tampered, digest, reference=doc)
        assert not result.ok
        assert result.diff
        assert any("track_id" in line for line in result.diff)

    def test_altered_media_digest_breaks_the_hash(self):
        """The chain: file bytes -> item digest -> evidence doc -> hash."""
        doc = self._doc()
        digest = evidence_hash(doc)
        doc2 = json.loads(json.dumps(doc))
        doc2["items"][0]["sha256"] = "cd" * 32
        assert evidence_hash(doc2) != digest

    def test_verify_reports_canonical_length(self):
        doc = self._doc()
        result = verify(doc, evidence_hash(doc))
        assert result.canonical_length == len(canonicalise(strip_excluded(doc)))

    def test_hash_matches_manual_computation(self):
        """No hidden steps: the hash is sha256 over the canonical bytes."""
        doc = self._doc()
        manual = hashlib.sha256(canonicalise(strip_excluded(doc))).hexdigest()
        assert evidence_hash(doc) == manual


@given(
    payload=st.dictionaries(
        st.text(min_size=1, max_size=8),
        st.one_of(
            st.integers(-10_000, 10_000),
            st.text(max_size=20),
            st.booleans(),
            st.none(),
        ),
        max_size=8,
    )
)
@settings(max_examples=300)
def test_canonicalisation_is_deterministic(payload):
    assert canonicalise(payload) == canonicalise(dict(reversed(list(payload.items()))))
