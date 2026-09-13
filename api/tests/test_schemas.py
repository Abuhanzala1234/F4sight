"""Schema validation — the invariants enforced on the way out (§8)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from drishti_api.schemas import AlertDetail, ZoneIn


def _alert(score: float, breakdown: list[dict]) -> dict:
    return {
        "id": "a-1",
        "site_id": "s",
        "camera_id": "c",
        "kind": "ZONE_INTRUSION",
        "severity": "high",
        "risk_score": score,
        "reason_codes": ["ZONE_INTRUSION"],
        "ts_utc": datetime.now(UTC),
        "status": "raised",
        "ledger_status": "pending",
        "track_id": 1,
        "zone_id": "z",
        "window_start": None,
        "window_end": None,
        "risk_breakdown": breakdown,
        "evidence_hash": "a" * 64,
        "evidence_doc": {},
    }


class TestRiskSumInvariant:
    """P2, checked one last time before an operator sees the number."""

    def test_correct_sum_is_accepted(self):
        alert = AlertDetail.model_validate(
            _alert(
                60.0,
                [
                    {"code": "ZONE_INTRUSION", "weight": 40.0},
                    {"code": "NIGHT_MOVEMENT", "weight": 20.0},
                ],
            )
        )
        assert alert.risk_score == 60.0

    def test_mismatched_sum_is_refused(self):
        """If the stored breakdown does not add up, the explanation we would
        show is a lie. Better a 500 than a confident wrong number."""
        with pytest.raises(ValidationError, match="does not sum"):
            AlertDetail.model_validate(_alert(60.0, [{"code": "ZONE_INTRUSION", "weight": 40.0}]))

    def test_negative_contributions_are_fine(self):
        alert = AlertDetail.model_validate(
            _alert(
                30.0,
                [
                    {"code": "ZONE_INTRUSION", "weight": 40.0},
                    {"code": "LOW_CONFIDENCE", "weight": -10.0},
                ],
            )
        )
        assert alert.risk_score == 30.0

    def test_rounding_tolerance(self):
        AlertDetail.model_validate(
            _alert(33.33, [{"code": "A", "weight": 11.11}, {"code": "B", "weight": 22.22}])
        )


class TestZoneValidation:
    def test_normalised_coordinates_required(self):
        """Zones are stored normalised so a resolution change does not
        invalidate every polygon an operator drew (§6.2)."""
        with pytest.raises(ValidationError, match="normalised"):
            ZoneIn(name="z", kind="area", polygon=[(0.1, 0.1), (640.0, 0.5), (0.2, 0.9)])

    def test_tripwire_needs_exactly_two_points(self):
        with pytest.raises(ValidationError, match="exactly 2 points"):
            ZoneIn(name="w", kind="tripwire", polygon=[(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)])

    def test_area_needs_three_points(self):
        with pytest.raises(ValidationError, match="at least 3 points"):
            ZoneIn(name="a", kind="area", polygon=[(0.1, 0.1), (0.5, 0.5)])

    def test_valid_shapes_pass(self):
        ZoneIn(name="w", kind="tripwire", polygon=[(0.5, 1.0), (0.5, 0.0)])
        ZoneIn(name="a", kind="area", polygon=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)])
        ZoneIn(name="m", kind="mask", polygon=[(0.0, 0.8), (1.0, 0.8), (1.0, 1.0)])

    def test_severity_is_bounded(self):
        with pytest.raises(ValidationError):
            ZoneIn(
                name="a",
                kind="area",
                polygon=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)],
                severity_base=9,
            )
