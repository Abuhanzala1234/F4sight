"""Sinks (§7.13). Fail-soft, but never fail-silent and never lose an alert."""

from __future__ import annotations

import json

import pytest

from drishti_worker.sinks import AlertRecord, FanoutSink, NullSink, SpoolSink


def record(alert_id: str = "a-1") -> AlertRecord:
    return AlertRecord(
        alert_id=alert_id,
        site_id="s",
        site_code="BOP-03",
        camera_id="c",
        camera_code="CAM-01",
        track_id=1,
        zone_id="z",
        kind="ZONE_INTRUSION",
        severity="high",
        risk_score=62.0,
        risk_breakdown=[{"code": "ZONE_INTRUSION", "weight": 62.0, "detail": {}}],
        reason_codes=["ZONE_INTRUSION"],
        ts_utc="2026-09-12T22:00:00+00:00",
        window_start="2026-09-12T21:59:50+00:00",
        window_end="2026-09-12T22:00:00+00:00",
        evidence_hash="ab" * 32,
        evidence_doc={"schema": "drishti.evidence/v1"},
    )


class ExplodingSink:
    def __init__(self, name: str = "postgres"):
        self._name = name
        self.calls = 0

    def emit(self, alert):
        self.calls += 1
        raise ConnectionError("database is on fire")

    @property
    def name(self):
        return self._name


class TestFanout:
    def test_all_succeed(self):
        a, b = NullSink(), NullSink()
        results = FanoutSink(sinks=[a, b]).emit(record())
        assert results == {"null": True}
        assert len(a.emitted) == 1

    def test_one_failure_does_not_stop_the_others(self, tmp_path):
        """Losing the Redis publish must not lose the database row."""
        good = NullSink()
        bad = ExplodingSink("redis")
        fan = FanoutSink(sinks=[bad, good], spool=SpoolSink(tmp_path))
        fan.emit(record())
        assert len(good.emitted) == 1
        assert fan.failures["redis"] == 1

    def test_failures_are_counted_for_health(self, tmp_path):
        bad = ExplodingSink("redis")
        fan = FanoutSink(sinks=[bad], spool=SpoolSink(tmp_path))
        for i in range(3):
            fan.emit(record(f"a-{i}"))
        assert fan.health()["degraded"] is True
        assert fan.failures["redis"] == 3

    def test_alert_is_spooled_when_no_durable_sink_accepts_it(self, tmp_path):
        """§13: never drop an alert."""
        spool = SpoolSink(tmp_path)
        fan = FanoutSink(sinks=[ExplodingSink("postgres")], spool=spool)
        fan.emit(record("a-99"))

        files = list(tmp_path.glob("alerts-*.jsonl"))
        assert len(files) == 1
        spooled = json.loads(files[0].read_text().strip())
        assert spooled["alert_id"] == "a-99"
        assert spooled["evidence_hash"] == "ab" * 32

    def test_no_spool_when_a_durable_sink_succeeded(self, tmp_path):
        class Durable(NullSink):
            @property
            def name(self):
                return "postgres"

        fan = FanoutSink(sinks=[Durable(), ExplodingSink("redis")], spool=SpoolSink(tmp_path))
        fan.emit(record())
        assert list(tmp_path.glob("alerts-*.jsonl")) == []

    def test_fail_fast_mode_propagates(self):
        fan = FanoutSink(sinks=[ExplodingSink("postgres")], fail_soft=False)
        with pytest.raises(ConnectionError):
            fan.emit(record())


class TestSerialisation:
    def test_record_round_trips_through_json(self):
        """AlertRecord crosses a process boundary, so it must serialise."""
        payload = json.loads(json.dumps(record().as_dict()))
        assert payload["kind"] == "ZONE_INTRUSION"
        assert payload["risk_score"] == 62.0

    def test_summary_is_small_enough_for_a_websocket(self):
        summary = record().summary()
        assert "evidence_doc" not in summary
        assert set(summary) == {
            "alert_id",
            "site_code",
            "camera_code",
            "kind",
            "severity",
            "risk_score",
            "reason_codes",
            "ts_utc",
            "status",
        }
