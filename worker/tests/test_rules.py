"""Rules and debouncing (§7.7). Blocker #1 is the centrepiece."""

from __future__ import annotations

from datetime import timedelta

import pytest
from helpers import T0, make_track

from drishti_worker.risk import RiskConfig
from drishti_worker.rules import (
    DebounceConfig,
    Debouncer,
    Decision,
    RuleConfig,
    RuleEngine,
    in_window,
    zone_active,
)
from drishti_worker.types import ZoneKind, ZoneRuntime

RULES = RuleConfig()
RISK = RiskConfig()


class TestGating:
    """§7.7.1 — this single check removes most detector flicker."""

    def test_young_track_is_gated(self):
        engine = RuleEngine(RULES, RISK)
        passes, reason = engine.gate(make_track(hits=1, age=1))
        assert not passes
        assert "hits" in reason

    def test_low_confidence_track_is_gated(self):
        engine = RuleEngine(RULES, RISK)
        passes, reason = engine.gate(make_track(conf=0.1))
        assert not passes
        assert "max_conf" in reason

    def test_established_track_passes(self):
        engine = RuleEngine(RULES, RISK)
        passes, _ = engine.gate(make_track(hits=10, age=20, conf=0.9))
        assert passes


class TestZoneRules:
    def test_intrusion_fires_on_entry_only(self, camera, area_zone):
        """Being inside is not news every frame — otherwise the debouncer has
        to mop up an avoidable mess."""
        engine = RuleEngine(RULES, RISK)
        inside = make_track(box=(280.0, 300.0, 320.0, 400.0))

        first = engine.evaluate([inside], camera, [area_zone], T0, "day")
        assert any(s.code == "ZONE_INTRUSION" for s in first[inside.track_id])

        second = engine.evaluate([inside], camera, [area_zone], T0, "day")
        assert not any(
            s.code == "ZONE_INTRUSION" for s in second.get(inside.track_id, [])
        )

    def test_outside_the_zone_is_silent(self, camera, area_zone):
        engine = RuleEngine(RULES, RISK)
        outside = make_track(box=(900.0, 600.0, 940.0, 700.0))
        assert engine.evaluate([outside], camera, [area_zone], T0, "day") == {}

    def test_mask_zone_erases_the_detection(self, camera, area_zone, mask_zone):
        """§7.7.2 — a detection inside a mask does not exist. This is the
        flapping tree and the public road."""
        engine = RuleEngine(RULES, RISK)
        on_the_road = make_track(box=(280.0, 600.0, 320.0, 700.0))
        assert (
            engine.evaluate([on_the_road], camera, [area_zone, mask_zone], T0, "day")
            == {}
        )

    def test_class_filter(self, camera, area_zone):
        engine = RuleEngine(RULES, RISK)
        vehicle = make_track(cls="vehicle", box=(280.0, 300.0, 320.0, 400.0))
        signals = engine.evaluate([vehicle], camera, [area_zone], T0, "day")
        assert not any(
            s.code == "ZONE_INTRUSION" for s in signals.get(vehicle.track_id, [])
        )

    def test_severity_base_scales_the_weight(self, camera):
        engine = RuleEngine(RULES, RISK)
        hot = ZoneRuntime(
            zone_id="z",
            name="Hot",
            kind=ZoneKind.AREA,
            polygon=((100.0, 100.0), (500.0, 100.0), (500.0, 500.0), (100.0, 500.0)),
            classes=("person",),
            severity_base=5,
        )
        track = make_track(box=(280.0, 300.0, 320.0, 400.0))
        signals = engine.evaluate([track], camera, [hot], T0, "day")[track.track_id]
        intrusion = next(s for s in signals if s.code == "ZONE_INTRUSION")
        assert intrusion.weight == pytest.approx(40.0 * 5 / 3, abs=0.01)


class TestTripwire:
    def test_crossing_fires_with_direction(self, camera, tripwire_zone):
        engine = RuleEngine(RULES, RISK)
        crossing = make_track(
            box=(590.0, 200.0, 630.0, 400.0), history=((560.0, 400.0), (610.0, 400.0))
        )
        signals = engine.evaluate([crossing], camera, [tripwire_zone], T0, "day")
        wire = next(s for s in signals[crossing.track_id] if s.code == "TRIPWIRE_CROSS")
        assert wire.detail["direction"] in ("in", "out")

    def test_jitter_below_threshold_is_ignored(self, camera, tripwire_zone):
        """A foot-point wobbling either side of the line is a detector twitch,
        not a crossing."""
        engine = RuleEngine(RULES, RISK)
        # Box centre is 601, so the foot-point moves 599 -> 601: 2 px of wobble.
        jitter = make_track(
            box=(581.0, 200.0, 621.0, 400.0), history=((599.0, 400.0), (601.0, 400.0))
        )
        signals = engine.evaluate([jitter], camera, [tripwire_zone], T0, "day")
        assert not any(
            s.code == "TRIPWIRE_CROSS" for s in signals.get(jitter.track_id, [])
        )


class TestNightMovement:
    """NIGHT_MOVEMENT is CONTEXTUAL: it enriches a real event but never raises
    one alone. Otherwise every patrol produces an alert every 45 seconds all
    night, which is blocker #1 wearing a different hat. P3."""

    def test_does_not_raise_an_alert_on_its_own(self, camera):
        engine = RuleEngine(RULES, RISK)
        track = make_track()
        assert engine.evaluate([track], camera, [], T0, "night") == {}

    def test_attaches_to_a_real_event(self, camera, area_zone):
        engine = RuleEngine(RULES, RISK)
        intruder = make_track(box=(280.0, 300.0, 320.0, 400.0))
        signals = engine.evaluate([intruder], camera, [area_zone], T0, "night")
        codes = {s.code for s in signals[intruder.track_id]}
        assert "ZONE_INTRUSION" in codes
        assert "NIGHT_MOVEMENT" in codes

    def test_requires_both_clock_and_evqm(self, camera, area_zone):
        """EVQM alone would fire inside a dark warehouse at noon; the clock
        alone would fire under floodlights."""
        at_noon = RuleEngine(RULES, RISK).evaluate(
            [make_track(box=(280.0, 300.0, 320.0, 400.0))],
            camera,
            [area_zone],
            T0.replace(hour=12),
            "night",
        )
        assert "NIGHT_MOVEMENT" not in {s.code for s in next(iter(at_noon.values()))}

        in_daylight = RuleEngine(RULES, RISK).evaluate(
            [make_track(box=(280.0, 300.0, 320.0, 400.0))],
            camera,
            [area_zone],
            T0,
            "day",
        )
        assert "NIGHT_MOVEMENT" not in {
            s.code for s in next(iter(in_daylight.values()))
        }


class TestSchedules:
    def test_midnight_spanning_window(self):
        assert in_window(T0, "18:30", "06:00")  # 22:00
        assert in_window(T0.replace(hour=3), "18:30", "06:00")
        assert not in_window(T0.replace(hour=12), "18:30", "06:00")

    def test_same_day_window(self):
        assert in_window(T0.replace(hour=10), "09:00", "17:00")
        assert not in_window(T0.replace(hour=20), "09:00", "17:00")

    def test_zone_outside_schedule_contributes_nothing(self, area_zone):
        scheduled = ZoneRuntime(
            zone_id=area_zone.zone_id,
            name=area_zone.name,
            kind=area_zone.kind,
            polygon=area_zone.polygon,
            classes=area_zone.classes,
            schedule={"windows": [{"from": "09:00", "to": "17:00"}]},
        )
        assert not zone_active(scheduled, T0)  # 22:00
        assert zone_active(scheduled, T0.replace(hour=10))

    def test_disabled_zone_is_never_active(self, area_zone):
        disabled = ZoneRuntime(
            zone_id="z",
            name="x",
            kind=ZoneKind.AREA,
            polygon=area_zone.polygon,
            enabled=False,
        )
        assert not zone_active(disabled, T0)


class TestDebouncer:
    """Blocker #1."""

    def test_standing_on_a_tripwire_does_not_spam(self):
        """The canonical failure: 360 candidate firings over a minute must not
        become 360 alerts."""
        d = Debouncer(DebounceConfig(cooldown_s=45.0, escalate_after_s=120.0))
        emitted = 0
        for i in range(360):  # 6 fps for 60 s
            decision = d.submit(
                camera_id="C1",
                track_id=7,
                zone_id="z1",
                rule_code="TRIPWIRE_CROSS",
                now=T0 + timedelta(seconds=i / 6),
                alert_id=f"a{i}",
            )
            if decision.should_write:
                emitted += 1
        assert emitted == 2  # one at t=0, one when the cooldown expires

    def test_a_second_intruder_is_not_suppressed(self):
        """The debounce key includes track_id. Getting this wrong turns a spam
        bug into a missed-intrusion bug."""
        d = Debouncer()
        first = d.submit(
            camera_id="C1",
            track_id=1,
            zone_id="z1",
            rule_code="ZONE_INTRUSION",
            now=T0,
            alert_id="a1",
        )
        second = d.submit(
            camera_id="C1",
            track_id=2,
            zone_id="z1",
            rule_code="ZONE_INTRUSION",
            now=T0 + timedelta(seconds=1),
            alert_id="a2",
        )
        assert first.decision is Decision.EMIT
        assert second.decision is Decision.EMIT

    def test_same_track_in_two_zones_correlates_into_one_alert(self):
        """One person crossing two zones in two seconds is one event with two
        reason codes. Correlation is keyed on the TRACK, deliberately (§7.7.4)."""
        d = Debouncer()
        a = d.submit(
            camera_id="C1",
            track_id=1,
            zone_id="z1",
            rule_code="ZONE_INTRUSION",
            now=T0,
            alert_id="a",
        )
        b = d.submit(
            camera_id="C1",
            track_id=1,
            zone_id="z2",
            rule_code="ZONE_INTRUSION",
            now=T0 + timedelta(seconds=2),
            alert_id="b",
        )
        assert a.decision is Decision.EMIT
        assert b.decision is Decision.MERGE and b.merge_into == "a"

    def test_different_tracks_in_different_zones_are_independent(self):
        d = Debouncer()
        a = d.submit(
            camera_id="C1",
            track_id=1,
            zone_id="z1",
            rule_code="ZONE_INTRUSION",
            now=T0,
            alert_id="a",
        )
        b = d.submit(
            camera_id="C1",
            track_id=2,
            zone_id="z2",
            rule_code="ZONE_INTRUSION",
            now=T0,
            alert_id="b",
        )
        assert a.decision is Decision.EMIT and b.decision is Decision.EMIT

    def test_correlation_merges_rules_on_one_track(self):
        """Intrusion AND night AND loitering is one alert with three reason
        codes, not three alerts."""
        d = Debouncer(DebounceConfig(correlate_window_s=8.0))
        d.submit(
            camera_id="C1",
            track_id=5,
            zone_id="z1",
            rule_code="ZONE_INTRUSION",
            now=T0,
            alert_id="A",
        )
        merged = d.submit(
            camera_id="C1",
            track_id=5,
            zone_id=None,
            rule_code="NIGHT_MOVEMENT",
            now=T0 + timedelta(seconds=2),
            alert_id="B",
        )
        assert merged.decision is Decision.MERGE
        assert merged.merge_into == "A"

    def test_correlation_expires(self):
        d = Debouncer(DebounceConfig(correlate_window_s=8.0))
        d.submit(
            camera_id="C1",
            track_id=5,
            zone_id="z1",
            rule_code="ZONE_INTRUSION",
            now=T0,
            alert_id="A",
        )
        later = d.submit(
            camera_id="C1",
            track_id=5,
            zone_id=None,
            rule_code="NIGHT_MOVEMENT",
            now=T0 + timedelta(seconds=30),
            alert_id="B",
        )
        assert later.decision is Decision.EMIT

    def test_ongoing_situation_escalates(self):
        """Still happening two minutes later is not spam, it is a situation."""
        d = Debouncer(DebounceConfig(cooldown_s=45.0, escalate_after_s=120.0))
        d.submit(
            camera_id="C1",
            track_id=1,
            zone_id="z1",
            rule_code="LOITER",
            now=T0,
            alert_id="a",
        )
        escalation = d.submit(
            camera_id="C1",
            track_id=1,
            zone_id="z1",
            rule_code="LOITER",
            now=T0 + timedelta(seconds=130),
            alert_id="b",
        )
        assert escalation.decision is Decision.ESCALATE

    def test_rate_ceiling_is_last_resort(self):
        d = Debouncer(DebounceConfig(max_alerts_per_camera_per_min=3))
        decisions = [
            d.submit(
                camera_id="C1",
                track_id=i,
                zone_id="z",
                rule_code="ZONE_INTRUSION",
                now=T0 + timedelta(seconds=i),
                alert_id=f"a{i}",
            ).decision
            for i in range(6)
        ]
        assert decisions.count(Decision.EMIT) == 3
        assert Decision.RATE_LIMITED in decisions

    def test_rate_window_slides(self):
        d = Debouncer(DebounceConfig(max_alerts_per_camera_per_min=2))
        for i in range(2):
            d.submit(
                camera_id="C1",
                track_id=i,
                zone_id="z",
                rule_code="ZONE_INTRUSION",
                now=T0,
                alert_id=f"a{i}",
            )
        blocked = d.submit(
            camera_id="C1",
            track_id=9,
            zone_id="z",
            rule_code="ZONE_INTRUSION",
            now=T0,
            alert_id="x",
        )
        assert blocked.decision is Decision.RATE_LIMITED
        allowed = d.submit(
            camera_id="C1",
            track_id=9,
            zone_id="z",
            rule_code="ZONE_INTRUSION",
            now=T0 + timedelta(seconds=61),
            alert_id="y",
        )
        assert allowed.decision is Decision.EMIT

    def test_cameras_are_independent(self):
        d = Debouncer(DebounceConfig(max_alerts_per_camera_per_min=1))
        a = d.submit(
            camera_id="C1", track_id=1, zone_id="z", rule_code="R", now=T0, alert_id="a"
        )
        b = d.submit(
            camera_id="C2", track_id=1, zone_id="z", rule_code="R", now=T0, alert_id="b"
        )
        assert a.decision is Decision.EMIT and b.decision is Decision.EMIT

    def test_close_track_clears_state(self):
        d = Debouncer()
        d.submit(
            camera_id="C1", track_id=1, zone_id="z", rule_code="R", now=T0, alert_id="a"
        )
        d.close_track("C1", 1)
        again = d.submit(
            camera_id="C1",
            track_id=1,
            zone_id="z",
            rule_code="R",
            now=T0 + timedelta(seconds=1),
            alert_id="b",
        )
        assert again.decision is Decision.EMIT


def test_rule_config_from_mapping():
    cfg = RuleConfig.from_mapping(
        {"rules": {"gating": {"min_hits": 7}, "loiter": {"seconds": 12}}}
    )
    assert cfg.min_hits == 7
    assert cfg.loiter_seconds == 12
    assert cfg.min_track_age_frames == 8
