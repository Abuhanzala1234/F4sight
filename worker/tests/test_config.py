"""Config loading, merge order and versioning (§11)."""

from __future__ import annotations

import pytest

from drishti_worker.config import (
    AppConfig,
    apply_env_overrides,
    config_version,
    deep_merge,
    iter_flat,
    load_config,
)


class TestMerge:
    def test_mappings_merge_recursively(self):
        base = {"a": {"x": 1, "y": 2}, "b": 1}
        override = {"a": {"y": 3}}
        assert deep_merge(base, override) == {"a": {"x": 1, "y": 3}, "b": 1}

    def test_lists_replace_rather_than_append(self):
        """A site overriding reconnect_backoff_s means USE THIS SCHEDULE, not
        append to the default one."""
        assert deep_merge({"k": [1, 2, 3]}, {"k": [9]}) == {"k": [9]}

    def test_base_is_not_mutated(self):
        base = {"a": {"x": 1}}
        deep_merge(base, {"a": {"x": 2}})
        assert base == {"a": {"x": 1}}


class TestEnvOverrides:
    def test_double_underscore_separates_levels(self):
        out = apply_env_overrides({}, {"DRISHTI__RULES__LOITER__SECONDS": "45"})
        assert out["rules"]["loiter"]["seconds"] == 45

    def test_single_underscore_survives_in_key_names(self):
        out = apply_env_overrides({}, {"DRISHTI__INGEST__ANALYTICS_FPS": "12"})
        assert out["ingest"]["analytics_fps"] == 12

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("true", True),
            ("false", False),
            ("12", 12),
            ("1.5", 1.5),
            ("null", None),
            ("a,b", ["a", "b"]),
            ("text", "text"),
        ],
    )
    def test_scalar_coercion(self, raw, expected):
        out = apply_env_overrides({}, {"DRISHTI__K": raw})
        assert out["k"] == expected

    def test_unprefixed_variables_are_ignored(self):
        out = apply_env_overrides({"a": 1}, {"PATH": "/usr/bin", "HOME": "/root"})
        assert out == {"a": 1}


class TestVersioning:
    def test_stable_under_key_reordering(self):
        """Without canonicalisation a dict reordering would change the version
        and every alert would claim a different ruleset produced it."""
        a = {"x": 1, "y": {"p": 2, "q": 3}}
        b = {"y": {"q": 3, "p": 2}, "x": 1}
        assert config_version(a) == config_version(b)

    def test_changes_when_a_threshold_changes(self):
        base = {"rules": {"loiter": {"seconds": 30}}}
        assert config_version(base) != config_version(
            {"rules": {"loiter": {"seconds": 31}}}
        )

    def test_is_a_sha256_hex_digest(self):
        version = config_version({"a": 1})
        assert len(version) == 64
        assert all(c in "0123456789abcdef" for c in version)


class TestRealConfigDirectory:
    def test_loads_the_shipped_config(self):
        cfg = load_config("config", profile="laptop", site="BOP-03", environ={})
        assert cfg.profile == "laptop"
        assert cfg.site == "BOP-03"
        assert len(cfg.version) == 64
        assert "defaults.yaml" in cfg.sources
        assert "profiles/laptop.yaml" in cfg.sources
        assert "sites/BOP-03.yaml" in cfg.sources

    def test_profile_overrides_defaults(self):
        laptop = load_config("config", profile="laptop", environ={})
        bop = load_config("config", profile="bop", environ={})
        assert laptop.get("detector.input_size") == [480, 480]
        assert bop.get("detector.input_size") == [640, 640]
        assert bop.get("ingest.analytics_fps") > laptop.get("ingest.analytics_fps")

    def test_site_overrides_profile(self):
        cfg = load_config("config", profile="laptop", site="BOP-03", environ={})
        assert cfg.get("rules.loiter.seconds") == 25  # site tightens the default 30

    def test_unknown_profile_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("config", profile="does-not-exist", environ={})

    def test_faces_are_disabled_by_default(self):
        """P6. If this test ever fails, someone has changed a privacy default."""
        cfg = load_config("config", profile="laptop", environ={})
        assert cfg.get("faces.enabled") is False

    def test_plate_storage_is_hash_only_by_default(self):
        cfg = load_config("config", profile="laptop", environ={})
        assert cfg.get("anpr.storage.hash_only") is True

    def test_evidence_enhancement_is_forbidden_by_default(self):
        cfg = load_config("config", profile="laptop", environ={})
        assert cfg.get("evidence.allow_enhanced_evidence") is False

    def test_hysteresis_is_configured(self):
        """Blocker #5 has specific required values."""
        cfg = load_config("config", profile="laptop", environ={})
        assert cfg.get("evqm.hysteresis.enter_samples") == 3
        assert cfg.get("evqm.hysteresis.exit_samples") == 5


class TestPrivacyInvariants:
    """The loader refuses to start on a config that violates a privacy rule."""

    def test_faces_on_with_retention_is_refused(self):
        with pytest.raises(ValueError, match="privacy invariant"):
            load_config(
                "config",
                profile="laptop",
                environ={},
                overrides={
                    "faces": {"enabled": True, "retain_non_matching_embeddings": True}
                },
            )

    def test_plate_plaintext_storage_is_refused(self):
        with pytest.raises(ValueError, match="privacy invariant"):
            load_config(
                "config",
                profile="laptop",
                environ={},
                overrides={"anpr": {"enabled": True, "storage": {"hash_only": False}}},
            )

    def test_enhanced_evidence_is_refused(self):
        with pytest.raises(ValueError, match="evidence invariant"):
            load_config(
                "config",
                profile="laptop",
                environ={},
                overrides={"evidence": {"allow_enhanced_evidence": True}},
            )


class TestAccess:
    def test_dotted_get(self):
        cfg = AppConfig(data={"a": {"b": {"c": 7}}}, version="x", profile="p", site="s")
        assert cfg.get("a.b.c") == 7

    def test_missing_key_raises_rather_than_returning_none(self):
        cfg = AppConfig(data={}, version="x", profile="p", site="s")
        with pytest.raises(KeyError):
            cfg.get("nope")

    def test_explicit_none_default_is_honoured(self):
        cfg = AppConfig(data={}, version="x", profile="p", site="s")
        assert cfg.get("nope", None) is None

    def test_iter_flat(self):
        flat = dict(iter_flat({"a": {"b": 1}, "c": 2}))
        assert flat == {"a.b": 1, "c": 2}
