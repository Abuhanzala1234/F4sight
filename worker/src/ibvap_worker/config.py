"""Configuration loading and versioning (BUILD_SPEC §11).

CLAUDE.md: *config over constants — if a number could plausibly need tuning at a
BOP, it belongs in config/, not in the source.*

Merge order, later wins::

    defaults.yaml
      -> <domain>.yaml (detector, evqm, rules, risk, anpr, gestures, weapons,
                        faces, ledger)
      -> profiles/<profile>.yaml
      -> sites/<SITE>.yaml
      -> environment (IBVAP__SECTION__KEY=value)
      -> CLI overrides

The merged document is hashed to produce ``config_version``, which is written
onto every alert. That is what makes an alert attributable: six weeks later,
looking at a disputed detection, you can say exactly which thresholds produced
it. A running worker never reloads config silently — changing a threshold means
a restart and a new ``config_version``.

Hashing reuses the evidence canonicaliser, so ``config_version`` is stable
across dict ordering and float formatting for exactly the same reasons alert
hashes are (§7.11).
"""

from __future__ import annotations

import copy
import os
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .evidence import canonicalise

__all__ = [
    "AppConfig",
    "apply_env_overrides",
    "config_version",
    "deep_merge",
    "load_config",
]

ENV_PREFIX = "IBVAP__"

# Domain files merged on top of defaults, in this order.
DOMAIN_FILES = (
    "detector",
    "evqm",
    "rules",
    "risk",
    "anpr",
    "gestures",
    "weapons",
    "faces",
    "ledger",
)

# Keys whose values are secrets: never logged, never written into evidence, and
# never read from config/ (§11, §12).
SECRET_ENV_KEYS = (
    "DB_PASSWORD",
    "MINIO_SECRET_KEY",
    "JWT_SECRET",
    "PLATE_HMAC_KEY",
    "EVIDENCE_ENC_KEY",
)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive merge. Scalars and lists replace; mappings merge.

    Lists replace rather than concatenate on purpose: a site that overrides
    ``reconnect_backoff_s`` means *use this schedule*, not *append to the
    default one*.
    """
    out: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce_scalar(text: str) -> Any:
    """Turn an environment string into the type it obviously is."""
    lowered = text.strip().lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", ""):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if "," in text:
        return [_coerce_scalar(part) for part in text.split(",")]
    return text


def apply_env_overrides(
    cfg: Mapping[str, Any], environ: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Apply ``IBVAP__SECTION__KEY=value`` overrides.

    Double underscore separates levels, so single-underscore key names
    (``analytics_fps``) survive intact. Keys are lower-cased to match the YAML.
    """
    env = dict(os.environ if environ is None else environ)
    out = copy.deepcopy(dict(cfg))
    for raw_key, raw_value in sorted(env.items()):
        if not raw_key.startswith(ENV_PREFIX):
            continue
        path = [part.lower() for part in raw_key[len(ENV_PREFIX) :].split("__") if part]
        if not path:
            continue
        cursor: MutableMapping[str, Any] = out
        for part in path[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, MutableMapping):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[path[-1]] = _coerce_scalar(raw_value)
    return out


def config_version(cfg: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical form of the merged config.

    Canonicalisation matters here for the same reason it matters for evidence:
    without it, a dict reordering would change the version and every alert would
    claim a different ruleset produced it.
    """
    return sha256(canonicalise(_jsonable(cfg))).hexdigest()


def _jsonable(value: Any) -> Any:
    """Make a YAML-derived structure canonicalisable.

    YAML happily produces dates, tuples and non-string keys; the canonicaliser
    accepts only JSON types and says so loudly. Converting here keeps that
    strictness where it belongs.
    """
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load config; run `make install`") from exc
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level, got {type(data)}")
    return data


class _Missing:
    """Sentinel distinguishing 'no default given' from 'default is None'."""

    def __repr__(self) -> str:  # pragma: no cover
        return "<missing>"


_MISSING = _Missing()


@dataclass(frozen=True, slots=True)
class AppConfig:
    """The merged, frozen configuration plus its version.

    Access is by path (``cfg.get("detector.input_size")``) so that callers do
    not carry nested dict indexing around, and a missing key with no default
    raises rather than returning None and failing three modules later.
    """

    data: Mapping[str, Any]
    version: str
    profile: str
    site: str
    sources: tuple[str, ...] = ()

    def get(self, path: str, default: Any = _MISSING) -> Any:
        cursor: Any = self.data
        for part in path.split("."):
            if not isinstance(cursor, Mapping) or part not in cursor:
                if default is _MISSING:
                    raise KeyError(f"config key not found: {path!r}")
                return default
            cursor = cursor[part]
        return cursor

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.get(name, {})
        return value if isinstance(value, Mapping) else {}

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.data))

    def summary(self) -> str:
        return (
            f"profile={self.profile} site={self.site} "
            f"config_version={self.version[:12]}… sources={len(self.sources)}"
        )


def load_config(
    config_dir: str | Path = "config",
    *,
    profile: str | None = None,
    site: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load, merge, version and freeze the configuration."""
    root = Path(config_dir)
    if not root.exists():
        raise FileNotFoundError(f"config directory not found: {root.resolve()}")

    env = dict(os.environ if environ is None else environ)
    sources: list[str] = []

    merged = _read_yaml(root / "defaults.yaml")
    sources.append("defaults.yaml")

    for domain in DOMAIN_FILES:
        path = root / f"{domain}.yaml"
        if path.exists():
            merged = deep_merge(merged, _read_yaml(path))
            sources.append(path.name)

    chosen_profile = (
        profile or env.get("IBVAP_PROFILE") or merged.get("runtime", {}).get("profile", "laptop")
    )
    profile_path = root / "profiles" / f"{chosen_profile}.yaml"
    if profile_path.exists():
        merged = deep_merge(merged, _read_yaml(profile_path))
        sources.append(f"profiles/{profile_path.name}")
    else:
        raise FileNotFoundError(
            f"unknown profile {chosen_profile!r}: {profile_path} does not exist"
        )

    chosen_site = site or env.get("IBVAP_SITE") or merged.get("runtime", {}).get("site", "")
    if chosen_site:
        site_path = root / "sites" / f"{chosen_site}.yaml"
        if site_path.exists():
            merged = deep_merge(merged, _read_yaml(site_path))
            sources.append(f"sites/{site_path.name}")

    merged = apply_env_overrides(merged, env)
    if overrides:
        merged = deep_merge(merged, overrides)
        sources.append("cli")

    merged.setdefault("runtime", {})
    merged["runtime"]["profile"] = chosen_profile
    merged["runtime"]["site"] = chosen_site

    _assert_privacy_invariants(merged)

    return AppConfig(
        data=merged,
        version=config_version(merged),
        profile=str(chosen_profile),
        site=str(chosen_site),
        sources=tuple(sources),
    )


def _assert_privacy_invariants(cfg: Mapping[str, Any]) -> None:
    """Refuse to start on a config that violates a privacy invariant.

    A site file must not be able to quietly weaken these. Turning faces on is a
    legitimate, audited decision — but it has to be made in ``faces.yaml`` with
    retention explicitly configured, not as a side effect of a site override.
    """
    faces = cfg.get("faces", {}) or {}
    if faces.get("enabled") and faces.get("retain_non_matching_embeddings"):
        raise ValueError(
            "config violates a privacy invariant: faces.enabled is true while "
            "faces.retain_non_matching_embeddings is true. Non-matching embeddings "
            "must be destroyed at track close (P6, BUILD_SPEC §7.10)."
        )

    anpr = cfg.get("anpr", {}) or {}
    storage = anpr.get("storage", {}) or {}
    if anpr.get("enabled") and storage.get("hash_only") is False:
        raise ValueError(
            "config violates a privacy invariant: anpr.storage.hash_only is false. "
            "Plate text is stored as HMAC outside of fired-alert evidence (P6)."
        )

    evidence = cfg.get("evidence", {}) or {}
    if evidence.get("allow_enhanced_evidence"):
        raise ValueError(
            "config violates an evidence invariant: evidence.allow_enhanced_evidence "
            "is true. Snapshots and clips must be original, unenhanced frames (P4)."
        )


def iter_flat(cfg: Mapping[str, Any], prefix: str = "") -> Iterable[tuple[str, Any]]:
    """Flatten config to dotted paths — used by the dashboard's config-diff view."""
    for key, value in cfg.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield from iter_flat(value, path)
        else:
            yield path, value
