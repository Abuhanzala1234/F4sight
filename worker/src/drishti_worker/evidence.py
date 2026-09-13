"""Evidence assembly, canonicalisation and hashing (BUILD_SPEC §7.11).

Blocker #7. From CLAUDE.md:

    Canonicalise (JCS, RFC 8785) before hashing, exclude evidence_hash and
    ledger fields, hash in the worker at assembly time. Get this wrong and
    verification silently fails later, which is worse than failing loudly.

Why canonicalisation at all: two JSON documents can be semantically identical
and byte-different — key order, whitespace, ``1.0`` vs ``1``, ``\\u00e9`` vs a
literal é. Hash the bytes and you have hashed an accident of serialisation.
Six months later a different library version reorders a dict and every piece of
evidence you collected becomes unverifiable, with no error anywhere.

So: RFC 8785 JSON Canonicalisation Scheme.

* UTF-8, no insignificant whitespace
* object keys sorted by **UTF-16 code unit**, not by code point (they differ
  above the BMP, and getting this wrong is invisible until it isn't)
* numbers formatted by the ECMAScript ``Number::toString`` algorithm, which is
  where every naive implementation breaks: ``1e21``, ``1e-7``, ``-0``, and
  integral floats that must print without a decimal point

Pure module — standard library only, no numpy — so the RFC test vectors run in
milliseconds with no environment.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "EXCLUDED_FIELDS",
    "VerificationResult",
    "assemble",
    "canonicalise",
    "es_number_to_string",
    "evidence_hash",
    "strip_excluded",
    "verify",
]

# Fields excluded from the hash. They cannot be inside it: evidence_hash is the
# output, and ledger is filled in AFTER anchoring, which happens after hashing.
EXCLUDED_FIELDS: frozenset[str] = frozenset({"evidence_hash", "ledger"})

_ESCAPES = {
    0x08: "\\b",
    0x09: "\\t",
    0x0A: "\\n",
    0x0C: "\\f",
    0x0D: "\\r",
    0x22: '\\"',
    0x5C: "\\\\",
}


# ---------------------------------------------------------------------------
# ECMAScript number formatting (RFC 8785 §3.2.2.3)
# ---------------------------------------------------------------------------


def _decompose(x: float) -> tuple[str, int]:
    """Return (digits, n) with value == 0.digits * 10**n, digits having no
    leading or trailing zeros.

    Python's ``repr`` already gives the shortest round-tripping decimal form,
    which is exactly the ``s``/``k`` the ECMAScript algorithm asks for. We only
    have to re-normalise it.
    """
    text = repr(abs(x))
    if "e" in text or "E" in text:
        mantissa, _, exp_text = (
            text.partition("e") if "e" in text else text.partition("E")
        )
        exp = int(exp_text)
    else:
        mantissa, exp = text, 0

    if "." in mantissa:
        int_part, _, frac_part = mantissa.partition(".")
    else:
        int_part, frac_part = mantissa, ""

    digits = int_part + frac_part
    n = len(int_part) + exp

    # Strip leading zeros (each one shifts the decimal exponent down).
    lead = len(digits) - len(digits.lstrip("0"))
    digits = digits[lead:]
    n -= lead

    digits = digits.rstrip("0")
    if not digits:  # the value was zero
        return "0", 1
    return digits, n


def es_number_to_string(x: float | int) -> str:
    """ECMAScript ``Number::toString`` — the serialisation JCS mandates.

    The cases that matter, and that plain ``repr``/``json.dumps`` get wrong:

        1.0      -> "1"        (not "1.0")
        -0.0     -> "0"        (not "-0.0")
        1e21     -> "1e+21"    (not "1e+21" by accident — by rule)
        1e-7     -> "1e-7"     (not "1e-07")
        0.000001 -> "0.000001" (exponent form only below 1e-6)
    """
    if isinstance(x, bool):  # bool is an int subclass; JSON booleans are not numbers
        raise TypeError("bool is not a JSON number")
    if isinstance(x, int):
        return str(x)
    if math.isnan(x) or math.isinf(x):
        raise ValueError(f"{x!r} is not representable in JSON; refusing to hash it")
    if x == 0.0:
        return "0"  # covers -0.0, which JCS normalises to "0"

    sign = "-" if x < 0 else ""
    digits, n = _decompose(x)
    k = len(digits)

    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    # Exponential form.
    e = n - 1
    exp_part = f"e{'+' if e >= 0 else '-'}{abs(e)}"
    if k == 1:
        return sign + digits + exp_part
    return sign + digits[0] + "." + digits[1:] + exp_part


# ---------------------------------------------------------------------------
# Canonical serialisation
# ---------------------------------------------------------------------------


def _serialise_string(s: str) -> str:
    out = ['"']
    for ch in s:
        cp = ord(ch)
        esc = _ESCAPES.get(cp)
        if esc is not None:
            out.append(esc)
        elif cp < 0x20:
            out.append(f"\\u{cp:04x}")
        else:
            out.append(ch)  # everything else stays literal UTF-8, incl. non-ASCII
    out.append('"')
    return "".join(out)


def _utf16_sort_key(s: str) -> bytes:
    """Sort key giving UTF-16 code-unit order (RFC 8785 §3.2.3).

    Python compares strings by code point. For characters outside the BMP the
    two orders disagree, because UTF-16 represents them as surrogate pairs whose
    code units (0xD800-0xDFFF) sort below ordinary BMP characters above 0xE000.
    Encoding to UTF-16BE and comparing bytes reproduces the required order.
    """
    return s.encode("utf-16-be", errors="surrogatepass")


def _serialise(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _serialise_string(value)
    if type(value) is int or type(value) is float:
        # Exact type, deliberately not isinstance(). numpy.float64 IS a
        # subclass of float and would pass an isinstance check here, but its
        # repr() -- what es_number_to_string relies on -- prints
        # "np.float64(123.4)" on numpy 2.x, not "123.4". That silently
        # corrupted every evidence_hash computed from a numpy-tainted value
        # while the value still looked completely normal once the standard
        # JSON encoder (numpy-agnostic) wrote it to storage -- exactly the
        # "fails silently" case CLAUDE.md calls out. A value that reaches here
        # not already a plain Python int/float is a bug at the call site, and
        # the fix is float(x)/int(x) there, not a wider check here.
        return es_number_to_string(value)
    if isinstance(value, (int, float)):
        raise TypeError(
            f"{type(value).__module__}.{type(value).__name__} reached the canonicaliser "
            f"where a plain float/int was expected (value={value!r}). This type's repr() "
            "does not match its JSON serialisation, which silently produces a wrong "
            "evidence_hash. Cast to float()/int() at the call site before this point."
        )
    if isinstance(value, datetime):
        raise TypeError(
            "datetime reached the canonicaliser; convert to an ISO-8601 string "
            "at assembly time so the exact serialisation is explicit"
        )
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda kv: _utf16_sort_key(str(kv[0])))
        inner = ",".join(
            f"{_serialise_string(str(k))}:{_serialise(v)}" for k, v in items
        )
        return "{" + inner + "}"
    if isinstance(value, (list, tuple)) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    ):
        return "[" + ",".join(_serialise(v) for v in value) + "]"
    raise TypeError(f"cannot canonicalise {type(value).__name__}: {value!r}")


def canonicalise(doc: Any) -> bytes:
    """RFC 8785 canonical UTF-8 bytes for ``doc``."""
    return _serialise(doc).encode("utf-8")


def strip_excluded(
    doc: Mapping[str, Any], excluded: frozenset[str] = EXCLUDED_FIELDS
) -> dict[str, Any]:
    """Remove excluded keys at the TOP LEVEL only.

    Deliberately not recursive: a nested field legitimately named ``ledger``
    (say, inside a free-text note) must still be covered by the hash. Only the
    document's own envelope fields are excluded.
    """
    return {k: v for k, v in doc.items() if k not in excluded}


def evidence_hash(doc: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical form, excluding ``evidence_hash`` and ``ledger``."""
    return hashlib.sha256(canonicalise(strip_excluded(doc))).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Streaming digest, so hashing a 200 MB clip does not cost 200 MB of RSS."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble(
    *,
    alert_id: str,
    site: Mapping[str, Any],
    camera: Mapping[str, Any],
    detection: Mapping[str, Any],
    risk: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    config_version: str,
    spec_version: str,
    worker_version: str,
    created_at: str,
) -> dict[str, Any]:
    """Build the evidence document that will be hashed and anchored.

    Every media file's own SHA-256 is *inside* this document, so altering a
    snapshot on disk invalidates the alert hash (§7.11). That is the whole
    chain: file bytes -> item digest -> evidence doc -> Merkle leaf -> root ->
    ledger.

    ``evidence_hash`` and ``ledger`` are added by the caller afterwards and are
    excluded from the hash by construction.
    """
    for item in items:
        if not item.get("sha256"):
            raise ValueError(
                f"evidence item {item.get('kind')!r} has no sha256; "
                "the file digest must be inside the hashed document"
            )
        # P4, enforced here as well as in the database.
        if item.get("kind") in {"snapshot", "clip"} and item.get("enhanced"):
            raise ValueError(
                f"evidence item {item.get('kind')!r} is marked enhanced; "
                "evidence must be original, unenhanced frames (P4)"
            )

    return {
        "schema": "drishti.evidence/v1",
        "alert_id": alert_id,
        "created_at": created_at,
        "site": dict(site),
        "camera": dict(camera),
        "detection": dict(detection),
        "risk": dict(risk),
        "items": [dict(i) for i in items],
        "provenance": {
            "spec_version": spec_version,
            "worker_version": worker_version,
            "config_version": config_version,
            "canonicalisation": "RFC8785",
            "hash_algorithm": "SHA-256",
        },
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Never just ``False``. Failing loudly with a diff is the entire point."""

    ok: bool
    expected_hash: str
    computed_hash: str
    canonical_length: int
    checks: tuple[tuple[str, bool, str], ...] = ()  # (name, passed, detail)
    diff: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "expected_hash": self.expected_hash,
            "computed_hash": self.computed_hash,
            "canonical_length": self.canonical_length,
            "checks": [
                {"name": n, "passed": p, "detail": d} for n, p, d in self.checks
            ],
            "diff": list(self.diff),
        }


def _flatten(doc: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(doc, Mapping):
        for k, v in doc.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(doc, (list, tuple)):
        for i, v in enumerate(doc):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = doc
    return out


def verify(
    doc: Mapping[str, Any],
    expected_hash: str,
    *,
    reference: Mapping[str, Any] | None = None,
) -> VerificationResult:
    """Recompute the hash and report every intermediate value.

    ``reference`` is the stored copy of the document. When supplied and the
    hashes disagree, we produce a field-level diff so an investigator sees
    *what* changed, not merely that something did.
    """
    stripped = strip_excluded(doc)
    canonical = canonicalise(stripped)
    computed = hashlib.sha256(canonical).hexdigest()
    ok = computed == expected_hash

    checks: list[tuple[str, bool, str]] = [
        (
            "schema",
            doc.get("schema") == "drishti.evidence/v1",
            f"schema={doc.get('schema')!r}",
        ),
        (
            "canonicalisation",
            doc.get("provenance", {}).get("canonicalisation") == "RFC8785",
            "document declares RFC8785 canonicalisation",
        ),
        ("evidence_hash", ok, f"expected={expected_hash} computed={computed}"),
    ]

    diff: list[str] = []
    if not ok and reference is not None:
        a = _flatten(strip_excluded(reference))
        b = _flatten(stripped)
        for key in sorted(set(a) | set(b)):
            if key not in a:
                diff.append(f"+ {key} = {b[key]!r}")
            elif key not in b:
                diff.append(f"- {key} = {a[key]!r}")
            elif a[key] != b[key]:
                diff.append(f"~ {key}: {a[key]!r} -> {b[key]!r}")

    return VerificationResult(
        ok=ok,
        expected_hash=expected_hash,
        computed_hash=computed,
        canonical_length=len(canonical),
        checks=tuple(checks),
        diff=tuple(diff),
    )
