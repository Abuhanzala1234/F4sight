"""Structured logging (BUILD_SPEC §11, CLAUDE.md/Code).

    No bare `except:`. No silent failures. Every caught exception logs with
    context.

Two formats: ``console`` (human, coloured, for `make worker`) and ``json`` (one
object per line, for shipping). Both always carry camera id and site where the
caller has them.

Credential redaction is applied at the formatter, not at each call site. A
redaction you have to remember is a redaction that eventually leaks an RTSP
password into a log file someone emails to a vendor.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any, ClassVar

__all__ = ["RedactingFilter", "configure_logging"]

# rtsp://user:pass@host  and  postgres://user:pass@host
_CREDENTIAL_RE = re.compile(r"(?P<scheme>[a-z][a-z0-9+.-]*://)[^/@\s]+:[^/@\s]+@")
_SECRET_KV_RE = re.compile(r"(?i)\b(password|secret|token|api[_-]?key|hmac[_-]?key)\b\s*[=:]\s*\S+")


class RedactingFilter(logging.Filter):
    """Strips credentials from every record before it is formatted."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        redacted = _CREDENTIAL_RE.sub(r"\g<scheme>***:***@", message)
        redacted = _SECRET_KV_RE.sub(lambda m: f"{m.group(1)}=***", redacted)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "thread": record.threadName,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key.startswith("drishti_"):
                payload[key[8:]] = value
        return json.dumps(payload, separators=(",", ":"))


class ConsoleFormatter(logging.Formatter):
    COLOURS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[38;5;244m",
        "INFO": "\033[38;5;39m",
        "WARNING": "\033[38;5;214m",
        "ERROR": "\033[38;5;203m",
        "CRITICAL": "\033[48;5;203m\033[38;5;231m",
    }
    RESET = "\033[0m"

    def __init__(self, colour: bool = True) -> None:
        super().__init__()
        self.colour = colour and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, UTC).strftime("%H:%M:%S")
        level = record.levelname[:4]
        if self.colour:
            level = f"{self.COLOURS.get(record.levelname, '')}{level}{self.RESET}"
        name = record.name.replace("drishti_worker.", "")
        line = f"{ts} {level} {name:<14} {record.getMessage()}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Install handlers. Idempotent — safe to call from tests."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())
    handler.addFilter(RedactingFilter())

    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # These libraries are chatty at DEBUG and tell us nothing we want.
    for noisy in ("urllib3", "botocore", "minio", "asyncio", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
