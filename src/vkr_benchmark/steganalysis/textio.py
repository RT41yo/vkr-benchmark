"""Lossless UTF-8 storage for generated carrier texts.

Path.read_text() uses universal-newline translation by default. Generated
text may contain CR or CRLF *as model output*, so normalizing these bytes
would invalidate recorded hashes and silently alter detector inputs.

Use byte-preserving reads/writes for plain carrier texts. JSON/JSONL may
continue to use ordinary text I/O (embedded CR is JSON-escaped).
"""
from __future__ import annotations

from pathlib import Path

from vkr_benchmark.errors import ConfigurationError


def read_carrier_text(path: Path) -> str:
    """Read a carrier without normalizing any newlines; require valid UTF-8."""
    try:
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError(f"cannot read exact UTF-8 carrier {path}: {exc}") from exc


def write_carrier_text(path: Path, text: str) -> None:
    """Write exactly the UTF-8 bytes used by SHA-256 text manifests."""
    path.write_bytes(text.encode("utf-8"))
