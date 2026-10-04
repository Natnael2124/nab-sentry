"""Startup checks: offline environment, model manifest, loopback binding, security warning.

This module imports only the standard library so that ``nab_sentry/__init__.py`` can
call :func:`enable_offline_mode` before any model library (torch, open_clip,
huggingface_hub, onnxruntime) is imported (Requirement 13.3).

Manifest format (written by ``scripts/fetch_models.py``)::

    {
      "manifest_version": 1,
      "created_utc": "...",
      "files": [
        {"role": "detector", "path": "yolo11n.onnx", "sha256": "<64 hex>"},
        ...
      ]
    }

Paths are relative to ``models/`` with forward slashes. For robustness ``files`` may
also be a plain mapping ``{"relative/path": "<64 hex>"}``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_NAME = "manifest.json"
CHUNK_SIZE = 1024 * 1024  # 1 MiB
LOOPBACK_HOST = "127.0.0.1"
EXIT_MODELS = 4

FETCH_HINT = (
    r"venv\Scripts\python.exe scripts\fetch_models.py  "
    r"(run on a connected machine, then copy models\)"
)

SECURITY_WARNING = (
    "SECURITY: NAB Sentry MVP has no authentication, no access control, and no audit "
    "logging; the API listens only on 127.0.0.1. The Query_Log is a local file and is "
    "not tamper-evident."
)

# Failure reasons
MANIFEST_MISSING = "manifest missing"
MANIFEST_UNREADABLE = "manifest unreadable"
FILE_MISSING = "file missing"
SHA256_MISMATCH = "sha256 mismatch"
INVALID_PATH = "invalid path"

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


class StartupError(RuntimeError):
    """Raised when a startup precondition (such as loopback-only binding) is violated."""


# ------------------------------------------------------------------ offline mode


def enable_offline_mode(models_dir: Path | None = None) -> None:
    """Force Hugging Face / Torch libraries offline and keep their caches under ``models/``.

    Must run before any model library is imported (Requirement 13.3). Values are
    overwritten, not defaulted, so an inherited ``HF_HUB_OFFLINE=0`` cannot re-enable
    network access.
    """
    models = Path(models_dir) if models_dir is not None else _WORKSPACE_ROOT / "models"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(models / ".hf")
    os.environ["TORCH_HOME"] = str(models / ".torch")


# ------------------------------------------------------------------ manifest check


@dataclass(frozen=True)
class ManifestFailure:
    path: str  # manifest-relative path (or the manifest file name for manifest-level failures)
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


@dataclass
class ManifestResult:
    files: dict[str, Path] = field(default_factory=dict)  # relative path -> absolute path (verified only)
    roles: dict[str, Path] = field(default_factory=dict)  # role -> absolute path (verified only)
    failures: list[ManifestFailure] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def sha256_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    """Hex SHA-256 of a file, read in ``chunk_size`` chunks (1 MiB by default)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def is_safe_relative_path(rel: Any) -> bool:
    """True for a non-empty, forward-slash, relative path with no ``..``/``.``/empty segments."""
    if not isinstance(rel, str) or not rel or "\\" in rel or "\x00" in rel:
        return False
    if PurePosixPath(rel).is_absolute():
        return False
    win = PureWindowsPath(rel)
    if win.drive or win.root:
        return False
    parts = rel.split("/")
    return all(p not in ("", ".", "..") for p in parts)


def _entries(data: Any) -> list[tuple[Any, Any, Any]] | None:
    """Normalise manifest JSON into (role, path, sha256) triples, or None if malformed."""
    if not isinstance(data, dict):
        return None
    files = data.get("files")
    if isinstance(files, list):
        out = []
        for item in files:
            if not isinstance(item, dict):
                return None
            out.append((item.get("role"), item.get("path"), item.get("sha256")))
        return out
    if isinstance(files, dict):
        return [(None, p, s) for p, s in files.items()]
    return None


def verify_manifest(models_dir: Path) -> ManifestResult:
    """Check every manifest-listed file against its SHA-256 (Requirements 13.4, 13.5).

    Returns verified files and one :class:`ManifestFailure` per failing entry.
    """
    models_dir = Path(models_dir)
    manifest = models_dir / MANIFEST_NAME
    result = ManifestResult()
    if not manifest.is_file():
        result.failures.append(ManifestFailure(MANIFEST_NAME, MANIFEST_MISSING))
        return result
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        result.failures.append(ManifestFailure(MANIFEST_NAME, MANIFEST_UNREADABLE))
        return result
    entries = _entries(data)
    if not entries:
        result.failures.append(ManifestFailure(MANIFEST_NAME, MANIFEST_UNREADABLE))
        return result

    base = models_dir.resolve()
    for role, rel, digest in entries:
        label = rel if isinstance(rel, str) and rel else repr(rel)
        if not is_safe_relative_path(rel):
            result.failures.append(ManifestFailure(label, INVALID_PATH))
            continue
        if not isinstance(digest, str) or not _HEX64.match(digest):
            result.failures.append(ManifestFailure(label, MANIFEST_UNREADABLE))
            continue
        target = (base / rel).resolve()
        if not target.is_relative_to(base):  # e.g. a symlink escaping models/
            result.failures.append(ManifestFailure(label, INVALID_PATH))
            continue
        if not target.is_file():
            result.failures.append(ManifestFailure(label, FILE_MISSING))
            continue
        try:
            actual = sha256_file(target)
        except OSError:
            result.failures.append(ManifestFailure(label, FILE_MISSING))
            continue
        if actual.lower() != digest.lower():
            result.failures.append(ManifestFailure(label, SHA256_MISMATCH))
            continue
        result.files[rel] = target
        if isinstance(role, str) and role:
            result.roles[role] = target
    return result


def format_manifest_failures(failures: list[ManifestFailure], models_dir: Path) -> str:
    lines = [f"Model check failed in {Path(models_dir)}:"]
    lines += [f"  - {f}" for f in failures]
    lines.append(f"Run the Model_Fetcher: {FETCH_HINT}")
    return "\n".join(lines)


def require_models(models_dir: Path, logger: logging.Logger | None = None) -> dict[str, Path]:
    """Verify the manifest; on any failure print each failing file plus FETCH_HINT and exit 4."""
    result = verify_manifest(models_dir)
    if result.failures:
        message = format_manifest_failures(result.failures, models_dir)
        if logger is not None:
            logger.error(message)
        else:
            print(message, file=sys.stderr)
        raise SystemExit(EXIT_MODELS)
    return result.files


# ------------------------------------------------------------------ binding / security


def ensure_loopback(host: Any) -> None:
    """Raise StartupError unless ``host`` is exactly ``"127.0.0.1"`` (Requirement 18.5)."""
    if not isinstance(host, str) or host != LOOPBACK_HOST:
        raise StartupError(
            f"Refusing to bind to {host!r}: the MVP permits only loopback binding "
            f"({LOOPBACK_HOST})."
        )


def log_security_warning(logger: logging.Logger) -> None:
    """Log the MVP security posture at WARNING level (Requirement 18.2)."""
    logger.warning(SECURITY_WARNING)
