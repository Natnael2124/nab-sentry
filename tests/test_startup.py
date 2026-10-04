"""Example tests for nab_sentry.startup and nab_sentry.logging_setup."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import pytest

from nab_sentry import startup
from nab_sentry.logging_setup import LOG_FILE_NAME, setup_logging


def _write_models(models: Path, files: dict[str, bytes]) -> None:
    entries = []
    for rel, content in files.items():
        p = models / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        entries.append({"role": rel.split("/")[-1], "path": rel, "sha256": hashlib.sha256(content).hexdigest()})
    (models / "manifest.json").write_text(json.dumps({"manifest_version": 1, "files": entries}), encoding="utf-8")


def test_offline_env_is_set_on_import():
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_DATASETS_OFFLINE"] == "1"
    assert Path(os.environ["HF_HOME"]).parts[-2:] == ("models", ".hf")
    assert Path(os.environ["TORCH_HOME"]).parts[-2:] == ("models", ".torch")


def test_verify_manifest_ok(models_dir: Path):
    _write_models(models_dir, {"yolo11n.onnx": b"onnx", "open_clip/x/model.bin": b"clip" * 1000})
    result = startup.verify_manifest(models_dir)
    assert result.ok
    assert set(result.files) == {"yolo11n.onnx", "open_clip/x/model.bin"}
    assert result.roles["model.bin"] == (models_dir / "open_clip/x/model.bin").resolve()


def test_verify_manifest_missing(models_dir: Path):
    result = startup.verify_manifest(models_dir)
    assert [(f.path, f.reason) for f in result.failures] == [("manifest.json", startup.MANIFEST_MISSING)]


def test_verify_manifest_unreadable(models_dir: Path):
    (models_dir / "manifest.json").write_text("{not json", encoding="utf-8")
    assert startup.verify_manifest(models_dir).failures[0].reason == startup.MANIFEST_UNREADABLE


def test_verify_manifest_reports_missing_and_mismatch(models_dir: Path):
    _write_models(models_dir, {"a.bin": b"a", "b.bin": b"b", "c.bin": b"c"})
    (models_dir / "a.bin").unlink()
    (models_dir / "b.bin").write_bytes(b"tampered")
    result = startup.verify_manifest(models_dir)
    assert {(f.path, f.reason) for f in result.failures} == {
        ("a.bin", startup.FILE_MISSING),
        ("b.bin", startup.SHA256_MISMATCH),
    }
    assert set(result.files) == {"c.bin"}


def test_verify_manifest_accepts_path_to_hash_mapping(models_dir: Path):
    (models_dir / "m.onnx").write_bytes(b"x")
    digest = hashlib.sha256(b"x").hexdigest()
    (models_dir / "manifest.json").write_text(json.dumps({"files": {"m.onnx": digest}}), encoding="utf-8")
    assert startup.verify_manifest(models_dir).ok


@pytest.mark.parametrize(
    "bad",
    ["../escape.bin", "a/../b.bin", "/abs.bin", "C:/abs.bin", "C:rel.bin", "a\\b.bin", "", "./a.bin"],
)
def test_verify_manifest_rejects_unsafe_paths(models_dir: Path, bad: str):
    (models_dir / "manifest.json").write_text(
        json.dumps({"files": [{"path": bad, "sha256": "0" * 64}]}), encoding="utf-8"
    )
    failures = startup.verify_manifest(models_dir).failures
    assert len(failures) == 1 and failures[0].reason == startup.INVALID_PATH


def test_require_models_exits_4_with_hint(models_dir: Path, capsys):
    _write_models(models_dir, {"yolo11n.onnx": b"onnx"})
    (models_dir / "yolo11n.onnx").unlink()
    with pytest.raises(SystemExit) as exc:
        startup.require_models(models_dir)
    assert exc.value.code == 4
    err = capsys.readouterr().err
    assert "yolo11n.onnx" in err and "file missing" in err and startup.FETCH_HINT in err


def test_require_models_returns_paths(models_dir: Path):
    _write_models(models_dir, {"yolo11n.onnx": b"onnx"})
    assert startup.require_models(models_dir) == {"yolo11n.onnx": (models_dir / "yolo11n.onnx").resolve()}


@pytest.mark.parametrize("host", ["0.0.0.0", "localhost", "::1", " 127.0.0.1", "127.0.0.2", "", None])
def test_ensure_loopback_rejects(host):
    with pytest.raises(startup.StartupError, match="loopback"):
        startup.ensure_loopback(host)


def test_ensure_loopback_accepts():
    startup.ensure_loopback("127.0.0.1")


def test_security_warning_and_log_file(tmp_path: Path, caplog):
    logs = tmp_path / "data" / "logs"
    logger = setup_logging(logs)
    setup_logging(logs)  # idempotent: no duplicate handlers
    assert sum(getattr(h, "_nab_sentry_handler", False) for h in logger.handlers) == 2
    with caplog.at_level(logging.WARNING, logger="nab_sentry"):
        startup.log_security_warning(logger)
    for h in logger.handlers:
        h.flush()
    rec = [r for r in caplog.records if r.levelno == logging.WARNING][-1]
    for phrase in ("authentication", "access control", "audit logging", "not tamper-evident"):
        assert phrase in rec.getMessage()
    assert "not tamper-evident" in (logs / LOG_FILE_NAME).read_text(encoding="utf-8")
    for h in list(logger.handlers):  # release the file on Windows
        if getattr(h, "_nab_sentry_handler", False):
            logger.removeHandler(h)
            h.close()
