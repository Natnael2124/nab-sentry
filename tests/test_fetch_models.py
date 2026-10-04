"""Unit tests for the Model_Fetcher (scripts/fetch_models.py).

Validates: Requirements 13.2, 13.8
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from nab_sentry.startup import MANIFEST_NAME, verify_manifest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_models.py"
_MODULE_NAME = "nab_sentry_fetch_models_under_test"


def _load_fetcher():
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


fm = _load_fetcher()

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


# ------------------------------------------------------------------ stubs


def ok_downloader(rel: str, models_dir: Path) -> Path:
    target = Path(models_dir) / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(f"fake weights for {rel}".encode())
    return target


def ok_exporter(pt: Path, onnx: Path) -> Path:
    Path(onnx).write_bytes(b"fake onnx graph")
    return Path(onnx)


class Recorder:
    def __init__(self):
        self.smoke_calls: list[tuple[str, Path]] = []
        self.sleeps: list[float] = []

    def smoke_loader(self, role: str, path: Path) -> None:
        self.smoke_calls.append((role, Path(path)))

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def _write_stale_manifest(models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    stale = models_dir / MANIFEST_NAME
    stale.write_text(json.dumps({"manifest_version": 1, "files": []}), encoding="utf-8")
    (models_dir / (MANIFEST_NAME + ".tmp")).write_text("{}", encoding="utf-8")
    return stale


def _assert_no_manifest(models_dir: Path) -> None:
    assert not (models_dir / MANIFEST_NAME).exists()
    assert not (models_dir / (MANIFEST_NAME + ".tmp")).exists()


# ------------------------------------------------------------------ failure paths


def test_downloader_failing_three_times_leaves_no_manifest_and_names_file(tmp_path, capsys):
    models_dir = tmp_path / "models"
    _write_stale_manifest(models_dir)
    rec = Recorder()
    attempts: list[str] = []

    def failing_downloader(rel: str, md: Path) -> Path:
        attempts.append(rel)
        if rel == fm.YOLO_PT_REL:
            raise ConnectionError("network unreachable")
        return ok_downloader(rel, md)

    rc = fm.main(
        [],
        downloader=failing_downloader,
        exporter=ok_exporter,
        smoke_loader=rec.smoke_loader,
        sleep=rec.sleep,
        models_dir=models_dir,
    )

    assert rc == 1
    assert attempts.count(fm.YOLO_PT_REL) == fm.MAX_ATTEMPTS == 3
    assert attempts.count(fm.CLIP_REL) == 1
    # Exponential backoff between attempts, none after the last one.
    assert len(rec.sleeps) == fm.MAX_ATTEMPTS - 1
    assert rec.sleeps[1] > rec.sleeps[0] > 0
    _assert_no_manifest(models_dir)
    err = capsys.readouterr().err
    assert f"Model_Fetcher FAILED: {fm.YOLO_PT_REL}:" in err
    assert rec.smoke_calls == []


def test_downloader_recovers_on_third_attempt(tmp_path):
    models_dir = tmp_path / "models"
    rec = Recorder()
    calls = {"n": 0}

    def flaky(rel: str, md: Path) -> Path:
        if rel == fm.CLIP_REL:
            calls["n"] += 1
            if calls["n"] < fm.MAX_ATTEMPTS:
                raise TimeoutError("slow mirror")
        return ok_downloader(rel, md)

    rc = fm.main(
        [], downloader=flaky, exporter=ok_exporter, smoke_loader=rec.smoke_loader,
        sleep=rec.sleep, models_dir=models_dir,
    )
    assert rc == 0
    assert calls["n"] == fm.MAX_ATTEMPTS
    assert verify_manifest(models_dir).ok


def test_empty_download_counts_as_failure(tmp_path, capsys):
    models_dir = tmp_path / "models"
    rec = Recorder()

    def empty_downloader(rel: str, md: Path) -> Path:
        target = Path(md) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
        return target

    rc = fm.main(
        [], downloader=empty_downloader, exporter=ok_exporter, smoke_loader=rec.smoke_loader,
        sleep=rec.sleep, models_dir=models_dir,
    )
    assert rc == 1
    _assert_no_manifest(models_dir)
    assert f"Model_Fetcher FAILED: {fm.CLIP_REL}:" in capsys.readouterr().err


def test_export_failure_leaves_no_manifest_and_names_onnx(tmp_path, capsys):
    models_dir = tmp_path / "models"
    _write_stale_manifest(models_dir)
    rec = Recorder()

    def failing_exporter(pt: Path, onnx: Path) -> Path:
        raise RuntimeError("onnx export crashed")

    rc = fm.main(
        [], downloader=ok_downloader, exporter=failing_exporter, smoke_loader=rec.smoke_loader,
        sleep=rec.sleep, models_dir=models_dir,
    )
    assert rc == 1
    _assert_no_manifest(models_dir)
    err = capsys.readouterr().err
    assert f"Model_Fetcher FAILED: {fm.YOLO_ONNX_REL}:" in err
    assert "onnx export crashed" in err


def test_smoke_load_failure_leaves_no_manifest_and_names_file(tmp_path, capsys):
    models_dir = tmp_path / "models"
    _write_stale_manifest(models_dir)

    def failing_smoke(role: str, path: Path) -> None:
        if role == fm.ROLE_DETECTOR:
            raise RuntimeError("bad input shape")

    rc = fm.main(
        [], downloader=ok_downloader, exporter=ok_exporter, smoke_loader=failing_smoke,
        sleep=lambda s: None, models_dir=models_dir,
    )
    assert rc == 1
    _assert_no_manifest(models_dir)
    assert f"Model_Fetcher FAILED: {fm.YOLO_ONNX_REL}:" in capsys.readouterr().err


# ------------------------------------------------------------------ success path


def test_successful_run_writes_verifiable_manifest(tmp_path, capsys):
    models_dir = tmp_path / "models"
    _write_stale_manifest(models_dir)
    rec = Recorder()

    rc = fm.main(
        [], downloader=ok_downloader, exporter=ok_exporter, smoke_loader=rec.smoke_loader,
        sleep=rec.sleep, models_dir=models_dir,
    )

    assert rc == 0
    assert rec.sleeps == []
    assert not (models_dir / (MANIFEST_NAME + ".tmp")).exists()
    data = json.loads((models_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert data["manifest_version"] == 1
    entries = {f["path"]: f for f in data["files"]}
    assert set(entries) == {fm.CLIP_REL, fm.YOLO_ONNX_REL}
    assert entries[fm.CLIP_REL]["role"] == fm.ROLE_EMBEDDER
    assert entries[fm.YOLO_ONNX_REL]["role"] == fm.ROLE_DETECTOR
    for path, entry in entries.items():
        assert _HEX64.match(entry["sha256"]), entry
        assert "\\" not in path and not Path(path).is_absolute()

    # Smoke loader received absolute local paths for both runtime files.
    roles = {role: p for role, p in rec.smoke_calls}
    assert set(roles) == {fm.ROLE_EMBEDDER, fm.ROLE_DETECTOR}
    assert all(p.is_absolute() for p in roles.values())

    result = verify_manifest(models_dir)
    assert result.ok, result.failures
    assert set(result.roles) == {fm.ROLE_EMBEDDER, fm.ROLE_DETECTOR}
    assert "Model_Fetcher OK" in capsys.readouterr().out


def test_models_dir_cli_argument(tmp_path):
    models_dir = tmp_path / "custom_models"
    rc = fm.main(
        ["--models-dir", str(models_dir)], downloader=ok_downloader, exporter=ok_exporter,
        smoke_loader=lambda r, p: None, sleep=lambda s: None,
    )
    assert rc == 0
    assert verify_manifest(models_dir).ok


@pytest.mark.parametrize("rel", ["CLIP_REL", "YOLO_PT_REL", "YOLO_ONNX_REL"])
def test_relative_path_constants_use_forward_slashes(rel):
    value = getattr(fm, rel)
    assert "\\" not in value and not value.startswith("/")
