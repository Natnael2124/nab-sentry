"""Fast tests for scripts/ingest.py: exit codes and the summary line (no real models)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

from nab_sentry.ingest.detector import Detection
from nab_sentry.ingest.sources import UnreadableVideo
from nab_sentry.logging_setup import LOGGER_NAME
from nab_sentry.startup import FETCH_HINT
from nab_sentry.store.vector_index import VectorIndex
from tests.fakes import FakeDetector, FakeEncoder, FakePlayback, FakeSource

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ingest.py"


def _load():
    spec = importlib.util.spec_from_file_location("ingest_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ingest = _load()

W, H, FPS, N = 64, 48, 4.0, 24


def _frames(seed: int) -> list[np.ndarray]:
    out = []
    for i in range(N):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        x = (seed * 7 + (i // 4) * 9) % (W - 12)
        img[10:22, x:x + 12] = 255
        out.append(img)
    return out


@pytest.fixture(autouse=True)
def _close_log_handlers():
    """Undo setup_logging (handlers + level) so other tests' caplog counts are unaffected."""
    logger = logging.getLogger(LOGGER_NAME)
    level = logger.level
    yield
    logger.setLevel(level)
    for h in list(logger.handlers):
        if getattr(h, "_nab_sentry_handler", False):
            logger.removeHandler(h)
            h.close()


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    """Workspace root with a stub models manifest and an empty data/videos/."""
    models = tmp_path / "models"
    models.mkdir()
    stub = models / "stub.bin"
    stub.write_bytes(b"stub weights")
    digest = hashlib.sha256(stub.read_bytes()).hexdigest()
    (models / "manifest.json").write_text(
        json.dumps({"manifest_version": 1,
                    "files": [{"role": "embedder", "path": "stub.bin", "sha256": digest}]}),
        encoding="utf-8")
    (tmp_path / "data" / "videos").mkdir(parents=True)
    return tmp_path


def _add_video(ws: Path, name: str, seed: int) -> Path:
    p = ws / "data" / "videos" / name
    p.write_bytes(f"video-{seed}".encode())
    return p


def _open_source(path: Path) -> FakeSource:
    seed = int(Path(path).read_bytes().decode().split("-")[1])
    return FakeSource(_frames(seed), fps=FPS, width=W, height=H, path=path)


def _run(ws: Path, *extra: str, open_source=_open_source, **kw) -> int:
    # Defaults first so a test's own --set (later, so it wins) can override them.
    argv = ["--set", f"root={ws}", "--set", "enable_detector=false",
            "--set", "sample_rate=2", "--set", "keyframe_interval_s=2", *extra]
    kw.setdefault("encoder_factory", lambda cfg, files: FakeEncoder(batch_size=cfg.batch_size))
    return ingest.main(argv, open_source=open_source, start_playback=FakePlayback(), **kw)


def test_no_video_files_exits_1_without_writing(ws: Path, capsys) -> None:
    assert _run(ws) == 1
    assert "no video files found" in capsys.readouterr().err
    assert not (ws / "data" / "nab_sentry.db").exists()


def test_invalid_config_exits_2(ws: Path, capsys) -> None:
    _add_video(ws, "CAM-A_20250101T080000.mp4", 1)
    assert _run(ws, "--set", "sample_rate=99") == 2
    assert "sample_rate" in capsys.readouterr().err
    assert _run(ws, "--set", "no_such_param=1") == 2
    assert not (ws / "data" / "nab_sentry.db").exists()


def test_missing_manifest_exits_4(ws: Path) -> None:
    (ws / "models" / "manifest.json").unlink()
    _add_video(ws, "CAM-A_20250101T080000.mp4", 1)
    assert _run(ws) == 4


def test_successful_run_prints_summary_and_reingest_skips(ws: Path, capsys) -> None:
    _add_video(ws, "CAM-A_20250101T080000.mp4", 1)
    _add_video(ws, "CAM-B_20250101T090000.mp4", 2)
    _add_video(ws, "not-a-camera-name.mp4", 3)  # unresolved metadata -> failed

    assert _run(ws) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    fields = dict(kv.split("=") for kv in line.split())
    assert (fields["ingested"], fields["failed"], fields["already_indexed"]) == ("2", "1", "0")
    vectors = int(fields["vectors"])
    assert vectors > 0
    assert VectorIndex.load(ws / "data" / "vectors.faiss").ntotal == vectors

    assert _run(ws) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line == f"ingested=0 failed=1 already_indexed=2 vectors={vectors}"


def test_zero_vectors_afterwards_exits_1(ws: Path, capsys) -> None:
    _add_video(ws, "CAM-A_20250101T080000.mp4", 1)

    def unreadable(path: Path):
        raise UnreadableVideo(f"cannot open {path}")

    assert _run(ws, open_source=unreadable) == 1
    out = capsys.readouterr()
    assert "ingested=0 failed=1 already_indexed=0 vectors=0" in out.out
    assert "zero vectors" in out.err


def test_enabled_detector_with_missing_onnx_exits_6(ws: Path, capsys) -> None:
    # The stub manifest has no yolo11n.onnx, so the default factory raises ModelMissingError
    # before any video is opened or the store is created (Requirement 4.6).
    _add_video(ws, "CAM-A_20250101T080000.mp4", 1)
    opened: list[Path] = []

    def spy_source(path: Path) -> FakeSource:
        opened.append(path)
        return _open_source(path)

    assert _run(ws, "--set", "enable_detector=true", open_source=spy_source) == 6
    err = capsys.readouterr().err
    assert str(ws / "models" / "yolo11n.onnx") in err
    assert FETCH_HINT in err
    assert opened == []
    assert not (ws / "data" / "nab_sentry.db").exists()


def test_enabled_detector_writes_crop_vectors(ws: Path, capsys) -> None:
    _add_video(ws, "CAM-A_20250101T080000.mp4", 1)
    det = FakeDetector(lambda frame, i: [
        Detection("person", 0.9, (2, 3, 30, 40)),
        Detection("car", 0.5, (10, 5, 60, 45)),
    ])
    made: list[tuple[float, float, int]] = []

    def factory(cfg, files):
        made.append((cfg.det_conf, cfg.det_iou, cfg.det_max_per_frame))
        return det

    assert _run(ws, "--set", "enable_detector=true", detector_factory=factory) == 0
    assert made == [(0.35, 0.45, 10)]
    assert det.call_count > 0

    con = sqlite3.connect(ws / "data" / "nab_sentry.db")
    try:
        n_frames = con.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        crops = con.execute(
            "SELECT det_class, det_conf, x1, y1, x2, y2 FROM vectors WHERE kind='crop'"
        ).fetchall()
        n_frame_vecs = con.execute(
            "SELECT COUNT(*) FROM vectors WHERE kind='frame'").fetchone()[0]
    finally:
        con.close()
    assert n_frames == det.call_count == n_frame_vecs
    assert len(crops) == 2 * n_frames
    assert sorted(set(crops)) == [("car", 0.5, 10, 5, 60, 45), ("person", 0.9, 2, 3, 30, 40)]
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.endswith(f"vectors={3 * n_frames}")


def test_unloadable_index_exits_5_and_repair_fixes_it(ws: Path, capsys) -> None:
    _add_video(ws, "CAM-A_20250101T080000.mp4", 1)
    assert _run(ws) == 0
    index_path = ws / "data" / "vectors.faiss"
    index_path.write_bytes(b"corrupt")

    assert _run(ws) == 5
    assert "--repair" in capsys.readouterr().err

    assert _run(ws, "--repair") == 0
    assert "deleted_videos=1" in capsys.readouterr().out
    assert VectorIndex.load(index_path).ntotal == 0

    assert _run(ws) == 0  # the video whose vectors were lost is ingested again
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith(
        "ingested=1 failed=0 already_indexed=0")
