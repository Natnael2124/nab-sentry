"""Fast tests for scripts/search_cli.py: table output and exit codes (no real models)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from nab_sentry.config import Config
from nab_sentry.ingest.pipeline import IngestPipeline
from nab_sentry.logging_setup import LOGGER_NAME
from nab_sentry.store.db import MetadataStore, as_aware
from nab_sentry.store.vector_index import VectorIndex
from tests.fakes import FakeEncoder, FakePlayback, FakeSource

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "search_cli.py"


def _load():
    spec = importlib.util.spec_from_file_location("search_cli_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cli = _load()

W, H, FPS, N = 64, 48, 4.0, 40  # 10 s: red square for 0-5 s, blue square for 5-10 s
VIDEO = "CAM-A_20250101T080000.mp4"


def _frames() -> list[np.ndarray]:
    out = []
    for i in range(N):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        x = (i // 2 * 3) % (W - 12)  # moves every other frame so the Motion_Gate passes
        colour = (0, 0, 255) if i < N // 2 else (255, 0, 0)  # BGR red, then blue
        img[10:22, x:x + 12] = colour
        out.append(img)
    return out


@pytest.fixture(autouse=True)
def _close_log_handlers():
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
    """Workspace with a stub models manifest and one ingested synthetic video."""
    models = tmp_path / "models"
    models.mkdir()
    stub = models / "stub.bin"
    stub.write_bytes(b"stub weights")
    digest = hashlib.sha256(stub.read_bytes()).hexdigest()
    (models / "manifest.json").write_text(
        json.dumps({"manifest_version": 1,
                    "files": [{"role": "embedder", "path": "stub.bin", "sha256": digest}]}),
        encoding="utf-8")
    videos = tmp_path / "data" / "videos"
    videos.mkdir(parents=True)
    (videos / VIDEO).write_bytes(b"video")

    cfg = Config(root=tmp_path, enable_detector=False, sample_rate=2.0, keyframe_interval_s=2.0)
    db = MetadataStore(cfg.db_path)
    try:
        pipe = IngestPipeline(
            cfg, db, VectorIndex(), FakeEncoder(colour_mode=True), None,
            open_source=lambda p: FakeSource(_frames(), fps=FPS, width=W, height=H, path=p),
            start_playback=FakePlayback(),
        )
        report = pipe.ingest_paths([videos])
        assert [r.status for r in report.results] == ["ingested"]
    finally:
        db.close()
    return tmp_path


def _run(ws: Path, *argv: str) -> int:
    return cli.main([*argv, "--set", f"root={ws}"],
                    encoder_factory=lambda cfg, files: FakeEncoder(colour_mode=True))


def test_prints_ranked_table_with_absolute_times(ws: Path, capsys) -> None:
    assert _run(ws, "red square", "--limit", "5") == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].split() == list(cli.HEADERS)
    rows = [ln.split() for ln in lines[2:]]
    assert rows
    camera, label, start, end, offsets, score, rel, thumb = rows[0]
    assert camera == "CAM-A"
    lo, hi = (float(x) for x in offsets.rstrip("s").split("-"))
    assert lo < 5.0 and hi <= 10.0  # top Event covers the red half
    video_start = as_aware(datetime(2025, 1, 1, 8, 0, 0))
    assert datetime.fromisoformat(start) == video_start + timedelta(seconds=lo)
    assert datetime.fromisoformat(end) == video_start + timedelta(seconds=hi)
    assert float(score) > 0.9
    assert thumb.endswith(".jpg")
    # Ranked by descending score (9.12).
    scores = [float(r[5]) for r in rows]
    assert scores == sorted(scores, reverse=True)


def test_unknown_camera_prints_no_results(ws: Path, capsys) -> None:
    assert _run(ws, "red square", "--camera", "CAM-Z") == 0
    assert capsys.readouterr().out.strip() == "no results"


@pytest.mark.parametrize("extra", [
    ["--start", "2025-01-02T00:00:00", "--end", "2025-01-01T00:00:00"],  # start > end
    ["--cls", "unicorn"],
    ["--start", "not-a-date"],
    ["--limit", "0"],
])
def test_invalid_filter_exits_1(ws: Path, capsys, extra: list[str]) -> None:
    assert _run(ws, "red square", *extra) == 1
    assert "error:" in capsys.readouterr().err


def test_empty_query_exits_1(ws: Path) -> None:
    assert _run(ws, "   ") == 1


def test_inconsistent_index_exits_5(ws: Path, capsys) -> None:
    index_path = ws / "data" / "vectors.faiss"
    index = VectorIndex.load(index_path)
    index.add(np.array([999_999]), np.eye(1, 512, dtype=np.float32))
    index.save(index_path)

    assert _run(ws, "red square") == 5
    assert "inconsistent" in capsys.readouterr().err


def test_missing_manifest_exits_4(ws: Path) -> None:
    (ws / "models" / "manifest.json").unlink()
    assert _run(ws, "red square") == 4


def test_invalid_config_exits_2(ws: Path) -> None:
    assert _run(ws, "red square", "--set", "no_such_param=1") == 2
