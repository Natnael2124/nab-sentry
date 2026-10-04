"""Quick sanity checks for scripts/evaluate.py with a FakeEncoder (fuller tests: task 17.5)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

from nab_sentry.config import Config
from nab_sentry.evaluation import load_queries_file
from nab_sentry.ingest.pipeline import IngestPipeline
from nab_sentry.logging_setup import LOGGER_NAME
from nab_sentry.store.db import MetadataStore
from nab_sentry.store.vector_index import VectorIndex
from tests.fakes import FakeEncoder, FakePlayback, FakeSource

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate.py"


def _load():
    spec = importlib.util.spec_from_file_location("evaluate_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ev = _load()

W, H, FPS, N = 64, 48, 4.0, 40  # 10 s: red square 0-5 s, blue square 5-10 s
VIDEO = "CAM-A_20250101T080000.mp4"


def _frames() -> list[np.ndarray]:
    out = []
    for i in range(N):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        x = (i // 2 * 3) % (W - 12)
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
        assert [r.status for r in pipe.ingest_paths([videos]).results] == ["ingested"]
    finally:
        db.close()
    return tmp_path


def _entry(text: str, cam: str, start: str, end: str) -> str:
    return f'  - text: "{text}"\n    camera_id: {cam}\n    start: "{start}"\n    end: "{end}"\n'


def _queries(ws: Path, n_unknown: int = 13, known: bool = True) -> Path:
    body = "queries:\n"
    if known:
        body += _entry("red square", "CAM-A", "2025-01-01T08:00:00", "2025-01-01T08:00:04")
        body += _entry("blue square", "CAM-A", "2025-01-01T08:00:06", "2025-01-01T08:00:10")
    for i in range(n_unknown):
        body += _entry(f"placeholder {i}", "CAM-NOPE", "2025-01-01T09:00:00",
                       "2025-01-01T09:01:00")
    p = ws / "queries.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def _run(ws: Path, *argv: str) -> int:
    return ev.main([*argv, "--set", f"root={ws}"],
                   encoder_factory=lambda cfg, files: FakeEncoder(colour_mode=True))


def _report(ws: Path) -> dict:
    reports = sorted((ws / "data" / "eval").glob("report-*.json"))
    assert len(reports) >= 1
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def test_template_queries_file_is_valid() -> None:
    queries, errors = load_queries_file(ROOT / "eval" / "queries.yaml")
    assert len(queries) == 15 and errors == []
    assert [q.text for q in queries[:2]] == ["red square", "blue circle"]
    assert {q.camera_id for q in queries[:2]} == {"CAM-SYN01"}


def test_run_writes_report_with_metrics_and_errors(ws: Path) -> None:
    assert _run(ws, "--queries", str(_queries(ws))) == 0
    rep = _report(ws)
    assert set(rep["config"]) == set(ev.REPORT_PARAMS)
    assert rep["config"]["top_k"] == 300
    rows = rep["queries"]
    assert len(rows) == 15
    assert rows[0]["text"] == "red square" and rows[0]["hit_at_1"] == 1
    assert rows[1]["hit_at_1"] == 1
    assert 0.2 <= rows[0]["precision_at_5"] <= 1.0
    assert all("not in the Metadata_Store" in r["error"] for r in rows[2:])
    agg = rep["aggregate"]
    assert agg["queries_run"] == 15 and agg["queries_errored"] == 13
    assert agg["hit_at_1"] == 1.0


def test_two_runs_give_identical_metrics(ws: Path) -> None:
    q = str(_queries(ws))
    assert _run(ws, "--queries", q) == 0
    assert _run(ws, "--queries", q) == 0
    reports = sorted((ws / "data" / "eval").glob("report-*.json"))
    assert len(reports) == 2
    a, b = (json.loads(p.read_text(encoding="utf-8")) for p in reports)

    def metrics(r: dict) -> tuple:
        return ([(x.get("precision_at_5"), x.get("hit_at_1")) for x in r["queries"]],
                r["aggregate"]["precision_at_5"], r["aggregate"]["hit_at_1"])

    assert metrics(a) == metrics(b)


def test_all_errors_exits_nonzero(ws: Path, capsys) -> None:
    assert _run(ws, "--queries", str(_queries(ws, n_unknown=15, known=False))) == 1
    assert "every query" in capsys.readouterr().err
    assert _report(ws)["aggregate"] is None


@pytest.mark.parametrize("case", ["missing", "unparseable", "too_few"])
def test_file_errors_exit_1(ws: Path, capsys, case: str) -> None:
    p = ws / "queries.yaml"
    if case == "unparseable":
        p.write_text("queries: [\n", encoding="utf-8")
    elif case == "too_few":
        p = _queries(ws, n_unknown=3)
    assert _run(ws, "--queries", str(p)) == 1
    assert "error:" in capsys.readouterr().err
    assert not (ws / "data" / "eval").exists()
