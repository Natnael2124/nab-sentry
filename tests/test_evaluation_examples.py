"""Example-based unit tests for the Evaluator, ``scripts/evaluate.py`` (task 17.5).

Covers Requirement 16.1 (15..20 query bounds), 16.5/16.6 (report contents and location),
16.7 (per-query error rows excluded from aggregates), 16.8 (file-level failures exit non-zero
with a cause and no aggregate), and 16.9 (two runs give identical metrics).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from nab_sentry.config import Config
from nab_sentry.evaluation import QueryResult, aggregate
from nab_sentry.ingest.pipeline import IngestPipeline
from nab_sentry.logging_setup import LOGGER_NAME
from nab_sentry.store.db import MetadataStore
from nab_sentry.store.vector_index import VectorIndex
from tests.fakes import FakeEncoder, FakePlayback, FakeSource

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("evaluate_script_examples", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ev = _load_script()

# Requirement 16.5, in order: Sample_Rate, Motion_Threshold, Keyframe_Interval, gate width,
# detection threshold, per-frame detection maximum, batch size, Top_K, Merge_Gap,
# Event_Padding, Label_Boost -> Config field names.
REQ_16_5_PARAMS = [
    "sample_rate", "motion_threshold", "keyframe_interval_s", "gate_width", "det_conf",
    "det_max_per_frame", "batch_size", "top_k", "merge_gap_s", "event_padding_s", "label_boost",
]
REPORT_NAME = re.compile(r"report-\d{8}-\d{6}(-\d+)?\.json")

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
    """Workspace with a stub model manifest and one fake-ingested CAM-A video."""
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


def _known() -> list[str]:
    """Three scoreable queries: two that hit, one on CAM-A with a window holding no Events."""
    return [
        _entry("red square", "CAM-A", "2025-01-01T08:00:00", "2025-01-01T08:00:04"),
        _entry("blue square", "CAM-A", "2025-01-01T08:00:06", "2025-01-01T08:00:10"),
        _entry("red square", "CAM-A", "2025-01-01T12:00:00", "2025-01-01T12:00:10"),
    ]


def _unknown_cam(i: int) -> str:
    return _entry(f"placeholder {i}", "CAM-NOPE", "2025-01-01T09:00:00", "2025-01-01T09:01:00")


def _write(ws: Path, entries: list[str]) -> Path:
    p = ws / "queries.yaml"
    p.write_text("queries:\n" + "".join(entries), encoding="utf-8")
    return p


def _queries_of(ws: Path, total: int) -> Path:
    """``total`` entries: the three known queries, one end<=start error, the rest unknown camera."""
    bad_window = _entry("bad window", "CAM-A", "2025-01-01T08:00:05", "2025-01-01T08:00:05")
    entries = _known() + [bad_window]
    entries += [_unknown_cam(i) for i in range(total - len(entries))]
    return _write(ws, entries)


def _run(ws: Path, *argv: str) -> int:
    return ev.main([*argv, "--set", f"root={ws}"],
                   encoder_factory=lambda cfg, files: FakeEncoder(colour_mode=True))


def _reports(ws: Path) -> list[Path]:
    d = ws / "data" / "eval"
    return sorted(d.glob("report-*.json")) if d.exists() else []


def _report(ws: Path) -> dict:
    reports = _reports(ws)
    assert len(reports) == 1
    return json.loads(reports[0].read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------
# 16.1 / 16.8: query count bounds


@pytest.mark.parametrize("count", [14, 21])
def test_count_outside_bounds_exits_1_with_cause_and_no_report(ws: Path, capsys,
                                                               count: int) -> None:
    assert _run(ws, "--queries", str(_queries_of(ws, count))) == 1
    err = capsys.readouterr().err
    assert f"holds {count} queries" in err and "15 to 20" in err
    assert _reports(ws) == []


@pytest.mark.parametrize("count", [15, 20])
def test_count_at_bounds_runs(ws: Path, count: int) -> None:
    assert _run(ws, "--queries", str(_queries_of(ws, count))) == 0
    rep = _report(ws)
    assert len(rep["queries"]) == count
    assert rep["aggregate"]["queries_run"] == count


# --------------------------------------------------------------------------------------------
# 16.8: missing / unparseable / malformed files


@pytest.mark.parametrize("content, cause", [
    ("queries: [\n  - text: \"x\"\n", "cannot parse"),
    ("- text: a\n- text: b\n", "must be a mapping"),
    ("just a string\n", "must be a mapping"),
])
def test_unparseable_or_non_mapping_exits_1(ws: Path, capsys, content: str, cause: str) -> None:
    p = ws / "queries.yaml"
    p.write_text(content, encoding="utf-8")
    assert _run(ws, "--queries", str(p)) == 1
    assert cause in capsys.readouterr().err
    assert _reports(ws) == []


def test_missing_file_exits_1(ws: Path, capsys) -> None:
    assert _run(ws, "--queries", str(ws / "nope.yaml")) == 1
    assert "queries file not found" in capsys.readouterr().err
    assert _reports(ws) == []


def test_every_query_in_error_exits_1_with_null_aggregate(ws: Path, capsys) -> None:
    assert _run(ws, "--queries", str(_write(ws, [_unknown_cam(i) for i in range(15)]))) == 1
    assert "every query" in capsys.readouterr().err
    rep = _report(ws)
    assert rep["aggregate"] is None
    assert all("error" in r for r in rep["queries"])


# --------------------------------------------------------------------------------------------
# 16.3 / 16.7: error rows recorded and excluded from aggregates


def test_aggregate_excludes_error_rows(ws: Path) -> None:
    assert _run(ws, "--queries", str(_queries_of(ws, 15))) == 0
    rep = _report(ws)
    rows = rep["queries"]
    ok = [r for r in rows if "error" not in r]
    errs = [r for r in rows if "error" in r]
    assert len(ok) == 3 and len(errs) == 12
    assert "not later than start" in rows[3]["error"]
    assert all("not in the Metadata_Store" in r["error"] for r in rows[4:])
    assert all("precision_at_5" not in r and "hit_at_1" not in r for r in errs)

    agg = rep["aggregate"]
    assert agg["queries_run"] == 15 and agg["queries_errored"] == 12
    assert agg["precision_at_5"] == pytest.approx(sum(r["precision_at_5"] for r in ok) / 3)
    assert agg["hit_at_1"] == pytest.approx(sum(r["hit_at_1"] for r in ok) / 3)
    assert agg["latency_ms_mean"] == pytest.approx(sum(r["latency_ms"] for r in ok) / 3)
    # The out-of-window query scores 0, so the mean differs from a mean over all 15 rows.
    assert [r["hit_at_1"] for r in ok] == [1, 1, 0]
    assert agg["hit_at_1"] == pytest.approx(2 / 3)


def test_aggregate_function_ignores_error_results() -> None:
    results = [
        QueryResult("a", precision_at_5=0.4, hit_at_1=1, latency_ms=10.0),
        QueryResult("b", error="missing query text"),
        QueryResult("c", precision_at_5=0.0, hit_at_1=0, latency_ms=30.0),
    ]
    agg = aggregate(results)
    assert agg.precision_at_5 == pytest.approx(0.2)
    assert agg.hit_at_1 == pytest.approx(0.5)
    assert agg.latency_ms_mean == pytest.approx(20.0)
    assert (agg.queries_run, agg.queries_errored) == (3, 1)


# --------------------------------------------------------------------------------------------
# 16.5 / 16.6: report contents and location


def test_report_contents_and_path(ws: Path) -> None:
    before = datetime.now().astimezone().replace(microsecond=0)
    assert _run(ws, "--queries", str(_queries_of(ws, 15))) == 0
    (path,) = _reports(ws)
    assert path.parent == ws / "data" / "eval"
    assert REPORT_NAME.fullmatch(path.name)

    rep = json.loads(path.read_text(encoding="utf-8"))
    assert set(rep) == {"run_at", "config", "aggregate", "queries"}
    run_at = datetime.fromisoformat(rep["run_at"])
    assert run_at.tzinfo is not None and run_at >= before
    assert path.name.startswith(f"report-{run_at:%Y%m%d-%H%M%S}")

    assert sorted(rep["config"]) == sorted(REQ_16_5_PARAMS)
    cfg = Config(root=ws)
    assert rep["config"] == {name: getattr(cfg, name) for name in REQ_16_5_PARAMS}
    assert [r["text"] for r in rep["queries"][:3]] == ["red square", "blue square", "red square"]


def test_report_records_overridden_config_values(ws: Path) -> None:
    assert _run(ws, "--queries", str(_queries_of(ws, 15)), "--set", "top_k=50") == 0
    assert _report(ws)["config"]["top_k"] == 50


# --------------------------------------------------------------------------------------------
# 16.9: repeatability


def test_two_runs_give_identical_metrics(ws: Path) -> None:
    q = str(_queries_of(ws, 15))
    assert _run(ws, "--queries", q) == 0
    assert _run(ws, "--queries", q) == 0
    paths = _reports(ws)
    assert len(paths) == 2 and all(REPORT_NAME.fullmatch(p.name) for p in paths)
    a, b = (json.loads(p.read_text(encoding="utf-8")) for p in paths)

    def metrics(r: dict) -> tuple:
        rows = [(x["text"], x.get("precision_at_5"), x.get("hit_at_1"), x.get("error"))
                for x in r["queries"]]
        agg = r["aggregate"]
        return rows, agg["precision_at_5"], agg["hit_at_1"], agg["queries_errored"]

    assert metrics(a) == metrics(b)
    assert a["config"] == b["config"]
