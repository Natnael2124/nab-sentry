"""Fast tests for scripts/benchmark.py: pure helpers, projection, --search and failure paths (no models)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark.py"


def _load():
    spec = importlib.util.spec_from_file_location("benchmark_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look up the defining module
    spec.loader.exec_module(mod)
    return mod


bench = _load()


@pytest.mark.parametrize(
    "samples",
    [[5.0], [1.0, 2.0], [3.0, 1.0, 2.0], list(range(1, 101)), [0.5, 9.0, 2.25, 7.0, 7.0, 1.0]],
)
def test_percentile_matches_numpy(samples):
    for q in (0, 50, 95, 100):
        assert bench.percentile(samples, q) == pytest.approx(float(np.percentile(samples, q)))


def test_percentile_rejects_empty_and_bad_q():
    with pytest.raises(ValueError):
        bench.percentile([], 50)
    with pytest.raises(ValueError):
        bench.percentile([1.0], 101)


def test_summarize_median_and_p95():
    s = bench.summarize([float(x) for x in range(1, 101)])
    assert s.n == 100
    assert s.median == pytest.approx(50.5)
    assert s.p95 == pytest.approx(95.05)


def test_time_iterations_counts_warmup_and_divides_per_item():
    calls = []
    ticks = iter(float(i) for i in range(1000))

    def fn():
        calls.append(1)
        return 4

    out = bench.time_iterations(fn, iters=50, warmup=5, clock=lambda: next(ticks))
    assert len(calls) == 55
    assert len(out) == 50
    assert all(v == pytest.approx(1000.0 / 4) for v in out)  # 1 s tick / 4 items


def test_report_marks_failed_stage():
    text = bench.format_report(
        [
            bench.StageResult("decode", "ms/frame", bench.summarize([1.0, 2.0, 3.0])),
            bench.StageResult("detector", "ms/frame", file="x.onnx", error="boom"),
        ]
    )
    assert "decode" in text and "2.00" in text
    assert "detector" in text and "FAILED" in text


def test_iters_below_minimum_rejected():
    with pytest.raises(SystemExit) as exc:
        bench.build_parser().parse_args(["v.mp4", "--iters", "10"])
    assert exc.value.code == 2


def test_missing_models_exits_4(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "nope.mp4"), "--set", f"models_dir={tmp_path}"],
        capture_output=True, text=True, timeout=120, cwd=ROOT,
    )
    assert proc.returncode == 4
    assert "manifest" in proc.stderr


def test_projection_formula_matches_design():
    # (3600*30*0.005 + 3600*1*0.001 + 3600*1*0.5*(0.050 + 0.002 + 3*0.020)) / 60 = 745.2 / 60
    minutes = bench.project_ingest_minutes(
        source_fps=30, sample_rate=1, pass_fraction=0.5, dets_per_passed=2,
        t_decode_ms=5, t_gate_ms=1, t_det_ms=50, t_thumb_ms=2, t_embed8_ms=20,
    )
    assert minutes == pytest.approx(12.42)


def test_projection_with_no_passed_frames_is_decode_plus_gate():
    minutes = bench.project_ingest_minutes(
        source_fps=25, sample_rate=2, pass_fraction=0.0, dets_per_passed=0,
        t_decode_ms=4, t_gate_ms=2, t_det_ms=1000, t_thumb_ms=1000, t_embed8_ms=1000,
    )
    assert minutes == pytest.approx((3600 * 25 * 0.004 + 3600 * 2 * 0.002) / 60)


def test_projection_rejects_bad_inputs():
    kw = dict(source_fps=30, sample_rate=1, pass_fraction=0.5, dets_per_passed=1,
              t_decode_ms=1, t_gate_ms=1, t_det_ms=1, t_thumb_ms=1, t_embed8_ms=1)
    for bad in ({"pass_fraction": 1.5}, {"sample_rate": 0}, {"dets_per_passed": -1}):
        with pytest.raises(ValueError):
            bench.project_ingest_minutes(**{**kw, **bad})


def test_search_fixture_rows_match_index():
    cfg = bench.Config()
    fx = bench.build_search_fixture(501, cfg)
    try:
        assert fx.index.ntotal == 501
        assert np.array_equal(np.sort(fx.index.ids()), fx.db.vector_ids())
        allowed = fx.db.allowed_vector_ids(fx.filter)
        assert allowed.size > 0
        plain, filtered = bench.bench_search(fx, cfg, iters=5)
        assert plain.n == 5 and filtered.n == 5
    finally:
        fx.db.close()


def test_search_mode_needs_no_models(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--search", "2000", "--search-iters", "100",
         "--set", f"models_dir={tmp_path / 'models'}", "--set", f"data_dir={tmp_path / 'data'}"],
        capture_output=True, text=True, timeout=300, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "search p95 unfiltered:" in proc.stdout
    assert "search p95 filtered:" in proc.stdout
    assert "over 100 queries" in proc.stdout
    assert "peak RSS" in proc.stdout


def _write_clip(path: Path, frames: int = 30) -> None:
    import cv2

    w, h = 96, 64
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    assert vw.isOpened()
    for i in range(frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        x = (i * 3) % (w - 16)
        img[20:36, x:x + 16] = (0, 0, 255)
        vw.write(img)
    vw.release()


def test_unloadable_models_report_stage_and_file_exit_1(tmp_path):
    from nab_sentry.model_files import DETECTOR_REL, EMBEDDER_REL
    from nab_sentry.startup import sha256_file

    models = tmp_path / "models"
    entries = []
    for role, rel in (("detector", DETECTOR_REL), ("embedder", EMBEDDER_REL)):
        f = models / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"not a model " + role.encode())
        entries.append({"role": role, "path": rel, "sha256": sha256_file(f)})
    (models / "manifest.json").write_text(
        json.dumps({"manifest_version": 1, "files": entries}), encoding="utf-8")
    video = tmp_path / "clip.mp4"
    _write_clip(video)
    data = tmp_path / "data"

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(video),
         "--set", f"models_dir={models}", "--set", f"data_dir={data}"],
        capture_output=True, text=True, timeout=600, cwd=ROOT,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    # model stages fail naming the stage and the file
    assert "stage=embedder_b8" in proc.stderr and "open_clip_pytorch_model.bin" in proc.stderr
    assert "stage=detector" in proc.stderr and "yolo11n.onnx" in proc.stderr
    # completed stages are still reported
    for stage in ("decode", "motion_gate", "thumbnail"):
        line = next(ln for ln in proc.stdout.splitlines() if ln.startswith(stage + " "))
        assert "FAILED" not in line
    assert "scan: fps=10" in proc.stdout
    assert "projected ingest: not available" in proc.stdout
    # thumbnail scratch dir removed
    assert not (data / "bench").exists()
