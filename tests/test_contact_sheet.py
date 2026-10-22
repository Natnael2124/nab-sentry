"""Fast tests for scripts/contact_sheet.py (pure helpers + an end-to-end run on a tiny video)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from nab_sentry.ground_truth import parse_ground_truth

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "contact_sheet.py"


def _load():
    spec = importlib.util.spec_from_file_location("contact_sheet_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cs = _load()


def test_fmt_offset():
    assert cs.fmt_offset(0) == "00:00 (0s)"
    assert cs.fmt_offset(125) == "02:05 (125s)"


def test_cut_score_and_suggested_cuts():
    a = np.zeros((36, 64), np.float32)
    b = np.full((36, 64), 200, np.float32)
    assert cs.cut_score(None, a) == 0.0
    assert cs.cut_score(a, b) == 200.0
    assert cs.suggested_cuts([0.0, 3.0, 55.0, 1.0, 40.0], 40.0) == [0, 2, 4]


def test_grid_pads_to_full_rows():
    cells = [np.zeros((10, 20, 3), np.uint8)] * 7
    g = cs.grid(cells, cols=3)
    assert g.shape == (30, 60, 3)


def test_template_is_valid_ground_truth():
    text = cs.template_yaml("CAM01_20260101T000000.mp4", 61.0, [0, 12, 40])
    gt = parse_ground_truth(yaml.safe_load(text))
    assert gt.scene_cuts == (0.0, 12.0, 40.0) and gt.incidents == () and gt.duration_s == 61.0


def test_end_to_end_marks_the_cut(tmp_path: Path, monkeypatch):
    video = tmp_path / "CAMX_20260101T000000.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
    for i in range(30):  # 6 s: black for 3 s, then white
        w.write(np.full((48, 64, 3), 0 if i < 15 else 255, np.uint8))
    w.release()

    from nab_sentry import config as config_mod

    monkeypatch.setattr(cs, "Config", lambda: config_mod.Config(root=tmp_path))
    assert cs.main([str(video), "--per-sheet", "4", "--cols", "2"]) == 0
    out = tmp_path / "data" / "bench" / "contact" / video.stem
    assert sorted(f.name for f in out.glob("sheet_*.jpg")) == ["sheet_01.jpg", "sheet_02.jpg"]
    gt = parse_ground_truth(yaml.safe_load((out / "CAMX.yaml").read_text(encoding="utf-8")))
    assert gt.scene_cuts == (0.0, 3.0)
