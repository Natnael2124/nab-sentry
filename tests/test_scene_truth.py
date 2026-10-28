"""Ground truth of continuous recordings and prepare_footage.py (V2 tasks 5.4, 17.3)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.scene_truth import SceneTruthError, parse_scene_truth, segment_offsets, to_offset

ROOT = Path(__file__).resolve().parents[1]
START = datetime(2026, 10, 26, 23, 30, 0)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


prep = _load("prepare_footage")


def test_clock_times_and_seconds():
    assert to_offset(12.5, START, "x") == 12.5
    assert to_offset("23:31:05", START, "x") == 65.0
    assert to_offset("00:10:00", START, "x") == 40 * 60  # past midnight = next day
    assert to_offset("23:29:59", START, "x") == -1.0  # a label rounded down to just before the start
    for bad in ("25:00:00", "23:61:00", "soon", True):
        with pytest.raises(SceneTruthError):
            to_offset(bad, START, "x")


@given(st.integers(0, 86398))  # 1 s before start counts as just before it, not the next day
def test_every_second_of_the_next_day_maps_back(sec):
    from datetime import timedelta
    t = START + timedelta(seconds=sec)
    assert to_offset(t.strftime("%H:%M:%S"), START, "x") == sec


def test_parse_the_documented_example():
    gt = parse_scene_truth(yaml.safe_load("""
camera: CAM01
start: '2026-10-26T18:00:00'
duration_s: 3600
objects:
  - name: neighbour walks to the gate
    class: person
    visible: ['18:04:10', '18:04:31']
expected_alerts:
  - {rule: gate-in, t: '18:04:22'}
"""), ["gate-in"])
    assert gt["objects"][0]["visible"] == [250.0, 271.0]
    assert gt["expected_alerts"] == [{"rule": "gate-in", "t": 262.0}]


@pytest.mark.parametrize("change,msg", [
    ({"objects": [{"class": "dog", "visible": [1, 2]}]}, "class"),
    ({"objects": [{"class": "person", "visible": [5, 2]}]}, "ends before"),
    ({"expected_alerts": [{"rule": "nope", "t": 3}]}, "not a rule"),
    ({"start": "yesterday"}, "ISO"),
])
def test_bad_labels_are_named(change, msg):
    data = {"camera": "C", "start": "2026-10-26T18:00:00", "duration_s": 60, **change}
    with pytest.raises(SceneTruthError, match=msg):
        parse_scene_truth(data, ["gate-in"])


def test_segment_offsets_from_names():
    offs = segment_offsets([Path("C_20261026T180010.mkv"), Path("C_20261026T180000.mkv")],
                           datetime(2026, 10, 26, 18, 0, 0))
    assert [(p.name, o) for p, o in offs] == [("C_20261026T180000.mkv", 0.0), ("C_20261026T180010.mkv", 10.0)]
    with pytest.raises(SceneTruthError):
        segment_offsets([Path("clip.mkv")], START)


def test_select_segments_by_camera_and_window():
    names = ["CAM01_20261026T180000.mkv", "CAM01_20261026T190000.mkv", "CAM02_20261026T180500.mkv",
             "CAM01_20261026T183000.mp4", "notes.txt"]
    got = prep.select_segments(names, "CAM01", datetime(2026, 10, 26, 18, 0), datetime(2026, 10, 26, 19, 0))
    assert [n for n, _ in got] == ["CAM01_20261026T180000.mkv"]


def test_template_parses_and_lists_rules():
    rules = [{"id": "gate-in", "name": "In", "camera": "CAM01", "line": "gate", "direction": "in",
              "classes": ["person"], "trigger": {"type": "cross"}}]
    text = prep.template_yaml("CAM01", datetime(2026, 10, 26, 18, 0), 3600, rules)
    assert "gate-in: In (cross line gate, in, classes person)" in text
    gt = parse_scene_truth(yaml.safe_load(text), ["gate-in"])
    assert gt["objects"] == [] and gt["expected_alerts"] == [] and gt["duration_s"] == 3600


def test_prepare_footage_end_to_end(tmp_path: Path):
    import cv2
    import numpy as np

    videos = tmp_path / "videos"
    videos.mkdir()
    for name in ("CAM01_20261026T180000.mkv", "CAM01_20261026T180010.mkv"):
        w = cv2.VideoWriter(str(videos / name), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
        for i in range(50):
            w.write(np.full((48, 64, 3), i * 4, np.uint8))
        w.release()
    cams = tmp_path / "cameras.yaml"
    cams.write_text("cameras:\n  - {camera_id: CAM01, label: Gate, source: 'rtsp://u:secret@10.0.0.2/s',"
                    " zones: {yard: [[0, 0.5], [1, 0.5], [1, 1], [0, 1]]}}\n", encoding="utf-8")
    rules = tmp_path / "rules.yaml"
    rules.write_text("rules:\n  - {id: yard-in, name: Yard, camera: CAM01, zone: yard, classes: [person],"
                     " trigger: {type: enter}}\n  - {id: other, name: O, camera: CAM02, zone: z, classes: [car],"
                     " trigger: {type: enter}}\n", encoding="utf-8")
    out = tmp_path / "out"
    rc = prep.main(["CAM01", "--videos", str(videos), "--cameras", str(cams), "--rules", str(rules),
                    "--out", str(out)])
    assert rc == 0
    assert sorted(p.name for p in (out / "videos").iterdir()) == sorted(p.name for p in videos.iterdir())
    assert "secret" not in (out / "cameras.yaml").read_text(encoding="utf-8")
    assert [r["id"] for r in yaml.safe_load((out / "rules.yaml").read_text(encoding="utf-8"))["rules"]] == ["yard-in"]
    gt = parse_scene_truth(yaml.safe_load((out / "ground_truth.yaml").read_text(encoding="utf-8")))
    assert gt["start"] == "2026-10-26T18:00:00" and gt["duration_s"] == 20
    assert len(list((out / "contact").glob("sheet_*.jpg"))) == 1
    from nab_sentry.live.camera_config import load_cameras
    from nab_sentry.rules.config import load_rules
    load_rules(out / "rules.yaml", load_cameras(out / "cameras.yaml"))  # what acceptance.py loads
    assert prep.main(["CAM09", "--videos", str(videos), "--cameras", str(cams)]) == 2
    assert prep.main(["CAM01", "--videos", str(tmp_path / "none"), "--cameras", str(cams)]) == 1
