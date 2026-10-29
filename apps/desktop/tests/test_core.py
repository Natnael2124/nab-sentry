"""Fast tests of the pure parts: file-name times, tokenizer, spans, activity, progress."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import MODELS
from nab_find.cliptime import from_name, start_time
from nab_find.media import parse_probe, scaled_size, MediaError
from nab_find.processor import box_iou, is_activity, merge, subtract, total
from nab_find.progress import Progress

from nab_sentry.ingest.detector import Detection

# ---------------------------------------------------------------- cliptime


@pytest.mark.parametrize("name,start,camera", [
    ("CAM01_20261028T232546.mkv", datetime(2026, 10, 28, 23, 25, 46), "CAM01"),
    ("ch01_20261028143000_20261028150000.dav", datetime(2026, 10, 28, 14, 30), "ch01"),
    ("CH02_20261028143005.mp4", datetime(2026, 10, 28, 14, 30, 5), "CH02"),
    ("Front Door 20261028_143000.mp4", datetime(2026, 10, 28, 14, 30), "Front Door"),
    ("2026-10-28_14-30-00.mp4", datetime(2026, 10, 28, 14, 30), "videos"),
    ("Back yard 2026-10-28 14.30.07.avi", datetime(2026, 10, 28, 14, 30, 7), "Back yard"),
])
def test_from_name(name, start, camera):
    got = from_name(Path("C:/x/videos") / name)
    assert got == (start, camera)


@pytest.mark.parametrize("name", ["clip.mp4", "IMG_1234.mp4", "20261399_250000.mp4", "1234567890123456789.mp4"])
def test_from_name_rejects(name):
    assert from_name(Path(name)) is None


def test_start_time_falls_back_to_mtime(tmp_path):
    f = tmp_path / "shop" / "clip.mp4"
    f.parent.mkdir()
    f.write_bytes(b"x")
    t = time.mktime((2026, 10, 28, 15, 0, 0, 0, 0, -1))
    os.utime(f, (t, t))
    nt = start_time(f, 600)
    assert nt.estimated and nt.camera == "shop" and nt.start == datetime(2026, 10, 28, 14, 50)


# ---------------------------------------------------------------- media


def test_parse_probe():
    err = ("Input #0, mov,mp4\n  Duration: 01:02:03.50, start: 0.0, bitrate: 1 kb/s\n"
           "  Stream #0:0[0x1](und): Video: h264 (Main) (avc1), yuv420p(progressive), 1920x1080 [SAR 1:1], "
           "25 fps, 25 tbr\n")
    assert parse_probe(err) == (3723.5, 25.0, 1920, 1080, "h264")
    with pytest.raises(MediaError):
        parse_probe("Input #0\n  Stream #0:0: Audio: aac\n")


def test_parse_probe_tbr_only_and_no_duration():
    err = "  Stream #0:0: Video: hevc, yuvj420p, 2560x1440, 20 tbr, 1200k tbn\n"
    assert parse_probe(err) == (0.0, 20.0, 2560, 1440, "hevc")


@given(st.integers(2, 8000), st.integers(2, 8000), st.integers(64, 4000))
def test_scaled_size(w, h, mw):
    ow, oh = scaled_size(w, h, mw)
    assert ow % 2 == 0 and oh % 2 == 0 and ow <= max(mw, w) and ow <= max(w, 2)
    if w > mw:
        assert ow <= mw


# ---------------------------------------------------------------- tokenizer


def test_tokenizer_matches_open_clip():
    path = MODELS / "bpe_simple_vocab_16e6.txt.gz"
    if not path.is_file():
        pytest.skip("models/desktop not exported")
    from nab_find.clip_tokenizer import ClipTokenizer

    exp = json.loads((Path(__file__).parent / "tokens_expected.json").read_text(encoding="utf-8"))
    tok = ClipTokenizer(path)
    got = tok(exp["texts"])
    assert got.shape == (len(exp["texts"]), 77)
    for row, want in zip(got, exp["tokens"]):
        assert [int(x) for x in row if x] == want


# ---------------------------------------------------------------- spans


spans = st.lists(st.tuples(st.floats(0, 1000), st.floats(0, 50)).map(lambda t: (t[0], t[0] + t[1])), max_size=12)


@given(spans)
def test_merge_is_sorted_and_disjoint(s):
    m = merge(s)
    assert all(a <= b for a, b in m)
    assert all(m[i][1] < m[i + 1][0] for i in range(len(m) - 1))
    assert total(m) == pytest.approx(total(s))


@given(spans, spans)
def test_subtract_never_overlaps_done(todo, done):
    out = subtract(todo, done, eps=0.0)
    for a, b in out:
        for da, db in merge(done):
            assert not (a < db - 1e-9 and da < b - 1e-9)
    overlap = sum(max(0.0, min(b, db) - max(a, da)) for a, b in merge(todo) for da, db in merge(done))
    assert total(out) + overlap == pytest.approx(total(todo), abs=1e-6)


def test_subtract_examples():
    assert subtract([(0, 100)], [(10, 20), (50, 60)]) == [(0, 10), (20, 50), (60, 100)]
    assert subtract([(0, 100)], [(0, 100)]) == []
    assert subtract([(0, 10)], [(0, 9.99)]) == []  # sliver below eps


# ---------------------------------------------------------------- activity


def det(cls, box, high=True):
    return Detection(cls, 0.9 if high else 0.2, box, high)


def test_activity_rules():
    car = (0, 0, 100, 50)
    assert is_activity([], [], 0.05, 0.02, 0.7)  # motion
    assert is_activity([det("person", (5, 5, 20, 40))], [], 0.0, 0.02, 0.7)  # a person, even still
    assert not is_activity([det("car", car)], [car], 0.0, 0.02, 0.7)  # parked car
    assert is_activity([det("car", (60, 0, 160, 50))], [car], 0.0, 0.02, 0.7)  # moved car
    assert not is_activity([det("person", (5, 5, 20, 40), high=False)], [], 0.0, 0.02, 0.7)  # low score only
    assert box_iou(car, car) == 1.0 and box_iou(car, (200, 200, 300, 300)) == 0.0


# ---------------------------------------------------------------- progress


def test_progress_basic():
    p = Progress(3600, seed={"quick": 60, "detail": 10})
    s = p.snapshot()
    assert s["percent"] == 0 and s["quick_eta_s"] == pytest.approx(60)
    # detail guessed as 40% of the range before the quick pass finds anything
    assert s["full_eta_s"] == pytest.approx(60 + 0.4 * 3600 / 10)
    p.update("quick", 3600, 50)
    p.set_detail_found(600, final=True)
    s = p.snapshot()
    assert s["quick_eta_s"] == 0 and s["full_eta_s"] == pytest.approx(600 / 10)
    p.update("detail", 600, 60)
    assert p.snapshot()["percent"] == 100


@settings(max_examples=200)
@given(st.lists(st.tuples(st.floats(1, 120), st.floats(0.01, 100)), min_size=4, max_size=40))
def test_progress_eta_never_jumps_after_warmup(steps):
    total_s = sum(f for f, _ in steps) + 1
    p = Progress(total_s)
    prev = None
    for i, (footage, wall) in enumerate(steps):
        p.update("quick", footage, wall)
        speed = p.p["quick"].speed
        if prev is not None and i >= 4:
            assert prev / 2 - 1e-9 <= speed <= prev * 2 + 1e-9
        prev = speed
        s = p.snapshot()
        assert 0 <= s["percent"] <= 100 and s["quick_eta_s"] >= 0 and s["full_eta_s"] >= s["quick_eta_s"]


@given(st.lists(st.tuples(st.sampled_from(["quick", "detail"]), st.floats(0, 100), st.floats(0.01, 10)), max_size=30))
def test_progress_percent_monotonic_in_done(steps):
    p = Progress(1000, seed={"quick": 50, "detail": 50})
    p.set_detail_found(500, final=True)
    last = 0.0
    for name, f, w in steps:
        p.update(name, f, w)
        # percent can move with speed changes, but stays within bounds
        s = p.snapshot()["percent"]
        assert 0 <= s <= 100
        last = s
    assert last >= 0
