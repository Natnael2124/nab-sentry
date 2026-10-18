"""Fast tests for scripts/vlm_bench.py pure helpers (no models, no Ollama)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "vlm_bench.py"


def _load():
    spec = importlib.util.spec_from_file_location("vlm_bench_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look up the defining module
    spec.loader.exec_module(mod)
    return mod


vb = _load()


def test_find_windows_merges_gaps_and_drops_short_runs():
    secs = [0, 1, 2, 3, 10, 11, 20, 21, 22, 23, 24]
    assert vb.find_windows(secs) == [(0.0, 3.0), (20.0, 24.0)]
    assert vb.find_windows(secs, limit=1) == [(0.0, 3.0)]
    assert vb.find_windows([]) == []


@given(st.lists(st.integers(0, 400), max_size=200), st.integers(1, 5))
def test_find_windows_properties(secs, limit):
    runs = vb.find_windows(secs, limit=limit)
    assert len(runs) <= limit
    assert all(b - a >= 3.0 for a, b in runs)
    assert runs == sorted(runs)
    assert all(a in secs and b in secs for a, b in runs)


@given(st.floats(0, 1000), st.floats(0, 100))
def test_keyframe_offsets_evenly_spaced_within_span(start, length):
    offs = vb.keyframe_offsets(start, start + length)
    assert len(offs) == 4
    assert offs[0] == pytest.approx(start)
    assert offs[-1] <= start + min(length, 8.0) + 1e-9
    gaps = np.diff(offs)
    assert np.allclose(gaps, gaps[0])


@given(st.integers(1, 2000), st.integers(1, 2000))
def test_fit_panel_shape(h, w):
    panel = vb.fit_panel(np.full((h, w, 3), 200, np.uint8))
    assert panel.shape == (vb.PANEL_H, vb.PANEL_W, 3)


def test_storyboard_is_2x2_in_reading_order():
    panels = [np.full((vb.PANEL_H, vb.PANEL_W, 3), v, np.uint8) for v in (10, 20, 30, 40)]
    grid = vb.storyboard(panels)
    assert grid.shape == (2 * vb.PANEL_H, 2 * vb.PANEL_W, 3)
    assert grid[0, 0, 0] == 10 and grid[0, -1, 0] == 20
    assert grid[-1, 0, 0] == 30 and grid[-1, -1, 0] == 40
    with pytest.raises(ValueError):
        vb.storyboard(panels[:3])


def test_bust_cache_changes_bytes_per_rep_only():
    img = np.full((4, 4, 3), 250, np.uint8)
    assert vb.bust_cache(img, 0) is img
    a, b = vb.bust_cache(img, 1), vb.bust_cache(img, 2)
    assert not np.array_equal(a, img) and not np.array_equal(a, b)
    assert np.array_equal(a[:-1], img[:-1]) and (img == 250).all()


def test_padded_crop_is_clamped():
    img = np.zeros((100, 200, 3), np.uint8)
    assert vb.padded_crop(img, (0, 0, 40, 40)).shape == (50, 50, 3)
    assert vb.padded_crop(img, (180, 80, 200, 100)).shape == (25, 25, 3)


def test_parse_timing_converts_nanoseconds():
    t = vb.parse_timing({"load_duration": 2_000_000_000, "prompt_eval_count": 300,
                         "prompt_eval_duration": 4_000_000_000, "eval_count": 50,
                         "eval_duration": 5_000_000_000}, wall_s=11.5)
    assert (t.load_s, t.prompt_tokens, t.prompt_s, t.gen_tokens, t.gen_s) == (2.0, 300, 4.0, 50, 5.0)
    assert t.gen_tok_per_s == pytest.approx(10.0)
    assert vb.parse_timing({}, 1.0).gen_tok_per_s == 0.0


def test_summarize_rows_separates_cold_from_warm():
    base = {"num_ctx": 4096, "variant": "grid", "prompt_tokens": 300, "prompt_s": 4.0,
            "gen_tokens": 80, "gen_tok_per_s": 10.0, "peak_rss_mb": 5000.0}
    rows = [{**base, "cold": True, "load_s": 9.0, "wall_s": 30.0},
            {**base, "cold": False, "load_s": 0.0, "wall_s": 12.0},
            {**base, "cold": False, "load_s": 0.0, "wall_s": 14.0}]
    [s] = vb.summarize_rows(rows)
    assert s["n"] == 2 and s["cold_load_s"] == 9.0
    assert s["wall_s"] == 13.0 and s["wall_max_s"] == 14.0
    assert "| 4096 | grid | 2 |" in vb.format_summary([s], "h")


def test_sanity_images_and_matching():
    imgs = vb.sanity_images()
    assert [n for n, *_ in imgs] == ["colour", "text"]
    assert all(img.shape == (448, 448, 3) for _, img, _, _ in imgs)
    assert vb.sanity_passed("Red.", "red") and vb.sanity_passed("HELLO 42", "42")
    assert not vb.sanity_passed("a sheet of paper with a grid of numbers", "42")
