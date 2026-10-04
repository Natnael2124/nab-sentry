"""Property 48: Invalid synthetic settings are rejected without output.

Each example starts from a valid spec and breaks exactly one setting (duration, fps, or one
object's interval). Both the library entry point and the CLI must reject it, name that
setting, and leave the output directory without new files (not created if it was absent).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.synthetic import SyntheticError, SyntheticObject, SyntheticSpec, write_synthetic

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_synthetic", _ROOT / "scripts" / "make_synthetic.py")
make_synthetic = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(make_synthetic)

# If the generator ever tried to encode, this would fail with "cannot run ffmpeg" instead of
# naming the invalid setting, and the output dir would have been created.
FFMPEG_STUB = str(_ROOT / "does-not-exist" / "ffmpeg-must-not-run.exe")

LABELS = ("red square", "blue circle")
_finite = dict(allow_nan=False, allow_infinity=False)


@st.composite
def valid_base(draw):
    """(duration, fps, [(start, end), (start, end)]) that would pass validation."""
    duration = draw(st.floats(1.0, 120.0, **_finite))
    fps = draw(st.sampled_from([1, 5, 10, 25, 30]))
    intervals = []
    for _ in LABELS:
        start = draw(st.floats(0.0, duration - 0.5, **_finite))
        end = draw(st.floats(start + 0.1, duration, **_finite))
        intervals.append((start, end))
    return duration, fps, intervals


@st.composite
def one_invalid_setting(draw):
    """Returns (duration, fps, intervals, expected_setting_name)."""
    duration, fps, intervals = draw(valid_base())
    kind = draw(st.sampled_from(["duration", "fps", "start<0", "end>duration", "end<=start"]))
    if kind == "duration":
        duration = draw(
            st.one_of(
                st.floats(max_value=0.0, **_finite),
                st.just(math.nan),
                st.just(math.inf),
                st.just(-math.inf),
            )
        )
        return duration, fps, intervals, "duration"
    if kind == "fps":
        fps = draw(
            st.one_of(
                st.floats(max_value=0.0, **_finite),
                st.just(math.nan),
                st.just(math.inf),
                st.just(0),
            )
        )
        return duration, fps, intervals, "fps"

    idx = draw(st.integers(0, len(LABELS) - 1))
    if kind == "start<0":
        start = draw(st.floats(-1e6, -1e-3, **_finite))
        end = draw(st.floats(0.0, duration, **_finite))
    elif kind == "end>duration":
        start = draw(st.floats(0.0, duration - 0.5, **_finite))
        end = draw(st.floats(duration + 1e-3, duration + 1e6, **_finite))
    else:  # end <= start, both inside [0, duration]
        start = draw(st.floats(0.0, duration, **_finite))
        end = draw(st.floats(0.0, start, **_finite))
    intervals = list(intervals)
    intervals[idx] = (start, end)
    return duration, fps, intervals, f"{LABELS[idx]} interval"


def _spec(duration, fps, intervals) -> SyntheticSpec:
    (rs, re_), (bs, be) = intervals
    base = SyntheticSpec()
    return replace(
        base,
        duration_s=duration,
        fps=fps,
        objects=(
            SyntheticObject("red square", "square", "red", rs, re_),
            SyntheticObject("blue circle", "circle", "blue", bs, be),
        ),
    )


def _snapshot(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


def _prepare(parent: Path, pre_existing: bool) -> Path:
    out_dir = parent / "out"
    if pre_existing:
        out_dir.mkdir()
        (out_dir / "existing.txt").write_text("keep", encoding="utf-8")
    return out_dir


def _assert_no_output(parent: Path, before: set[str], pre_existing: bool) -> None:
    assert _snapshot(parent) == before
    assert (parent / "out").exists() == pre_existing


# Feature: nab-sentry, Property 48: Invalid synthetic settings are rejected without output
@given(case=one_invalid_setting(), pre_existing=st.booleans())
def test_write_synthetic_rejects_invalid_setting_without_output(case, pre_existing):
    """**Validates: Requirements 15.8**"""
    duration, fps, intervals, expected = case
    spec = _spec(duration, fps, intervals)
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        out_dir = _prepare(parent, pre_existing)
        before = _snapshot(parent)
        with pytest.raises(SyntheticError) as exc:
            write_synthetic(spec, out_dir, ffmpeg=FFMPEG_STUB)
        msg = str(exc.value)
        assert f"invalid setting {expected}" in msg
        assert "ffmpeg" not in msg
        _assert_no_output(parent, before, pre_existing)


# Feature: nab-sentry, Property 48: Invalid synthetic settings are rejected without output
@given(case=one_invalid_setting(), pre_existing=st.booleans())
def test_cli_exits_nonzero_naming_invalid_setting_without_output(case, pre_existing):
    """**Validates: Requirements 15.8**"""
    duration, fps, intervals, expected = case
    (rs, re_), (bs, be) = intervals
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        out_dir = _prepare(parent, pre_existing)
        before = _snapshot(parent)
        argv = [
            "--out", str(out_dir),
            f"--duration={duration!r}",
            f"--fps={fps!r}",
            "--red", repr(float(rs)), repr(float(re_)),
            "--blue", repr(float(bs)), repr(float(be)),
        ]
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(io.StringIO()):
            rc = make_synthetic.main(argv)
        err = buf.getvalue()
        assert rc != 0
        assert f"invalid setting {expected}" in err
        _assert_no_output(parent, before, pre_existing)
