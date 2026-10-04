"""Property tests for Thumbnail dimensions (Property 23).

Property 23: *For any* BGR image with width and height in [1, 4000], ``write_thumbnail``
writes a decodable JPEG exactly 320 pixels wide whose height is within 1 pixel of
``320 * h / w`` (minimum 1).

JPEG limit: the format cannot encode more than 65,535 px in either dimension. For very tall
sources (``h / w`` above ~204.8, e.g. 1x4000) the target height ``round(320 * h / w)`` exceeds
that limit, so the property is split in two:

* when ``round(320 * h / w) <= 65535`` a decodable JPEG of the right size is written;
* otherwise ``write_thumbnail`` raises ``ThumbnailError`` (mentioning the JPEG limit) and
  writes no file.

Only BGR (3-channel) sources are covered. Image content is a single seeded colour so that
4000x4000 inputs stay cheap to build.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from nab_sentry.ingest.transcode import JPEG_MAX_DIM, ThumbnailError, write_thumbnail

THUMB_W = 320
dims = st.integers(min_value=1, max_value=4000)
colours = st.tuples(*(st.integers(min_value=0, max_value=255) for _ in range(3)))


def _bgr(w: int, h: int, colour: tuple[int, int, int]) -> np.ndarray:
    return np.full((h, w, 3), colour, dtype=np.uint8)


def _target_h(w: int, h: int) -> int:
    return round(THUMB_W * h / w)


# Feature: nab-sentry, Property 23: Thumbnail dimensions
@given(w=dims, h=dims, colour=colours)
def test_thumbnail_is_320_wide_with_aspect_height(w: int, h: int, colour: tuple[int, int, int]) -> None:
    """**Validates: Requirements 6.4**"""
    assume(_target_h(w, h) <= JPEG_MAX_DIM)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sub" / "thumb.jpg"
        write_thumbnail(_bgr(w, h, colour), path)

        assert path.is_file()
        decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None, "thumbnail is not a decodable JPEG"
    dh, dw = decoded.shape[:2]
    assert dw == THUMB_W
    expected = max(1.0, THUMB_W * h / w)
    assert abs(dh - expected) <= 1, f"{w}x{h}: height {dh}, expected ~{expected:.3f}"


@st.composite
def too_tall(draw: st.DrawFn) -> tuple[int, int]:
    """(w, h) in [1, 4000] whose 320-wide thumbnail would exceed the JPEG height limit."""
    # h / w must exceed ~204.8, so with h <= 4000 the width is at most 19.
    w = draw(st.integers(min_value=1, max_value=19))
    h_min = int((JPEG_MAX_DIM + 0.5) * w / THUMB_W) + 1
    h = draw(st.integers(min_value=min(h_min, 4000), max_value=4000))
    return w, h


# Feature: nab-sentry, Property 23: Thumbnail dimensions (JPEG-limit case)
@given(wh=too_tall(), colour=colours)
def test_thumbnail_beyond_jpeg_limit_raises(wh: tuple[int, int], colour: tuple[int, int, int]) -> None:
    """**Validates: Requirements 6.4**"""
    w, h = wh
    assume(_target_h(w, h) > JPEG_MAX_DIM)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "thumb.jpg"
        with pytest.raises(ThumbnailError, match="exceeds the JPEG limit"):
            write_thumbnail(_bgr(w, h, colour), path)
        assert not path.exists()


@pytest.mark.parametrize(
    ("w", "h", "expected_h"),
    [(1920, 1080, 180), (640, 480, 240), (4000, 1, 1), (1, 1, 320), (20, 4000, 64000)],
)
def test_thumbnail_examples(tmp_path: Path, w: int, h: int, expected_h: int) -> None:
    path = tmp_path / "t.jpg"
    write_thumbnail(_bgr(w, h, (10, 20, 30)), path)
    decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (expected_h, THUMB_W)


def test_thumbnail_1x4000_raises(tmp_path: Path) -> None:
    path = tmp_path / "t.jpg"
    with pytest.raises(ThumbnailError, match="exceeds the JPEG limit"):
        write_thumbnail(_bgr(1, 4000, (0, 0, 0)), path)
    assert not path.exists()
