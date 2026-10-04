"""Property 10: Sample count bounds.

**Validates: Requirements 2.5, 2.6, 2.7**

Floating-point note: the bound ``floor(D * r)`` with ``D = n / fps`` is computed
with exact rationals (``fractions.Fraction`` of the float ``fps`` and ``rate``
values actually passed to the sampler), so the expected bound never suffers
rounding of its own (e.g. ``n * r / fps`` landing on ``2.9999999999999996``
instead of ``3``). The sampler itself compares in floating point with the shared
``EPS = 1e-9`` slack; that slack only ever lets a frame satisfy its target
earlier, which cannot push the count past ``floor(D * r) + 1`` because
``EPS * rate`` is far below one frame period times the rate.
"""

from __future__ import annotations

import math
from fractions import Fraction

from hypothesis import given, strategies as st

from nab_sentry.ingest.sampler import Sampler, sample_indices
from tests.fakes import FakeSource

MAX_FRAMES = 1500

# Sample rates in the allowed range, mixing common decimal values (inexact in
# binary floating point) with arbitrary floats.
rates = st.one_of(
    st.sampled_from([0.1, 0.2, 0.25, 0.3, 0.5, 1.0, 2.0, 3.0, 7.5, 10.0, 29.97, 30.0]),
    st.floats(min_value=0.1, max_value=30.0, allow_nan=False, allow_infinity=False),
)

COMMON_FPS = [1.0, 5.0, 7.5, 10.0, 12.5, 15.0, 23.976, 24.0, 25.0, 29.97, 30.0,
              50.0, 59.94, 60.0, 120.0]


def _floor_dr(n: int, fps: float, rate: float) -> int:
    """floor(D * r) with D = n / fps, computed exactly."""
    return math.floor(Fraction(n) * Fraction(rate) / Fraction(fps))


@st.composite
def fast_videos(draw):
    """(n, fps, rate) with fps >= rate."""
    rate = draw(rates)
    fps = draw(st.one_of(
        st.sampled_from([f for f in COMMON_FPS if f >= rate] or [rate]),
        st.just(rate),
        st.floats(min_value=rate, max_value=120.0, allow_nan=False, allow_infinity=False),
    ))
    n = draw(st.integers(min_value=1, max_value=MAX_FRAMES))
    return n, fps, rate


@st.composite
def slow_videos(draw):
    """(n, fps, rate) with fps < rate."""
    rate = draw(rates.filter(lambda r: r > 0.1))
    fps = draw(st.floats(min_value=0.01, max_value=rate, exclude_max=True,
                         allow_nan=False, allow_infinity=False))
    n = draw(st.integers(min_value=1, max_value=MAX_FRAMES))
    return n, fps, rate


def _stream_count(source: FakeSource, rate: float) -> tuple[list[int], int]:
    sampler = Sampler(rate)
    indices = [fr.index for fr in sampler.select(source.frames())]
    return indices, sampler.selected_count


@given(fast_videos())
def test_fully_decodable_count_within_floor_bounds(video):
    """Req 2.5: fps >= rate, all frames decode -> count in [floor(D r), floor(D r) + 1]."""
    n, fps, rate = video
    lo = _floor_dr(n, fps, rate)

    ref = sample_indices(n, fps, rate)
    assert lo <= len(ref) <= lo + 1, (n, fps, rate, len(ref), lo)

    streamed, count = _stream_count(FakeSource(n_frames=n, fps=fps), rate)
    assert count == len(streamed)
    assert lo <= count <= lo + 1, (n, fps, rate, count, lo)


@given(slow_videos(), st.data())
def test_low_fps_selects_every_decoded_frame_once(video, data):
    """Req 2.6: fps < rate -> every decoded frame is selected exactly once."""
    n, fps, rate = video
    bad = data.draw(st.frozensets(st.integers(min_value=0, max_value=n - 1), max_size=n))
    decoded = [i for i in range(n) if i not in bad]

    assert sample_indices(n, fps, rate, lambda i: i not in bad) == decoded

    streamed, count = _stream_count(FakeSource(n_frames=n, fps=fps, undecodable=bad), rate)
    assert streamed == decoded
    assert count == len(decoded)


@given(fast_videos(), st.data())
def test_failed_frames_lower_count_by_at_most_failure_count(video, data):
    """Req 2.7: with m failed frames (fps >= rate), count >= floor(D r) - m."""
    n, fps, rate = video
    bad = data.draw(st.one_of(
        st.frozensets(st.integers(min_value=0, max_value=n - 1), max_size=min(n, 50)),
        # A contiguous run of failures (the hardest case for catching up).
        st.tuples(st.integers(0, n - 1), st.integers(1, n)).map(
            lambda t: frozenset(range(t[0], min(n, t[0] + t[1])))),
    ))
    m = len(bad)
    lo = _floor_dr(n, fps, rate)

    ref = sample_indices(n, fps, rate, lambda i: i not in bad)
    assert len(ref) >= lo - m, (n, fps, rate, sorted(bad), len(ref), lo)

    source = FakeSource(n_frames=n, fps=fps, undecodable=bad)
    streamed, count = _stream_count(source, rate)
    assert source.failed_frames == m
    assert streamed == ref
    assert count >= lo - m, (n, fps, rate, sorted(bad), count, lo)
