"""Property 9: Selected offsets are exact, strictly increasing, and in range.

Duration convention: the design defines video duration as ``D = n / fps``
(``decoded_index_count / fps``), so the inclusive range checked is ``[0, n / fps]``.
Since frame indices run ``0 .. n-1``, every offset is in fact at most ``(n-1) / fps``;
that tighter bound is asserted as well.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from nab_sentry.ingest.sampler import Sampler, sample_indices
from tests.fakes import FakeSource

fps_values = st.one_of(
    st.sampled_from([1.0, 5.0, 10.0, 15.0, 24.0, 25.0, 29.97, 30.0, 60.0, 23.976]),
    st.floats(min_value=0.05, max_value=120.0, allow_nan=False, allow_infinity=False),
)
rates = st.floats(min_value=0.1, max_value=30.0, allow_nan=False, allow_infinity=False)


@st.composite
def videos(draw):
    n = draw(st.integers(min_value=0, max_value=300))
    fps = draw(fps_values)
    rate = draw(rates)
    undecodable = draw(st.frozensets(st.integers(min_value=0, max_value=max(0, n - 1)),
                                     max_size=max(0, n // 3))) if n else frozenset()
    return n, fps, rate, undecodable


# Feature: nab-sentry, Property 9: Selected offsets are exact, strictly increasing, and in range
@settings(max_examples=200, deadline=None)
@given(videos())
def test_selected_offsets_exact_increasing_in_range(video):
    """**Validates: Requirements 1.1, 2.2, 2.4**"""
    n, fps, rate, undecodable = video
    source = FakeSource(n_frames=n, fps=fps, undecodable=undecodable, width=2, height=2)
    selected = list(Sampler(rate).select(source.frames()))
    duration = n / fps

    # Req 2.2: offset is exactly index / fps.
    for frame in selected:
        assert frame.offset_s == frame.index / fps
        assert frame.index not in undecodable

    # Req 2.4: strictly increasing in selection order, within [0, D].
    offsets = [f.offset_s for f in selected]
    assert all(a < b for a, b in zip(offsets, offsets[1:]))
    for off in offsets:
        assert 0.0 <= off <= duration
        assert off <= (n - 1) / fps

    # Req 1.1: first yielded frame is at 0.0 when frame 0 is decodable,
    # and the Sampler always selects the first yielded frame (target k = 0).
    if n > 0 and 0 not in undecodable:
        assert selected and selected[0].index == 0 and selected[0].offset_s == 0.0

    # Streaming selection agrees with the pure reference.
    assert [f.index for f in selected] == sample_indices(
        n, fps, rate, decodable=lambda i: i not in undecodable)


# Feature: nab-sentry, Property 9 (source side of Req 1.1)
@settings(max_examples=100, deadline=None)
@given(videos())
def test_source_offsets_start_at_zero_and_never_decrease(video):
    """**Validates: Requirements 1.1**"""
    n, fps, _rate, undecodable = video
    source = FakeSource(n_frames=n, fps=fps, undecodable=undecodable, width=2, height=2)
    offsets = [f.offset_s for f in source.frames()]
    assert all(a <= b for a, b in zip(offsets, offsets[1:]))
    if n > 0 and 0 not in undecodable:
        assert offsets[0] == 0.0


def test_example_25fps_rate_1():
    """Concrete example: 100 frames at 25 fps, 1 fps sampling -> offsets 0,1,2,3."""
    source = FakeSource(n_frames=100, fps=25.0, width=2, height=2)
    offsets = [f.offset_s for f in Sampler(1.0).select(source.frames())]
    assert offsets == [0.0, 1.0, 2.0, 3.0]


def test_example_first_frame_undecodable():
    """Frame 0 undecodable: first selection is frame 1 at 1/fps, not 0.0."""
    source = FakeSource(n_frames=30, fps=10.0, undecodable={0}, width=2, height=2)
    selected = list(Sampler(1.0).select(source.frames()))
    assert selected[0].index == 1
    assert selected[0].offset_s == 0.1
