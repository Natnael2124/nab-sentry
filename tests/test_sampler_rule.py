"""Property 8: Sampler follows the target-time rule.

**Validates: Requirements 2.1**

For each target time ``k / rate`` (k = 0, 1, ... while the target is no greater than
the duration ``n / fps``), the selected frame is the first decodable frame at or after
that target that has not already been selected. ``Sampler.select`` over a
``FakeSource`` and the pure ``sample_indices`` reference must both agree with an
independent, target-driven oracle (including undecodable frames).
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.ingest.sampler import EPS, Sampler, sample_indices
from tests.fakes import FakeSource

# Realistic source frame rates: arbitrary floats plus common broadcast/CCTV rates.
fps_values = st.one_of(
    st.floats(min_value=1.0, max_value=60.0, allow_nan=False, allow_infinity=False),
    st.sampled_from([1.0, 5.0, 7.5, 10.0, 12.5, 15.0, 23.976, 24.0, 25.0, 29.97, 30.0, 59.94, 60.0]),
)
rate_values = st.one_of(
    st.floats(min_value=0.1, max_value=30.0, allow_nan=False, allow_infinity=False),
    st.sampled_from([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 29.97, 30.0]),
)


@st.composite
def videos(draw):
    """(n_frames, fps, undecodable indices)."""
    n = draw(st.integers(min_value=0, max_value=600))
    fps = draw(fps_values)
    if n == 0:
        bad: frozenset[int] = frozenset()
    else:
        bad = frozenset(draw(st.sets(st.integers(min_value=0, max_value=n - 1), max_size=min(n, 60))))
    return n, fps, bad


def oracle(n_frames: int, fps: float, rate: float, undecodable: frozenset[int]) -> list[int]:
    """Target-driven reference: walk targets, not frames.

    For each target ``t_k = k / rate`` with ``t_k <= n / fps``, choose the smallest
    decodable index greater than the last selected one whose offset is at or after
    ``t_k`` (with the shared ``EPS`` tolerance). Once a target finds no frame, no later
    (larger) target can, so the walk stops.
    """
    duration = n_frames / fps
    decodable = [i for i in range(n_frames) if i not in undecodable]
    chosen: list[int] = []
    pos = 0  # next candidate position in `decodable` (everything before is used or skipped)
    k = 0
    while k / rate <= duration + EPS:
        target = k / rate
        while pos < len(decodable) and decodable[pos] / fps < target - EPS:
            pos += 1
        if pos == len(decodable):
            break
        chosen.append(decodable[pos])
        pos += 1
        k += 1
    return chosen


@given(video=videos(), rate=rate_values)
def test_sampler_follows_target_time_rule(video, rate):
    n, fps, bad = video
    expected = oracle(n, fps, rate, bad)

    source = FakeSource(n_frames=n, fps=fps, undecodable=bad, width=2, height=2)
    sampler = Sampler(rate)
    streamed = list(sampler.select(source.frames()))
    streamed_idx = [f.index for f in streamed]

    assert streamed_idx == expected
    assert sample_indices(n, fps, rate, decodable=lambda i: i not in bad) == expected
    assert sampler.selected_count == len(expected)
    # Each k-th selection is at or after its target and was actually decodable.
    for k, frame in enumerate(streamed):
        assert frame.index not in bad
        assert frame.offset_s >= k / rate - EPS


def test_examples():
    # fps 10, 1 frame/s, 2.5 s: targets 0, 1, 2 -> frames 0, 10, 20.
    assert oracle(25, 10.0, 1.0, frozenset()) == [0, 10, 20]
    assert [f.index for f in Sampler(1.0).select(FakeSource(n_frames=25, fps=10.0).frames())] == [0, 10, 20]
    # Undecodable frame 10: target 1.0 falls through to frame 11.
    src = FakeSource(n_frames=25, fps=10.0, undecodable={10})
    assert [f.index for f in Sampler(1.0).select(src.frames())] == [0, 11, 20]
    # fps below rate: every frame selected exactly once.
    assert [f.index for f in Sampler(2.0).select(FakeSource(n_frames=3, fps=1.0).frames())] == [0, 1, 2]
    # Empty video selects nothing.
    assert list(Sampler(1.0).select(FakeSource(n_frames=0, fps=10.0).frames())) == []
