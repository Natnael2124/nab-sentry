"""Property 35: Merge-gap invariant.

For any hits and merge gap >= 0, within each Event consecutive hits (sorted by
offset) differ by at most ``merge_gap_s``, and for any two Events of the same
video the gap between the earlier Event's last hit and the later Event's first
hit is greater than ``merge_gap_s`` (both using the shared ``EPS``).

**Validates: Requirements 9.2, 9.3, 9.5**
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from nab_sentry.search.clustering import (
    EPS,
    ClusterHit,
    ClusterParams,
    VideoInfo,
    cluster_hits,
)

DURATION_S = 600.0

# Local strategies (kept in this file on purpose).
_GRID = 0.5  # offsets and gaps on a 0.5 s grid make exact-equal gaps common

_offsets = st.one_of(
    st.integers(min_value=0, max_value=200).map(lambda i: i * _GRID),
    st.floats(min_value=0.0, max_value=DURATION_S, allow_nan=False, allow_infinity=False),
)
_merge_gaps = st.one_of(
    st.just(0.0),
    st.integers(min_value=0, max_value=20).map(lambda i: i * _GRID),
    st.floats(min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False),
)


@st.composite
def _hits(draw):
    n = draw(st.integers(min_value=1, max_value=40))
    offsets = draw(st.lists(_offsets, min_size=n, max_size=n))
    # Inject duplicate offsets (e.g. frame + crop vectors from the same frame).
    dup_count = draw(st.integers(min_value=0, max_value=5))
    for _ in range(dup_count):
        offsets.append(draw(st.sampled_from(offsets)))
    hits = []
    for vid, off in enumerate(offsets, start=1):
        cam = draw(st.sampled_from(["cam_a", "cam_b"]))
        video = draw(st.sampled_from([1, 2]))
        sim = draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False))
        hits.append(ClusterHit(vid, cam, video, off, sim, f"t/{vid}.jpg"))
    return hits


def _videos():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ms = int(start.timestamp() * 1000)
    return {
        1: VideoInfo(DURATION_S, ms, start, "Front Door"),
        2: VideoInfo(DURATION_S, ms + 3_600_000, start, "Back Yard"),
    }


def _params(gap: float) -> ClusterParams:
    return ClusterParams(merge_gap_s=gap, padding_s=2.0, label_boost=0.05)


@settings(max_examples=300, deadline=None)
@given(hits=_hits(), gap=_merge_gaps)
def test_merge_gap_invariant(hits, gap):
    events = cluster_hits(hits, _videos(), "person", _params(gap))

    by_video = defaultdict(list)
    for e in events:
        offs = [h.frame_offset_s for h in e.hits]
        assert offs == sorted(offs)
        # Within an Event: consecutive hits differ by at most merge_gap (9.2).
        for a, b in zip(offs, offs[1:]):
            assert b - a <= gap + EPS, (a, b, gap)
        by_video[(e.camera_id, e.video_id)].append(e)

    # Across Events of the same video: gap strictly greater than merge_gap (9.3, 9.5).
    for evs in by_video.values():
        evs.sort(key=lambda e: e.hits[0].frame_offset_s)
        for i, earlier in enumerate(evs):
            for later in evs[i + 1:]:
                diff = later.hits[0].frame_offset_s - earlier.hits[-1].frame_offset_s
                assert diff > gap + EPS, (earlier.hits[-1], later.hits[0], gap)


def _hit(vid: int, off: float) -> ClusterHit:
    return ClusterHit(vid, "cam_a", 1, off, 0.5, f"t/{vid}.jpg")


def test_gap_exactly_equal_merges():
    events = cluster_hits([_hit(1, 10.0), _hit(2, 12.5)], _videos(), "x", _params(2.5))
    assert len(events) == 1 and len(events[0].hits) == 2


def test_gap_just_over_splits():
    events = cluster_hits([_hit(1, 10.0), _hit(2, 12.6)], _videos(), "x", _params(2.5))
    assert len(events) == 2


def test_zero_gap_merges_duplicates_only():
    hits = [_hit(1, 5.0), _hit(2, 5.0), _hit(3, 5.5)]
    events = cluster_hits(hits, _videos(), "x", _params(0.0))
    sizes = sorted(len(e.hits) for e in events)
    assert sizes == [1, 2]
