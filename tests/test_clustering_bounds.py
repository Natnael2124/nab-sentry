"""Property 36: Event bounds with padding.

**Validates: Requirements 9.7, 9.8**
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from nab_sentry.search.clustering import (
    ClusterHit,
    ClusterParams,
    VideoInfo,
    cluster_hits,
)

_BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)
_CAMERAS = ["cam-a", "cam-b"]


@st.composite
def _scenarios(draw):
    """Videos with durations, hits whose offsets lie within [0, duration]."""
    n_videos = draw(st.integers(1, 3))
    videos: dict[int, VideoInfo] = {}
    video_camera: dict[int, str] = {}
    for vid in range(1, n_videos + 1):
        duration = draw(st.floats(0.0, 600.0, allow_nan=False, allow_infinity=False))
        start = _BASE + timedelta(seconds=vid * 1000)
        videos[vid] = VideoInfo(
            duration_s=duration,
            start_epoch_ms=int(start.timestamp() * 1000),
            start_time=start,
            camera_label=f"Label {vid}",
        )
        video_camera[vid] = draw(st.sampled_from(_CAMERAS))

    n_hits = draw(st.integers(1, 25))
    hits = []
    for vector_id in range(n_hits):
        vid = draw(st.sampled_from(sorted(videos)))
        offset = draw(
            st.floats(0.0, videos[vid].duration_s, allow_nan=False, allow_infinity=False)
        )
        hits.append(
            ClusterHit(
                vector_id=vector_id,
                camera_id=video_camera[vid],
                video_id=vid,
                frame_offset_s=offset,
                similarity=draw(st.floats(-1.0, 1.0, allow_nan=False)),
                thumb_path=f"t/{vector_id}.jpg",
            )
        )

    params = ClusterParams(
        merge_gap_s=draw(st.floats(0.0, 60.0, allow_nan=False)),
        padding_s=draw(st.floats(0.0, 120.0, allow_nan=False)),
        label_boost=0.05,
    )
    return hits, videos, params


def _check_bounds(events, videos, params):
    for e in events:
        duration = videos[e.video_id].duration_s
        first = e.hits[0].frame_offset_s
        last = e.hits[-1].frame_offset_s
        assert e.start_s == max(0.0, first - params.padding_s)
        assert e.end_s == min(duration, last + params.padding_s)
        assert 0.0 <= e.start_s <= first <= last <= e.end_s <= duration


@settings(max_examples=200, deadline=None)
@given(_scenarios())
def test_event_bounds_with_padding(scenario):
    hits, videos, params = scenario
    events = cluster_hits(hits, videos, "query text", params)
    assert events
    _check_bounds(events, videos, params)


def test_single_hit_event_clamped_both_sides():
    start = _BASE
    videos = {
        1: VideoInfo(
            duration_s=10.0,
            start_epoch_ms=int(start.timestamp() * 1000),
            start_time=start,
            camera_label="Front",
        )
    }
    hit = ClusterHit(1, "cam-a", 1, 4.0, 0.5, "t/1.jpg")
    params = ClusterParams(merge_gap_s=5.0, padding_s=20.0, label_boost=0.0)
    (event,) = cluster_hits([hit], videos, "x", params)
    assert (event.start_s, event.end_s) == (0.0, 10.0)

    params = ClusterParams(merge_gap_s=5.0, padding_s=1.5, label_boost=0.0)
    (event,) = cluster_hits([hit], videos, "x", params)
    assert (event.start_s, event.end_s) == (2.5, 5.5)
