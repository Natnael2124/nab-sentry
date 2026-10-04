"""Property 38: Event ordering (Requirement 9.12).

Strategies are local to this file. Similarities, offsets, padding, and video
start times come from small pools so that score ties and absolute-start ties
happen often, exercising the tie-break chain.
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
_BASE_MS = int(_BASE.timestamp() * 1000)

_CAMERAS = ["cam-a", "cam-b", "cam-c"]
_LABELS = ["Front Door", "Loading Dock", "Parking Lot"]
_SIMS = [0.25, 0.5, 0.75]
_DURATION_S = 60.0


@st.composite
def _scenarios(draw):
    """Videos with colliding start times plus hits drawn from small pools."""
    n_videos = draw(st.integers(min_value=1, max_value=4))
    videos: dict[int, VideoInfo] = {}
    video_camera: dict[int, str] = {}
    for vid in range(1, n_videos + 1):
        # Start offsets in seconds from a small pool so absolute starts collide.
        start_off_s = draw(st.sampled_from([0, 5, 10]))
        cam_idx = draw(st.integers(min_value=0, max_value=len(_CAMERAS) - 1))
        videos[vid] = VideoInfo(
            duration_s=_DURATION_S,
            start_epoch_ms=_BASE_MS + start_off_s * 1000,
            start_time=_BASE + timedelta(seconds=start_off_s),
            camera_label=_LABELS[cam_idx],
        )
        video_camera[vid] = _CAMERAS[cam_idx]

    n_hits = draw(st.integers(min_value=0, max_value=25))
    hits = []
    for vector_id in range(n_hits):
        vid = draw(st.sampled_from(sorted(videos)))
        offset = float(draw(st.integers(min_value=0, max_value=int(_DURATION_S) - 1)))
        hits.append(
            ClusterHit(
                vector_id=vector_id,
                camera_id=video_camera[vid],
                video_id=vid,
                frame_offset_s=offset,
                similarity=draw(st.sampled_from(_SIMS)),
                thumb_path=f"thumb/{vector_id}.jpg",
            )
        )
    hits = draw(st.permutations(hits))

    params = ClusterParams(
        merge_gap_s=float(draw(st.sampled_from([1, 3, 10]))),
        padding_s=float(draw(st.sampled_from([0, 2, 5]))),
        label_boost=draw(st.sampled_from([0.0, 0.25])),
    )
    query = draw(st.sampled_from(["person at door", "red truck", "dock parking"]))
    return hits, videos, query, params


def _abs_start_ms(e, videos) -> float:
    return videos[e.video_id].start_epoch_ms + e.start_s * 1000.0


@settings(max_examples=300)
@given(_scenarios())
def test_events_ordered_by_score_then_abs_start_then_camera(scenario):
    """Feature: nab-sentry, Property 38: Event ordering

    **Validates: Requirements 9.12**
    """
    hits, videos, query, params = scenario
    events = cluster_hits(hits, videos, query, params)

    for a, b in zip(events, events[1:]):
        # Score descending.
        assert a.score >= b.score
        if a.score == b.score:
            # Then earlier absolute start first.
            sa, sb = _abs_start_ms(a, videos), _abs_start_ms(b, videos)
            assert sa <= sb
            if sa == sb:
                # Then camera ID ascending.
                assert a.camera_id <= b.camera_id


def test_tie_break_example():
    """Equal scores: earlier absolute start wins, then lower camera ID."""
    videos = {
        1: VideoInfo(_DURATION_S, _BASE_MS + 10_000, _BASE + timedelta(seconds=10), "X"),
        2: VideoInfo(_DURATION_S, _BASE_MS, _BASE, "Y"),
        3: VideoInfo(_DURATION_S, _BASE_MS, _BASE, "Z"),
        4: VideoInfo(_DURATION_S, _BASE_MS, _BASE, "W"),
    }
    hits = [
        ClusterHit(0, "cam-a", 1, 0.0, 0.5, "t0"),  # abs start +10 s
        ClusterHit(1, "cam-c", 2, 0.0, 0.5, "t1"),  # abs start +0 s, cam-c
        ClusterHit(2, "cam-b", 3, 0.0, 0.5, "t2"),  # abs start +0 s, cam-b
        ClusterHit(3, "cam-z", 4, 0.0, 0.9, "t3"),  # highest score
    ]
    params = ClusterParams(merge_gap_s=1.0, padding_s=0.0, label_boost=0.0)
    events = cluster_hits(hits, videos, "nothing", params)
    assert [e.camera_id for e in events] == ["cam-z", "cam-b", "cam-c", "cam-a"]
