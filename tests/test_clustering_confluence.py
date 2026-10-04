"""Property 39: clustering is independent of hit order (Requirement 9.13).

Also covers the empty-input example of Requirement 9.15.
Strategies are kept local to this module.
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
_LABELS = ["Front Door", "back-yard", "Garage", "driveway cam"]
_QUERIES = ["person at front door", "car in driveway", "", "dog", "garage"]

# Small value pools so ties in similarity and frame offset are common.
_SIMS = st.one_of(
    st.sampled_from([0.1, 0.25, 0.3, 0.3, 0.5]),
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
)
_OFFSETS = st.one_of(
    st.sampled_from([0.0, 1.0, 2.0, 2.0, 5.0, 10.0]),
    st.floats(min_value=0.0, max_value=120.0, allow_nan=False),
)


@st.composite
def _scenario(draw):
    n_videos = draw(st.integers(min_value=1, max_value=4))
    videos: dict[int, VideoInfo] = {}
    video_camera: dict[int, str] = {}
    for vid in range(1, n_videos + 1):
        cam = draw(st.sampled_from(["cam-a", "cam-b", "cam-c"]))
        # Few distinct start times so absolute-start ties also occur.
        start_ms = draw(st.sampled_from([0, 1_000, 60_000]))
        videos[vid] = VideoInfo(
            duration_s=draw(st.floats(min_value=1.0, max_value=150.0)),
            start_epoch_ms=1_700_000_000_000 + start_ms,
            start_time=_BASE + timedelta(milliseconds=start_ms),
            camera_label=draw(st.sampled_from(_LABELS)),
        )
        video_camera[vid] = cam

    n_hits = draw(st.integers(min_value=0, max_value=25))
    vector_ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=10_000),
            min_size=n_hits,
            max_size=n_hits,
            unique=True,
        )
    )
    hits = []
    for vec_id in vector_ids:
        vid = draw(st.sampled_from(sorted(videos)))
        hits.append(
            ClusterHit(
                vector_id=vec_id,
                camera_id=video_camera[vid],
                video_id=vid,
                frame_offset_s=draw(_OFFSETS),
                similarity=draw(_SIMS),
                thumb_path=f"thumb_{vec_id}.jpg",
            )
        )

    params = ClusterParams(
        merge_gap_s=draw(st.sampled_from([0.0, 1.0, 3.0, 5.0])),
        padding_s=draw(st.sampled_from([0.0, 1.0, 2.5])),
        label_boost=draw(st.sampled_from([0.0, 0.05, 0.1])),
    )
    query = draw(st.sampled_from(_QUERIES))
    return hits, videos, query, params


@settings(max_examples=200, deadline=None)
@given(data=st.data())
def test_cluster_hits_independent_of_hit_order(data):
    """Feature: nab-sentry, Property 39: Clustering is independent of hit order.

    **Validates: Requirements 9.13**
    """
    hits, videos, query, params = data.draw(_scenario())
    permuted = data.draw(st.permutations(hits))

    expected = cluster_hits(hits, videos, query, params)
    actual = cluster_hits(permuted, videos, query, params)

    # Exact equality: every float sum in clustering is order-independent.
    assert actual == expected


def test_cluster_hits_empty_input_returns_empty_list():
    """Requirement 9.15: no hits -> no Events."""
    videos = {
        1: VideoInfo(
            duration_s=10.0,
            start_epoch_ms=0,
            start_time=_BASE,
            camera_label="Front Door",
        )
    }
    params = ClusterParams(merge_gap_s=3.0, padding_s=1.0, label_boost=0.1)
    assert cluster_hits([], videos, "person", params) == []
