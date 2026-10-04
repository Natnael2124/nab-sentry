"""Property 34: Clustering partitions hits by video.

**Validates: Requirements 9.1, 9.4, 9.6**
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.search.clustering import (
    ClusterHit,
    ClusterParams,
    VideoInfo,
    cluster_hits,
)

_BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


@st.composite
def clustering_inputs(draw):
    """Videos (each owned by exactly one camera) plus hits with unique vector IDs."""
    n_cameras = draw(st.integers(1, 3))
    camera_ids = [f"cam{i}" for i in range(n_cameras)]
    n_videos = draw(st.integers(1, 5))

    videos: dict[int, VideoInfo] = {}
    owner: dict[int, str] = {}
    for vid in range(1, n_videos + 1):
        cam = draw(st.sampled_from(camera_ids))
        duration = draw(st.floats(0.5, 120.0, allow_nan=False, allow_infinity=False))
        start = _BASE + timedelta(seconds=draw(st.integers(0, 86_400)))
        videos[vid] = VideoInfo(
            duration_s=duration,
            start_epoch_ms=int(start.timestamp() * 1000),
            start_time=start,
            camera_label=draw(st.sampled_from(["Front Door", "Back Yard", "Garage"])),
        )
        owner[vid] = cam

    n_hits = draw(st.integers(1, 40))
    vector_ids = draw(
        st.lists(st.integers(1, 10_000), min_size=n_hits, max_size=n_hits, unique=True)
    )
    hits = []
    for vector_id in vector_ids:
        vid = draw(st.sampled_from(sorted(videos)))
        dur = videos[vid].duration_s
        # Offsets on a coarse grid so equal offsets (frame + crop) occur often.
        fps = draw(st.sampled_from([1.0, 2.0, 5.0]))
        idx = draw(st.integers(0, max(0, int(dur * fps) - 1)))
        offset = min(idx / fps, dur)
        hits.append(
            ClusterHit(
                vector_id=vector_id,
                camera_id=owner[vid],
                video_id=vid,
                frame_offset_s=offset,
                similarity=draw(st.floats(-1.0, 1.0, allow_nan=False)),
                thumb_path=f"thumbs/{vector_id}.jpg",
            )
        )

    params = ClusterParams(
        merge_gap_s=draw(st.floats(0.0, 10.0, allow_nan=False)),
        padding_s=draw(st.floats(0.0, 5.0, allow_nan=False)),
        label_boost=draw(st.floats(0.0, 0.2, allow_nan=False)),
    )
    query = draw(st.sampled_from(["person at door", "car in garage", "dog"]))
    return hits, videos, query, params


@given(clustering_inputs())
def test_events_partition_hits_by_video(inputs):
    hits, videos, query, params = inputs
    events = cluster_hits(hits, videos, query, params)

    # 9.4: every hit in an Event shares that Event's camera and video.
    for e in events:
        assert e.hits, "Events must be non-empty"
        for h in e.hits:
            assert h.camera_id == e.camera_id
            assert h.video_id == e.video_id
        # 9.1: each Event's hits are sorted by Frame_Offset (total key with vector_id).
        keys = [(h.frame_offset_s, h.vector_id) for h in e.hits]
        assert keys == sorted(keys)

    # 9.6: multiset union of Event hits equals the input; each hit in exactly one Event.
    out = [h for e in events for h in e.hits]
    assert len(out) == len(hits)
    assert Counter(out) == Counter(hits)

    # Every (camera, video) group in the input yields at least one Event.
    assert {(h.camera_id, h.video_id) for h in hits} == {
        (e.camera_id, e.video_id) for e in events
    }


def test_equal_offsets_stay_in_same_group():
    """9.1 example: a frame and a crop hit at the same offset share one Event."""
    start = _BASE
    videos = {
        1: VideoInfo(10.0, int(start.timestamp() * 1000), start, "Front Door"),
        2: VideoInfo(10.0, int(start.timestamp() * 1000), start, "Back Yard"),
    }
    hits = [
        ClusterHit(1, "cam0", 1, 3.0, 0.3, "a.jpg"),
        ClusterHit(2, "cam0", 1, 3.0, 0.4, "b.jpg"),
        ClusterHit(3, "cam1", 2, 3.0, 0.5, "c.jpg"),
    ]
    events = cluster_hits(hits, videos, "dog", ClusterParams(0.0, 1.0, 0.0))
    assert len(events) == 2
    by_video = {e.video_id: e for e in events}
    assert [h.vector_id for h in by_video[1].hits] == [1, 2]
    assert [h.vector_id for h in by_video[2].hits] == [3]
