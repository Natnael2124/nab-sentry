"""Property 37: Event scoring, label boost, and representative thumbnail.

**Validates: Requirements 9.9, 9.10, 9.11, 9.14**
"""

from __future__ import annotations

import re
import statistics
from datetime import datetime, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from nab_sentry.search.clustering import (
    ClusterHit,
    ClusterParams,
    VideoInfo,
    cluster_hits,
    event_score,
)

# --- Local strategies (kept here so concurrent test files do not collide) ---

# Mixed-case words, including short ones (< 3 chars) that must never trigger a boost.
_VOCAB = ["Dock", "dock", "GATE", "gate", "North", "yard", "Lot", "ab", "x", "cam2", "loading"]
_SEPS = [" ", "-", "_", ", ", ".", "  /  "]


@st.composite
def _phrases(draw, min_words: int = 0) -> str:
    words = draw(st.lists(st.sampled_from(_VOCAB), min_size=min_words, max_size=4))
    if not words:
        return draw(st.sampled_from(["", " ", "--"]))
    out = words[0]
    for w in words[1:]:
        out += draw(st.sampled_from(_SEPS)) + w
    return out


# Small similarity pool so ties at the maximum are common.
_sims = st.one_of(
    st.sampled_from([0.1, 0.25, 0.5, 0.5, 0.75, 0.9]),
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
)
# Coarse offsets so equal offsets (tie on offset) also happen.
_offsets = st.integers(min_value=0, max_value=40).map(lambda n: n * 0.5)


@st.composite
def _scenarios(draw):
    n_videos = draw(st.integers(min_value=1, max_value=3))
    cameras = ["cam-a", "cam-b"]
    videos: dict[int, VideoInfo] = {}
    video_camera: dict[int, str] = {}
    for vid in range(1, n_videos + 1):
        cam = draw(st.sampled_from(cameras))
        video_camera[vid] = cam
        videos[vid] = VideoInfo(
            duration_s=30.0,
            start_epoch_ms=1_700_000_000_000 + vid * 60_000,
            start_time=datetime(2023, 11, 14, tzinfo=timezone.utc),
            camera_label=draw(_phrases(min_words=1)),
        )
    n_hits = draw(st.integers(min_value=1, max_value=12))
    ids = draw(
        st.lists(st.integers(min_value=0, max_value=10_000), min_size=n_hits, max_size=n_hits, unique=True)
    )
    hits = []
    for vector_id in ids:
        vid = draw(st.integers(min_value=1, max_value=n_videos))
        hits.append(
            ClusterHit(
                vector_id=vector_id,
                camera_id=video_camera[vid],
                video_id=vid,
                frame_offset_s=draw(_offsets),
                similarity=draw(_sims),
                thumb_path=f"thumbs/{vector_id}.jpg",
            )
        )
    params = ClusterParams(
        merge_gap_s=draw(st.sampled_from([0.5, 2.0, 5.0])),
        padding_s=draw(st.sampled_from([0.0, 1.0])),
        label_boost=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
    )
    query = draw(_phrases())
    return hits, videos, query, params


# --- Independent oracle for the label match (Req 9.10 wording) ---

def _split(text: str) -> set[str]:
    # Words split on whitespace and punctuation, compared case-insensitively.
    return {w for w in re.split(r"[\W_]+", text.lower()) if w}


def _expected_match(query: str, label: str) -> bool:
    q = {w for w in _split(query) if len(w) >= 3}
    return bool(q & _split(label))


def _top3_mean(sims: list[float]) -> float:
    top = sorted(sims, reverse=True)[:3]
    return sum(top) / len(top)


@settings(max_examples=300, deadline=None)
@given(_scenarios())
def test_property_37_scoring_boost_relative_and_thumbnail(scenario):
    """Feature: nab-sentry, Property 37: Event scoring, label boost, and representative thumbnail.

    **Validates: Requirements 9.9, 9.10, 9.11, 9.14**
    """
    hits, videos, query, params = scenario
    events = cluster_hits(hits, videos, query, params)
    median = statistics.median([h.similarity for h in hits])

    assert events
    for ev in events:
        sims = [h.similarity for h in ev.hits]
        label = videos[ev.video_id].camera_label
        base = _top3_mean(sims)
        expected = base + (params.label_boost if _expected_match(query, label) else 0.0)

        # 9.9 / 9.10 / 9.11: top-3 mean, boost added exactly once or not at all.
        assert abs(ev.score - expected) <= 1e-9
        # 9.9: Relative_Score uses the final (boosted) score minus median of all hits.
        assert abs(ev.relative_score - (ev.score - median)) <= 1e-9

        # 9.14: thumbnail of max-similarity hit, ties by earliest offset, then lowest id.
        rep = sorted(ev.hits, key=lambda h: (-h.similarity, h.frame_offset_s, h.vector_id))[0]
        assert ev.thumb_path == rep.thumb_path
        best = max(sims)
        holders = [h for h in ev.hits if h.thumb_path == ev.thumb_path]
        assert holders and holders[0].similarity == best
        assert holders[0].frame_offset_s == min(h.frame_offset_s for h in ev.hits if h.similarity == best)


# --- Unit examples ---

def _video(label: str) -> dict[int, VideoInfo]:
    return {
        1: VideoInfo(
            duration_s=100.0,
            start_epoch_ms=0,
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            camera_label=label,
        )
    }


def _hit(vid: int, off: float, sim: float) -> ClusterHit:
    return ClusterHit(vid, "cam-a", 1, off, sim, f"t{vid}.jpg")


_P = ClusterParams(merge_gap_s=10.0, padding_s=0.0, label_boost=0.2)


def test_event_score_top3_and_fewer():
    assert abs(event_score([0.9, 0.1, 0.8, 0.7]) - 0.8) <= 1e-9
    assert abs(event_score([0.4, 0.6]) - 0.5) <= 1e-9


def test_boost_added_once_when_multiple_words_match():
    hits = [_hit(1, 0.0, 0.5), _hit(2, 1.0, 0.7)]
    (ev,) = cluster_hits(hits, _video("North Loading-Dock"), "north dock", _P)
    assert abs(ev.score - (0.6 + 0.2)) <= 1e-9
    assert abs(ev.relative_score - (0.8 - 0.6)) <= 1e-9


def test_no_boost_for_short_or_partial_words():
    hits = [_hit(1, 0.0, 0.5)]
    (ev,) = cluster_hits(hits, _video("ab Docks"), "ab dock", _P)
    assert ev.score == 0.5


def test_thumbnail_tie_uses_earliest_offset():
    hits = [_hit(5, 3.0, 0.9), _hit(7, 1.0, 0.9), _hit(2, 2.0, 0.4)]
    (ev,) = cluster_hits(hits, _video("Gate"), "yard", _P)
    assert ev.thumb_path == "t7.jpg"
