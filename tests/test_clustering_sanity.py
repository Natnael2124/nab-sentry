"""Fast example tests for nab_sentry.search.clustering (Requirement 9)."""

from datetime import datetime, timezone

import pytest

from nab_sentry.search.clustering import (
    ClusterHit,
    ClusterParams,
    VideoInfo,
    cluster_hits,
    event_score,
    label_words,
    query_words,
)

PARAMS = ClusterParams(merge_gap_s=2.0, padding_s=1.0, label_boost=0.05)
T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
EPOCH0 = int(T0.timestamp() * 1000)


def hit(vid, offset, sim, cam="cam1", video=1):
    return ClusterHit(vid, cam, video, offset, sim, f"t{vid}.jpg")


def videos(label="Gate"):
    return {
        1: VideoInfo(10.0, EPOCH0, T0, label),
        2: VideoInfo(10.0, EPOCH0 + 60_000, T0, "Reception Lobby"),
    }


def test_empty_input_returns_empty_list():
    assert cluster_hits([], videos(), "anything", PARAMS) == []


def test_words():
    assert query_words("A red-car at the GATE!") == {"red", "car", "the", "gate"}
    assert label_words("North_Gate 2") == {"north", "gate", "2"}


def test_event_score_top3_mean():
    assert event_score([0.1, 0.5, 0.4, 0.3]) == pytest.approx(0.4)
    assert event_score([0.2]) == pytest.approx(0.2)


def test_gap_exactly_merge_gap_merges_and_larger_splits():
    hits = [hit(1, 1.0, 0.3), hit(2, 3.0, 0.4), hit(3, 5.5, 0.2)]
    events = cluster_hits(hits, videos(), "red square", PARAMS)
    assert [len(e.hits) for e in events] == [2, 1]
    first = events[0]
    assert (first.start_s, first.end_s) == (0.0, 4.0)
    assert first.score == pytest.approx(0.35)
    assert first.relative_score == pytest.approx(0.35 - 0.3)
    assert first.thumb_path == "t2.jpg"
    assert (events[1].start_s, events[1].end_s) == (4.5, 6.5)


def test_padding_clamped_to_duration():
    events = cluster_hits([hit(1, 9.8, 0.5)], videos(), "x", PARAMS)
    assert events[0].end_s == 10.0


def test_label_boost_added_once():
    hits = [hit(1, 1.0, 0.3)]
    plain = cluster_hits(hits, videos("North Gate"), "truck", PARAMS)[0]
    boosted = cluster_hits(hits, videos("North Gate"), "north gate truck", PARAMS)[0]
    assert boosted.score == pytest.approx(plain.score + PARAMS.label_boost)


def test_thumbnail_tie_uses_earliest_offset():
    hits = [hit(5, 2.0, 0.4), hit(4, 1.0, 0.4)]
    assert cluster_hits(hits, videos(), "x", PARAMS)[0].thumb_path == "t4.jpg"


def test_groups_by_video_and_order_independent():
    hits = [
        hit(1, 1.0, 0.3),
        hit(2, 1.0, 0.3, cam="cam2", video=2),
        hit(3, 1.5, 0.6),
        hit(4, 1.0, 0.3, cam="cam2", video=2),
    ]
    a = cluster_hits(hits, videos(), "x", PARAMS)
    b = cluster_hits(list(reversed(hits)), videos(), "x", PARAMS)
    assert a == b
    assert all({h.video_id for h in e.hits} == {e.video_id} for e in a)
    assert sum(len(e.hits) for e in a) == len(hits)
    # Highest score first.
    assert a[0].video_id == 1


def test_score_ties_broken_by_absolute_start():
    hits = [hit(2, 1.0, 0.3, cam="cam2", video=2), hit(1, 1.0, 0.3)]
    events = cluster_hits(hits, videos(), "x", PARAMS)
    assert [e.video_id for e in events] == [1, 2]
