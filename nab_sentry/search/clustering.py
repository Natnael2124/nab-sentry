"""Temporal clustering and ranking of search hits (Requirement 9).

Pure module: no database, FAISS, or model imports. Every ordering step uses a
total sort key so the output is independent of the input hit order (9.13).
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

__all__ = [
    "EPS",
    "ClusterHit",
    "VideoInfo",
    "ClusterParams",
    "Event",
    "query_words",
    "label_words",
    "label_matches",
    "event_score",
    "cluster_hits",
]

# Tolerance for the merge-gap comparison; shared with the property tests.
EPS = 1e-9

_WORD_SPLIT_RE = re.compile(r"[^0-9a-z]+")


@dataclass(frozen=True)
class ClusterHit:
    vector_id: int
    camera_id: str
    video_id: int
    frame_offset_s: float
    similarity: float
    thumb_path: str


@dataclass(frozen=True)
class VideoInfo:
    duration_s: float
    start_epoch_ms: int
    start_time: datetime  # timezone-aware
    camera_label: str


@dataclass(frozen=True)
class ClusterParams:
    merge_gap_s: float
    padding_s: float
    label_boost: float


@dataclass(frozen=True)
class Event:
    camera_id: str
    video_id: int
    hits: tuple[ClusterHit, ...]  # sorted by (frame_offset_s, vector_id)
    start_s: float
    end_s: float
    score: float  # Event_Score after Label_Boost
    relative_score: float
    thumb_path: str


def _words(text: str) -> list[str]:
    return [w for w in _WORD_SPLIT_RE.split(text.lower()) if w]


def query_words(text: str) -> set[str]:
    """Lowercased query words of 3 or more characters (9.10)."""
    return {w for w in _words(text) if len(w) >= 3}


def label_words(label: str) -> set[str]:
    """Lowercased words of a camera label, split on non-alphanumerics."""
    return set(_words(label))


def label_matches(query: str, label: str) -> bool:
    """True if any query word (len >= 3) equals a word of the label."""
    return bool(query_words(query) & label_words(label))


def event_score(sims: Sequence[float]) -> float:
    """Mean of the top-3 similarities (or of all if fewer than 3).

    Similarities are sorted descending first so the summation order, and so the
    float result, does not depend on the order of ``sims``.
    """
    if not sims:
        raise ValueError("event_score requires at least one similarity")
    top = sorted(sims, reverse=True)[:3]
    return sum(top) / len(top)


def _hit_key(h: ClusterHit) -> tuple[float, int]:
    return (h.frame_offset_s, h.vector_id)


def _build_event(
    group: list[ClusterHit],
    video: VideoInfo,
    qwords: set[str],
    median: float,
    params: ClusterParams,
) -> Event:
    first, last = group[0], group[-1]
    start = max(0.0, first.frame_offset_s - params.padding_s)
    end = min(video.duration_s, last.frame_offset_s + params.padding_s)

    score = event_score([h.similarity for h in group])
    if qwords & label_words(video.camera_label):
        score += params.label_boost  # added once, however many words match

    # Representative: max similarity, ties by earliest offset then lowest id.
    rep = min(group, key=lambda h: (-h.similarity, h.frame_offset_s, h.vector_id))

    return Event(
        camera_id=first.camera_id,
        video_id=first.video_id,
        hits=tuple(group),
        start_s=start,
        end_s=end,
        score=score,
        relative_score=score - median,
        thumb_path=rep.thumb_path,
    )


def cluster_hits(
    hits: Sequence[ClusterHit],
    videos: Mapping[int, VideoInfo],
    query: str,
    params: ClusterParams,
) -> list[Event]:
    """Merge hits into ranked Events following the design's algorithm."""
    # 1. Empty input (9.15).
    if not hits:
        return []

    # 2. Median similarity of all hits received; sorted for determinism.
    median = statistics.median(sorted(h.similarity for h in hits))

    # 3. Group by (camera, video) and sort each group by a total key (9.1, 9.13).
    groups: dict[tuple[str, int], list[ClusterHit]] = defaultdict(list)
    for h in hits:
        groups[(h.camera_id, h.video_id)].append(h)

    qwords = query_words(query)
    events: list[Event] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=_hit_key)
        video = videos[key[1]]

        # 4. Merge-gap walk: gap exactly equal to merge_gap merges (9.2, 9.3).
        current: list[ClusterHit] = [group[0]]
        for prev, cur in zip(group, group[1:]):
            if cur.frame_offset_s - prev.frame_offset_s > params.merge_gap_s + EPS:
                events.append(_build_event(current, video, qwords, median, params))
                current = [cur]
            else:
                current.append(cur)
        # 5. Bounds, score, boost, relative score, thumbnail.
        events.append(_build_event(current, video, qwords, median, params))

    # 6. Final ordering (9.12).
    def order_key(e: Event) -> tuple[float, float, str, int, float]:
        abs_start_ms = videos[e.video_id].start_epoch_ms + e.start_s * 1000.0
        return (-e.score, abs_start_ms, e.camera_id, e.video_id, e.start_s)

    events.sort(key=order_key)
    return events
