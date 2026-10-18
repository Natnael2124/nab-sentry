"""ByteTrack-style multi-object tracker (V2 Requirement 10, task 10.1).

Per frame and per class:

1. Predict every live track's box: last box + velocity x elapsed time (constant velocity).
2. Match **high** detections to predicted boxes greedily by IoU (highest first, IoU >=
   ``match_iou``).
3. Match the tracks still unmatched to **low** detections the same way (ByteTrack's second
   pass: a person who briefly scores low keeps their track).
4. Unmatched high detections start tentative tracks; low detections never start one.
5. A tentative track is confirmed after ``min_hits`` matches and dropped at its first miss
   (one-off false detections never become tracks). A confirmed track ends when it has gone
   unmatched for more than ``buffer_s`` seconds.

Greedy matching instead of the Hungarian algorithm is fine at <= 20 objects per frame and keeps
the code dependency-free and deterministic: pairs are taken by IoU descending, ties by track
key then detection index.

Times are seconds on one clock across videos: :meth:`ByteTracker.begin_video` sets the start
of the next video so its ``offset_s`` values become absolute and tracks can continue from one
Segment into the next (task 10.3).

Every track keeps the observations not yet handed out; :meth:`ByteTracker.drain` returns them
for confirmed and ended tracks - including the hits from before confirmation - so storage
(task 11) writes each observation exactly once.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field

from nab_sentry.ingest.detector import Detection
from nab_sentry.ingest.sources import DecodedFrame

Box = tuple[float, float, float, float]

VELOCITY_ALPHA = 0.5  # weight of the newest velocity measurement in the moving average


@dataclass(frozen=True)
class Observation:
    t: float  # seconds on the tracker clock
    box: tuple[int, int, int, int]
    conf: float
    high: bool


@dataclass
class Track:
    key: int
    cls: str
    box: Box
    velocity: Box = (0.0, 0.0, 0.0, 0.0)  # per second, for each box coordinate
    hits: int = 1
    first_t: float = 0.0
    last_t: float = 0.0
    confirmed: bool = False
    pending: list[Observation] = field(default_factory=list)  # not yet drained

    def predict(self, t: float) -> Box:
        dt = t - self.last_t
        return tuple(b + v * dt for b, v in zip(self.box, self.velocity))  # type: ignore[return-value]


@dataclass(frozen=True)
class Assignment:
    """One detection of the latest update and the track it went to (``None``: no track)."""

    detection: Detection
    track_key: int | None
    confirmed: bool


@dataclass(frozen=True)
class TrackRecord:
    """Drained data of one track: new observations since the last drain."""

    key: int
    cls: str
    first_t: float
    last_t: float
    observations: tuple[Observation, ...]
    ended: bool


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def greedy_match(tracks: Sequence[Track], boxes: Sequence[Box], preds: Sequence[Box],
                 thr: float) -> list[tuple[int, int]]:
    """One-to-one ``(track index, box index)`` pairs by IoU descending, IoU >= ``thr``."""
    pairs = []
    for i, p in enumerate(preds):
        for j, b in enumerate(boxes):
            v = iou(p, b)
            if v >= thr and v > 0:
                pairs.append((-v, tracks[i].key, j, i))
    pairs.sort()
    used_t: set[int] = set()
    used_b: set[int] = set()
    out = []
    for _v, _key, j, i in pairs:
        if i in used_t or j in used_b:
            continue
        used_t.add(i)
        used_b.add(j)
        out.append((i, j))
    return out


class ByteTracker:
    """Implements :class:`nab_sentry.track.TrackSink`."""

    def __init__(self, match_iou: float = 0.3, min_hits: int = 3, buffer_s: float = 2.0) -> None:
        if not 0 < match_iou <= 1:
            raise ValueError(f"match_iou must be in (0, 1], got {match_iou}")
        if min_hits < 1:
            raise ValueError(f"min_hits must be >= 1, got {min_hits}")
        if buffer_s <= 0:
            raise ValueError(f"buffer_s must be > 0, got {buffer_s}")
        self.match_iou = float(match_iou)
        self.min_hits = int(min_hits)
        self.buffer_s = float(buffer_s)
        self._next_key = 1
        self._tracks: list[Track] = []
        self._ended: list[Track] = []
        self._base_t = 0.0
        self._now: float | None = None
        self.last_assignments: list[Assignment] = []

    # -- TrackSink -----------------------------------------------------------------------------

    @property
    def active(self) -> bool:
        return bool(self._tracks)

    def update(self, frame: DecodedFrame, detections: Sequence[Detection]) -> None:
        self.step(self._base_t + float(frame.offset_s), detections)

    # -- clock ---------------------------------------------------------------------------------

    def begin_video(self, start_t: float) -> None:
        """Offsets of the next video's frames start at ``start_t`` on the tracker clock."""
        self._base_t = float(start_t)

    # -- core ----------------------------------------------------------------------------------

    @property
    def tracks(self) -> list[Track]:
        """Live tracks (tentative and confirmed), in key order."""
        return list(self._tracks)

    def step(self, t: float, detections: Sequence[Detection]) -> list[Assignment]:
        if self._now is not None and t < self._now:
            raise ValueError(f"time went backwards: {t} < {self._now}")
        self._now = t
        self.expire(t)

        assigned: dict[int, Track] = {}  # detection index -> track
        matched: set[int] = set()  # keys of tracks matched in this step
        classes = sorted({d.cls for d in detections} | {tr.cls for tr in self._tracks})
        for cls in classes:
            tracks = [tr for tr in self._tracks if tr.cls == cls]
            preds = [tr.predict(t) for tr in tracks]
            for want_high in (True, False):
                free = [k for k, tr in enumerate(tracks) if tr.key not in matched]
                idx = [n for n, d in enumerate(detections)
                       if d.cls == cls and d.high == want_high and n not in assigned]
                pairs = greedy_match([tracks[k] for k in free],
                                     [tuple(map(float, detections[n].box)) for n in idx],
                                     [preds[k] for k in free], self.match_iou)
                for fi, bj in pairs:
                    tr, n = tracks[free[fi]], idx[bj]
                    self._hit(tr, t, detections[n])
                    matched.add(tr.key)
                    assigned[n] = tr

        # Unmatched tracks: a tentative track dies at its first miss.
        self._tracks = [tr for tr in self._tracks if tr.key in matched or tr.confirmed]
        # Unmatched high detections start tentative tracks.
        for n, d in enumerate(detections):
            if n not in assigned and d.high:
                tr = Track(self._new_key(), d.cls, tuple(map(float, d.box)), first_t=t, last_t=t,
                           confirmed=self.min_hits <= 1)
                tr.pending.append(Observation(t, tuple(d.box), float(d.conf), d.high))
                self._tracks.append(tr)
                assigned[n] = tr
        self._tracks.sort(key=lambda tr: tr.key)
        self.last_assignments = [
            Assignment(d, assigned[n].key if n in assigned else None,
                       bool(n in assigned and assigned[n].confirmed))
            for n, d in enumerate(detections)
        ]
        return self.last_assignments

    def _new_key(self) -> int:
        key = self._next_key
        self._next_key += 1
        return key

    def _hit(self, tr: Track, t: float, d: Detection) -> None:
        box = tuple(map(float, d.box))
        dt = t - tr.last_t
        if dt > 0:
            meas = tuple((b - o) / dt for b, o in zip(box, tr.box))
            tr.velocity = tuple(VELOCITY_ALPHA * m + (1 - VELOCITY_ALPHA) * v
                                for m, v in zip(meas, tr.velocity))  # type: ignore[assignment]
        tr.box = box  # type: ignore[assignment]
        tr.last_t = t
        tr.hits += 1
        tr.pending.append(Observation(t, tuple(d.box), float(d.conf), d.high))
        if tr.hits >= self.min_hits:
            tr.confirmed = True

    def expire(self, t: float) -> None:
        """End confirmed tracks unmatched for more than ``buffer_s`` at time ``t``."""
        keep = []
        for tr in self._tracks:
            if t - tr.last_t > self.buffer_s:
                if tr.confirmed:
                    self._ended.append(tr)
            else:
                keep.append(tr)
        self._tracks = keep

    def end_all(self) -> list[int]:
        """End every live track (scene cut, gap between Segments); returns the ended keys."""
        ended = [tr.key for tr in self._tracks if tr.confirmed]
        self._ended.extend(tr for tr in self._tracks if tr.confirmed)
        self._tracks = []
        return ended

    def drain(self) -> list[TrackRecord]:
        """New observations of confirmed tracks (back-filled from before confirmation) and of
        tracks that ended; each observation is returned exactly once."""
        out = []
        for tr in self._ended:
            out.append(TrackRecord(tr.key, tr.cls, tr.first_t, tr.last_t, tuple(tr.pending), True))
        self._ended = []
        for tr in self._tracks:
            if tr.confirmed and tr.pending:
                out.append(TrackRecord(tr.key, tr.cls, tr.first_t, tr.last_t, tuple(tr.pending), False))
                tr.pending = []
        return sorted(out, key=lambda r: r.key)

    # -- rollback ------------------------------------------------------------------------------

    def snapshot(self) -> object:
        return copy.deepcopy((self._next_key, self._tracks, self._ended, self._base_t, self._now,
                              self.last_assignments))

    def restore(self, snap: object) -> None:
        (self._next_key, self._tracks, self._ended, self._base_t, self._now,
         self.last_assignments) = copy.deepcopy(snap)  # type: ignore[misc]
