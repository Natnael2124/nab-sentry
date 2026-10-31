"""Processing a case: the quick pass, then the detail pass (design: processor.py).

Quick pass, per video, over the part inside the case range:
  read keyframes (or 0.5 frames/s when keyframes are far apart), run the detector on each, and keep
  (thumbnail + whole-frame embedding) every frame with motion or a person, or a new or moved vehicle,
  and at least one every 30 s. Those frames also mark *activity* stretches.
Detail pass, over the activity stretches padded by a few seconds:
  2 frames/s, detector + V2 ByteTracker; per tracked person/vehicle one crop embedding per 3 s
  (the largest, most confident sighting in that window), and a Visit row (times, path, top speed,
  appearance tags).

Work is committed in chunks of about a minute of footage; a run can stop at any point (pause,
cancel, crash) and resume from the committed spans.
"""

from __future__ import annotations

import ctypes
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from nab_find.media import FrameReader, MediaError, VideoInfo
from nab_find.models import Models
from nab_find.progress import Progress
from nab_find.store import CaseStore, Chunk, CropRow, FrameRow, Video, VisitRow

from nab_sentry.describe.attributes import AttributeTagger, phrase  # noqa: E402  (sys.path set by models)
from nab_sentry.ingest.detector import Detection
from nab_sentry.ingest.motion import changed_fraction, downscale_gray
from nab_sentry.track.bytetrack import ByteTracker

log = logging.getLogger("nab_find.processor")

VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle"}


@dataclass
class Tuning:
    max_keyframe_gap_s: float = 4.0  # keyframes farther apart than this: sample at quick_rate
    quick_rate: float = 0.5
    keep_every_s: float = 30.0
    motion_threshold: float = 0.02
    pixel_delta: int = 25
    gate_width: int = 320
    detail_rate: float = 2.0
    pad_s: float = 3.0
    crop_every_s: float = 3.0
    min_crop_px: int = 24
    max_width: int = 1280
    chunk_s: float = 60.0
    moved_iou: float = 0.7  # a vehicle box overlapping less than this with all last boxes is new/moved


# ---------------------------------------------------------------- pure helpers


def intersect(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float] | None:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if hi > lo else None


def merge(spans: list[tuple[float, float]], gap: float = 0.0) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def subtract(spans: list[tuple[float, float]], done: list[tuple[float, float]], eps: float = 0.05
             ) -> list[tuple[float, float]]:
    """Parts of ``spans`` not covered by ``done``; slivers shorter than ``eps`` are dropped."""
    out = []
    done = merge(done)
    for a, b in merge(spans):
        cur = a
        for da, db in done:
            if db <= cur or da >= b:
                continue
            if da > cur:
                out.append((cur, da))
            cur = max(cur, db)
        if b > cur:
            out.append((cur, b))
    return [(a, b) for a, b in out if b - a >= eps]


def total(spans: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in merge(spans))


def box_iou(a, b) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def object_cue(dets: list[Detection], recent_vehicles: list[tuple], moved_iou: float) -> bool:
    """A person, or a vehicle not seen in that place in the recent quick frames (a car parked all day
    is not activity; one that arrives or leaves is)."""
    for d in dets:
        if not d.high:
            continue
        if d.cls == "person":
            return True
        if d.cls in VEHICLES and all(box_iou(d.box, v) < moved_iou for v in recent_vehicles):
            return True
    return False


class ActivityRule:
    """Motion counts at once. Without motion, an object cue (:func:`object_cue`) must repeat in two
    quick frames in a row: measured on a static, noisy street (task 2.8), one-frame detections (a
    "person" in the noise, a parked car's box flickering) were 25 false triggers per 10 minutes; a
    real person passing moves pixels, so they never depend on the second frame."""

    def __init__(self, threshold: float, moved_iou: float, memory: int = 5) -> None:
        self.threshold, self.moved_iou, self.memory = threshold, moved_iou, memory
        self._vehicles: list[list[tuple]] = []
        self._cue_before = False

    def __call__(self, dets: list[Detection], fraction: float) -> bool:
        # vehicles of the frames before the previous one: a car that moved is "new" in both of the two
        # frames that confirm it (with only one frame of history, that one is used)
        older = self._vehicles[:-1] if len(self._vehicles) > 1 else self._vehicles
        recent = [b for frame in older for b in frame]
        cue = object_cue(dets, recent, self.moved_iou)
        active = fraction >= self.threshold or (cue and self._cue_before)
        self._cue_before = cue
        self._vehicles = (self._vehicles + [[d.box for d in dets if d.high and d.cls in VEHICLES]])[-(self.memory + 1):]
        return active


def prefetch(it: Iterator, n: int = 6) -> Iterator:
    """Run ``it`` in a thread, ``n`` items ahead (decoding overlaps the models)."""
    q: queue.Queue = queue.Queue(maxsize=n)
    done = object()
    err: list[BaseException] = []
    stop = threading.Event()

    def work() -> None:
        try:
            for x in it:
                while not stop.is_set():
                    try:
                        q.put(x, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
        except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer
            err.append(exc)
        finally:
            q.put(done)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    try:
        while True:
            x = q.get()
            if x is done:
                break
            yield x
    finally:
        stop.set()
        while t.is_alive():
            try:
                q.get_nowait()
            except queue.Empty:
                t.join(0.05)
    if err:
        raise err[0]


def write_jpeg(img: np.ndarray, path: Path, max_side: int, quality: int = 80) -> None:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        path.write_bytes(buf.tobytes())


class Cancelled(Exception):
    pass


# ---------------------------------------------------------------- visits


@dataclass
class VisitAcc:
    key: int
    video_id: int
    cls: str
    t0: float
    t1: float
    w: int
    h: int
    n_obs: int = 0
    max_speed: float = 0.0
    path: list[tuple[float, float, float]] = field(default_factory=list)
    vectors: list[np.ndarray] = field(default_factory=list)
    best_area: float = 0.0
    thumb: str | None = None
    tags: dict = field(default_factory=dict)
    dirty: bool = True

    def observe(self, t: float, box) -> None:
        self.n_obs += 1
        self.t0, self.t1 = min(self.t0, t), max(self.t1, t)
        cx, cy = (box[0] + box[2]) / 2 / self.w, (box[1] + box[3]) / 2 / self.h
        if not self.path or t - self.path[-1][0] >= 1.0 - 1e-6:
            if self.path:
                pt, px, py = self.path[-1]
                self.max_speed = max(self.max_speed, math.hypot(cx - px, cy - py) / (t - pt))
            self.path.append((round(t, 2), round(cx, 4), round(cy, 4)))
        self.dirty = True

    def row(self) -> VisitRow:
        return VisitRow(self.key, self.video_id, self.cls, self.t0, self.t1, self.n_obs, round(self.max_speed, 4),
                        self.path, self.tags, self.thumb)


# ---------------------------------------------------------------- processor


class Processor:
    """Runs a case to completion (or until paused/cancelled). ``on_progress(snapshot)`` is called after
    every chunk; ``on_event(name)`` with "quick_done" and "done"."""

    def __init__(self, store: CaseStore, models: Models, tuning: Tuning | None = None,
                 seed_speeds: dict[str, float] | None = None,
                 on_progress: Callable[[dict], None] | None = None,
                 on_event: Callable[[str], None] | None = None) -> None:
        self.store, self.models = store, models
        self.t = tuning or Tuning()
        self.on_progress = on_progress or (lambda s: None)
        self.on_event = on_event or (lambda e: None)
        self._pause = threading.Event()
        self._cancel = threading.Event()
        self._tagger: AttributeTagger | None = None
        self.seed = seed_speeds or {}
        self.progress: Progress | None = None
        self._next_key = 1

    # -- control
    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def cancel(self) -> None:
        self._cancel.set()
        self._pause.clear()

    @property
    def paused(self) -> bool:
        return self._pause.is_set()

    def _checkpoint(self) -> None:
        while self._pause.is_set() and not self._cancel.is_set():
            time.sleep(0.1)
        if self._cancel.is_set():
            raise Cancelled()

    # -- plan
    def video_range(self, v: Video) -> tuple[float, float] | None:
        """The part of ``v`` (video seconds) inside the case range."""
        rng = self.store.meta().get("range")
        whole = (0.0, v.duration_s)
        if not rng:
            return whole
        return intersect(whole, (rng[0] - v.start_ts, rng[1] - v.start_ts))

    def plan(self) -> dict[int, dict]:
        out = {}
        for v in self.store.videos():
            if v.status == "error":
                continue
            r = self.video_range(v)
            if r is None:
                continue
            out[v.video_id] = {"video": v, "range": r,
                               "quick_todo": subtract([r], self.store.spans(v.video_id, "quick"))}
        return out

    def detail_spans(self, v: Video, r: tuple[float, float]) -> list[tuple[float, float]]:
        padded = [(max(r[0], a - self.t.pad_s), min(r[1], b + self.t.pad_s)) for a, b in self.store.motion(v.video_id)]
        return merge(padded, gap=self.t.pad_s)

    # -- run
    def run(self) -> str:
        """Returns "done", "cancelled" or "error"."""
        _keep_awake(True)
        try:
            return self._run()
        except Cancelled:
            return "cancelled"
        finally:
            _keep_awake(False)

    def _run(self) -> str:
        plan = self.plan()
        quick_total = sum(total([p["range"]]) for p in plan.values())
        prog = self.progress = Progress(quick_total, seed=self.seed)
        prog.update("quick", quick_total - sum(total(p["quick_todo"]) for p in plan.values()), 0)
        self._publish_detail_estimate(plan, final=False)
        self.on_progress(prog.snapshot())
        for vid, p in plan.items():
            for a, b in p["quick_todo"]:
                self._safe(p["video"], lambda: self.quick_pass(p["video"], a, b))
        self._publish_detail_estimate(plan, final=True)
        self.on_event("quick_done")
        self.on_progress(prog.snapshot())
        self.store.forget_track_keys()
        for vid, p in plan.items():
            v = p["video"]
            if self.store.video(vid).status == "error":
                continue
            todo = subtract(self.detail_spans(v, p["range"]), self.store.spans(vid, "detail"))
            for a, b in todo:
                self._safe(v, lambda: self.detail_pass(v, a, b))
        self.on_event("done")
        self.on_progress(prog.snapshot())
        return "done"

    def _safe(self, v: Video, fn: Callable[[], None]) -> None:
        try:
            fn()
        except MediaError as exc:
            log.warning("video %s failed: %s", v.path, exc)
            self.store.update_video(v.video_id, status="error", error=str(exc))

    def _publish_detail_estimate(self, plan: dict, final: bool) -> None:
        found = sum(total(self.detail_spans(p["video"], p["range"])) for p in plan.values())
        done = sum(total(self.store.spans(v, "detail")) for v in plan)
        st = self.progress.p["detail"]
        st.done_s = max(st.done_s, done)
        self.progress.set_detail_found(found, final=final)

    def _info(self, v: Video) -> VideoInfo:
        return VideoInfo(v.duration_s, v.fps, v.width, v.height, "", v.keyframe_gap_s)

    # -- quick pass
    def quick_pass(self, v: Video, a: float, b: float) -> None:
        t = self.t
        info = self._info(v)
        keyframes = v.keyframe_gap_s is not None and v.keyframe_gap_s <= t.max_keyframe_gap_s
        reader = FrameReader(v.path, info, a, b, mode="keyframes" if keyframes else "rate", rate=t.quick_rate,
                             max_width=t.max_width)
        prev_small = None
        prev_t = a
        last_kept = -math.inf
        rule = ActivityRule(t.motion_threshold, t.moved_iou)
        chunk = Chunk(v.video_id, "quick", (a, a))
        kept: list[tuple[float, np.ndarray]] = []
        wall0 = time.perf_counter()
        act_start: float | None = None
        for off, img in prefetch(reader.frames()):
            self._checkpoint()
            # a chunk covers [start, off): the frame at ``off`` belongs to the next one, so a resume
            # that starts at the committed end never processes it twice
            if off - chunk.span[0] >= t.chunk_s:
                if act_start is not None:
                    chunk.motion.append((act_start, off))
                    act_start = off
                wall0 = self._commit_quick(v, chunk, kept, off, wall0)
                chunk, kept = Chunk(v.video_id, "quick", (off, off)), []
            small = downscale_gray(img, t.gate_width)
            frac = 0.0 if prev_small is None else changed_fraction(prev_small, small, t.pixel_delta)
            prev_small = small
            dets = self.models.yolo.detect(img)
            active = rule(dets, frac)
            if active:
                if act_start is None:
                    act_start = prev_t
            elif act_start is not None:
                chunk.motion.append((act_start, off))
                act_start = None
            if active or off - last_kept >= t.keep_every_s - 1e-6:
                kept.append((off, img.copy()))
                last_kept = off
            prev_t = off
        if act_start is not None:
            chunk.motion.append((act_start, b))
        self._commit_quick(v, chunk, kept, b, wall0)

    def _commit_quick(self, v: Video, chunk: Chunk, kept, end: float, wall0: float) -> float:
        if kept:
            vecs = self.models.clip.encode_images([im for _, im in kept])
            for (off, im), vec in zip(kept, vecs):
                name = f"f{v.video_id}_{int(round(off * 1000))}.jpg"
                write_jpeg(im, self.store.thumbs / name, 320)
                chunk.frames.append(FrameRow(v.video_id, off, name, vec))
        chunk.span = (chunk.span[0], end)
        self.store.commit(chunk)
        now = time.perf_counter()
        self.progress.update("quick", end - chunk.span[0], now - wall0)
        self._publish_detail_estimate_partial()
        self.on_progress(self.progress.snapshot())
        return now

    def _publish_detail_estimate_partial(self) -> None:
        plan = self.plan()
        found = sum(total(self.detail_spans(p["video"], p["range"])) for p in plan.values())
        self.progress.set_detail_found(found, final=False)

    # -- detail pass
    def _key(self) -> int:
        k = self._next_key
        self._next_key += 1
        return k

    def tagger(self) -> AttributeTagger:
        if self._tagger is None:
            self._tagger = AttributeTagger(self.models.clip.encode_one_text)
        return self._tagger

    def detail_pass(self, v: Video, a: float, b: float) -> None:
        t = self.t
        reader = FrameReader(v.path, self._info(v), a, b, mode="rate", rate=t.detail_rate, max_width=t.max_width)
        tracker = ByteTracker(match_iou=0.3, min_hits=2, buffer_s=2.0)
        keys: dict[int, int] = {}  # tracker key -> run-unique visit key
        visits: dict[int, VisitAcc] = {}
        cand: dict[int, tuple[int, float, float, tuple, np.ndarray]] = {}  # key -> (window, score, t, box, crop)
        pending_crops: list[tuple[int, float, tuple, np.ndarray]] = []  # (visit key, t, box, crop)
        chunk_start = a
        wall0 = time.perf_counter()
        last_t = a
        for off, img in prefetch(reader.frames()):
            self._checkpoint()
            if off - chunk_start >= t.chunk_s:  # [chunk_start, off), as in the quick pass
                self._commit_detail(v, visits, pending_crops, chunk_start, off, wall0, final=False)
                pending_crops = []
                chunk_start, wall0 = off, time.perf_counter()
            h, w = img.shape[:2]
            dets = self.models.yolo.detect(img)
            assigns = tracker.step(off, dets)
            confirmed = {tr.key for tr in tracker.tracks if tr.confirmed}
            for asg in assigns:
                if asg.track_key is None or not asg.detection.high:
                    continue
                d = asg.detection
                x1, y1, x2, y2 = d.box
                if min(x2 - x1, y2 - y1) < t.min_crop_px:
                    continue
                win = int(off // t.crop_every_s)
                score = (x2 - x1) * (y2 - y1) * d.conf
                old = cand.get(asg.track_key)
                if old is not None and old[0] != win:
                    self._flush_cand(asg.track_key, old, confirmed, keys, pending_crops)
                    old = None
                if old is None or score > old[1]:
                    cand[asg.track_key] = (win, score, off, d.box, img[y1:y2, x1:x2].copy())
            for rec in tracker.drain():
                vk = keys.setdefault(rec.key, self._key())
                acc = visits.get(vk)
                if acc is None:
                    acc = visits[vk] = VisitAcc(vk, v.video_id, rec.cls, rec.first_t, rec.last_t, w, h)
                for ob in rec.observations:
                    acc.observe(ob.t, ob.box)
                if rec.ended and rec.key in cand:
                    self._flush_cand(rec.key, cand.pop(rec.key), {rec.key}, keys, pending_crops)
            last_t = off
        tracker.end_all()
        for rec in tracker.drain():
            vk = keys.setdefault(rec.key, self._key())
            acc = visits.get(vk) or visits.setdefault(
                vk, VisitAcc(vk, v.video_id, rec.cls, rec.first_t, rec.last_t, v.width, v.height))
            for ob in rec.observations:
                acc.observe(ob.t, ob.box)
        for key, c in list(cand.items()):
            if key in keys:
                self._flush_cand(key, c, {key}, keys, pending_crops)
        self._commit_detail(v, visits, pending_crops, chunk_start, max(b, last_t), wall0, final=True)

    @staticmethod
    def _flush_cand(key: int, c, confirmed: set[int], keys: dict[int, int], out: list) -> None:
        if key in confirmed and key in keys:
            out.append((keys[key], c[2], c[3], c[4]))

    def _commit_detail(self, v: Video, visits: dict[int, VisitAcc], crops, start: float, end: float, wall0: float,
                       final: bool) -> None:
        chunk = Chunk(v.video_id, "detail", (start, end))
        if crops:
            vecs = self.models.clip.encode_images([c[3] for c in crops])
            for (vk, off, box, crop), vec in zip(crops, vecs):
                name = f"c{v.video_id}_{int(round(off * 1000))}_{vk}.jpg"
                write_jpeg(crop, self.store.thumbs / name, 200)
                chunk.crops.append(CropRow(v.video_id, off, vk, tuple(int(x) for x in box), name, vec))
                acc = visits[vk]
                acc.vectors.append(vec)
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > acc.best_area:
                    acc.best_area, acc.thumb = area, name
                acc.dirty = True
        for acc in visits.values():
            if acc.dirty and acc.vectors:
                tags = self.tagger().tag(acc.cls, acc.vectors)
                acc.tags = {k: tg for k, (tg, _p) in tags.items()}
                acc.tags["phrase"] = phrase(acc.cls, tags)
        need = {c.visit_key for c in chunk.crops}
        chunk.visits = [acc.row() for acc in visits.values() if acc.dirty and (acc.vectors or acc.key in need)]
        self.store.commit(chunk)
        for acc in visits.values():
            if acc.vectors:
                acc.dirty = False
        now = time.perf_counter()
        self.progress.update("detail", end - start, now - wall0)
        self.on_progress(self.progress.snapshot())


def _keep_awake(on: bool) -> None:
    """Stop Windows from sleeping while processing (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)."""
    try:
        flags = 0x80000000 | (0x00000001 if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except (AttributeError, OSError):
        pass
