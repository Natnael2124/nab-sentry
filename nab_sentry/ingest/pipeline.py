"""Ingest pipeline: one SQLite transaction per video, FAISS add + save before COMMIT.

Requirements 1.5, 1.9-1.11, 2.3, 2.8-2.10, 3.6-3.8, 4.5, 4.7, 4.8, 5.1, 5.2, 5.10, 6.2-6.5,
6.7-6.10, 7.1, 7.3, 7.6-7.8, 8.10 (design: "ingest/pipeline.py", key decisions 1, 2, 5, 6).

Per video (:meth:`IngestPipeline.ingest_video`):

1. Src_Hash first; skip if already in ``videos`` or ingested earlier in this run.
2. Resolve camera metadata (sidecar, then file name).
3. Open the source (``UnreadableVideo`` / ``UnknownFrameRate`` -> skip).
4. ``BEGIN IMMEDIATE``; upsert camera; ``videos`` row with ``status='ingesting'``; start the
   playback job (runs concurrently with analysis).
5. Sample + gate; per passed frame: ``frames`` row, Thumbnail, detector (errors -> warning and
   zero crops), queue the frame and its crops for batched embedding; each embedded item gets
   its ``vectors`` row and its vector is staged in memory.
6. Flush the remaining batch; zero passed frames -> ``UnreadableVideo``.
7. Wait for playback, rename ``.part``, finalize the ``videos`` row.
8. ``index.add`` + atomic ``index.save``.
9. COMMIT.

Any ``Exception`` or ``KeyboardInterrupt`` in steps 4-9 is compensated in the design's order:
kill ffmpeg -> SQLite ROLLBACK -> remove staged IDs from the index (+ re-save if it was saved)
-> delete the files recorded for this video -> log path and reason. Other ``BaseException``
types are treated as process death (no compensation beyond the SQLite rollback, which a crash
also gets for free); :meth:`IngestPipeline.reconcile` repairs that state on the next run.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from nab_sentry.config import Config, effective_track_rate
from nab_sentry.embed.clip_encoder import EmbeddingError, Encoder
from nab_sentry.ingest.detector import Detection, Detector, clamp_box
from nab_sentry.ingest.metadata import resolve_metadata, src_hash, to_aware_local
from nab_sentry.ingest.motion import GateDecision, MotionGate
from nab_sentry.ingest.sampler import DualSampler, Sampler
from nab_sentry.ingest.sources import (
    FileSource,
    UnknownFrameRate,
    UnreadableVideo,
    VideoSource,
)
from nab_sentry.ingest.transcode import PlaybackJob, default_playback, part_path, write_thumbnail
from nab_sentry.logging_setup import get_logger
from nab_sentry.store.db import MetadataStore, NewFrame, NewTrackObs, NewVector, NewVideo
from nab_sentry.store.vector_index import VectorIndex
from nab_sentry.alerts.media import ALERT_FILE_RE, draw_snapshot, snapshot_name
from nab_sentry.rules.config import Rule
from nab_sentry.rules.engine import Firing, Obs, RuleEngine
from nab_sentry.rules.geometry import anchor
from nab_sentry.store.db import NewAlert
from nab_sentry.track import TrackSink
from nab_sentry.track.bytetrack import TrackRecord

log = get_logger("ingest")

VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".mpg", ".mpeg", ".ts", ".mts"}
)

# Named points at which the optional ``checkpoint`` hook is called (tests inject failures here).
CHECKPOINTS = (
    "video_row",         # after the videos row is inserted
    "playback_start",    # after the playback job has started
    "frame",             # before each sampled frame is gated (decode-error injection)
    "thumbnail",         # before each thumbnail write
    "embed",             # before each embedding batch
    "playback_wait",     # before waiting for the playback job
    "finalize",          # before the videos row is finalized
    "index_add",         # before index.add
    "index_save",        # before index.save
    "after_index_save",  # after index.save, before COMMIT
    "commit",            # immediately before COMMIT
)

Checkpoint = Callable[[str, Path], None]

# Exceptions that run the compensating rollback. Anything else (a simulated crash) skips it.
_COMPENSATED = (Exception, KeyboardInterrupt)

_THUMB_RE = re.compile(r"^v(\d+)_f(\d{7,})\.jpg$")
_PLAYBACK_RE = re.compile(r"^v(\d+)\.mp4(\.part)?$")


def thumb_name(video_id: int, frame_idx: int) -> str:
    """Thumbnail file name under ``data/thumbs/``, unique per ``frames`` row (6.4)."""
    return f"v{int(video_id)}_f{int(frame_idx):07d}.jpg"


def playback_name(video_id: int) -> str:
    """Playback_File name under ``data/playback/``."""
    return f"v{int(video_id)}.mp4"


def build_new_frame(
    video_id: int,
    start_time: datetime,
    frame_idx: int,
    offset_s: float,
    decision: GateDecision,
    thumb_path: str,
) -> NewFrame:
    """``NewFrame`` with ``abs_time = start_time + offset_s`` (2.3, 6.2).

    Arithmetic is done in UTC so a zone with DST transitions cannot shift the result; a naive
    start time is interpreted in the local zone (1.12).
    """
    start = to_aware_local(start_time)
    abs_utc = start.astimezone(timezone.utc) + timedelta(seconds=float(offset_s))
    return NewFrame(
        video_id=int(video_id),
        frame_idx=int(frame_idx),
        offset_s=float(offset_s),
        abs_time=abs_utc.astimezone(start.tzinfo),
        gate_reason=decision.reason,  # type: ignore[arg-type]  # passed -> first/motion/keyframe
        motion_frac=float(decision.fraction),
        thumb_path=thumb_path,
    )


def expand_paths(paths: Iterable[Path | str]) -> list[Path]:
    """Directories expand (non-recursively) to their video files; files pass through.

    Result is sorted and free of duplicate paths.
    """
    out: set[Path] = set()
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.update(
                c for c in p.iterdir() if c.is_file() and c.suffix.lower() in VIDEO_EXTENSIONS
            )
        else:
            out.add(p)
    return sorted(out, key=lambda q: str(q))


# ---------------------------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------------------------

Status = Literal["ingested", "already_indexed", "unresolved_metadata", "unreadable", "failed"]


@dataclass(frozen=True)
class VideoResult:
    path: Path
    status: Status
    video_id: int | None = None
    reason: str | None = None
    sampled: int = 0
    passed: int = 0
    vectors: int = 0


@dataclass(frozen=True)
class IngestReport:
    results: list[VideoResult]

    def _count(self, *statuses: str) -> int:
        return sum(1 for r in self.results if r.status in statuses)

    @property
    def ingested(self) -> int:
        return self._count("ingested")

    @property
    def already_indexed(self) -> int:
        return self._count("already_indexed")

    @property
    def failed(self) -> int:
        """Unresolved + unreadable + failed."""
        return self._count("unresolved_metadata", "unreadable", "failed")

    @property
    def vectors(self) -> int:
        """Vectors added by this run."""
        return sum(r.vectors for r in self.results if r.status == "ingested")


@dataclass(frozen=True)
class ReconcileReport:
    removed_index_ids: list[int] = field(default_factory=list)
    deleted_video_ids: list[int] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.removed_index_ids or self.deleted_video_ids or self.deleted_files)


# ---------------------------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------------------------


@dataclass
class _Pending:
    """One image queued for embedding."""

    frame_id: int
    image: np.ndarray
    detection: Detection | None  # None -> kind 'frame'
    box: tuple[int, int, int, int] | None = None
    track_key: int | None = None  # tracker key of the detection (V2), linked after end_video


@dataclass
class _VideoState:
    """What has been written for the current video, for compensation."""

    video_id: int | None = None
    job: Any = None
    staged_ids: list[int] = field(default_factory=list)
    staged_vecs: list[np.ndarray] = field(default_factory=list)
    written_files: list[Path] = field(default_factory=list)
    index_touched: bool = False
    index_saved: bool = False
    job_killed: bool = False
    tracking: bool = False  # TrackSink.begin_video ran for this video (rollback must undo it)
    camera_id: str | None = None
    track_ids_before: dict[int, int] | None = None  # camera's key -> track_id map at the start
    crop_keys: list[tuple[int, int]] = field(default_factory=list)  # (vector_id, track key)
    start_t: float = 0.0  # epoch seconds of the video's first frame
    engine: RuleEngine | None = None  # this camera's rules (V2 Requirement 12)
    engine_snap: object = None
    firings: list[_PendingAlert] = field(default_factory=list)


@dataclass
class _PendingAlert:
    """A rule that fired during the current video; written with the video's track rows."""

    rule: Rule
    firing: Firing
    offset_s: float
    det_class: str | None
    snapshot: str


class IngestPipeline:
    def __init__(
        self,
        cfg: Config,
        db: MetadataStore,
        index: VectorIndex,
        encoder: Encoder,
        detector: Detector | None,
        open_source: Callable[[Path], VideoSource] = FileSource.open,
        start_playback: Callable[[Path, Path], PlaybackJob] | None = None,
        *,
        checkpoint: Checkpoint | None = None,
        tracker: TrackSink | None = None,
        rules: Sequence[Rule] = (),
        rois: dict[str, tuple[float, float, float, float]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.db = db
        self.index = index
        self.encoder = encoder
        self.detector = detector
        self.tracker = tracker  # None: exactly V1 (sample rate only, no extra detector runs)
        # camera_id -> {tracker key -> tracks.track_id} for tracks already in the store; tracks
        # continuing into the next Segment extend their row instead of starting a new one.
        self._track_ids: dict[str, dict[int, int]] = {}
        # Watch rules need tracks: without a tracker they are ignored (logged once per camera).
        self.rules = list(rules)
        # camera_id -> motion ROI from the Camera_Config (V2 Requirement 3.1); none = whole frame
        self.rois = dict(rois or {})
        self._engines: dict[str, RuleEngine] = {}
        self.open_source = open_source
        self.start_playback = start_playback or (
            lambda src, dst: default_playback(src, dst, cfg.playback_max_width)
        )
        self.checkpoint = checkpoint
        self._seen_hashes: set[str] = set()

    # -- helpers -------------------------------------------------------------------------------

    def _check(self, step: str, path: Path) -> None:
        if self.checkpoint is not None:
            self.checkpoint(step, path)

    @property
    def _active_detector(self) -> Detector | None:
        return self.detector if self.cfg.enable_detector else None

    # -- run level -----------------------------------------------------------------------------

    def ingest_paths(self, paths: Iterable[Path | str]) -> IngestReport:
        """Validate Config, then ingest every video (sorted, de-duplicated by hash in this run).

        ``ConfigError`` is raised before any row, file, or index entry is written (2.8, 3.8).
        A per-video failure never aborts the run (6.8, 7.7).
        """
        self.cfg.require_valid()
        files = expand_paths(paths)
        self._seen_hashes = set()
        results = [self.ingest_video(p) for p in files]
        return IngestReport(results)

    # -- per video -----------------------------------------------------------------------------

    def ingest_video(self, path: Path | str) -> VideoResult:
        path = Path(path)

        # 1. Src_Hash before any write (7.1); unreadable file -> skip (7.8).
        try:
            h = src_hash(path)
        except OSError as exc:
            log.error("cannot read video to compute Src_Hash: %s: %s", path, exc)
            return VideoResult(path, "failed", reason=f"cannot read file: {exc}")
        if h in self._seen_hashes or self.db.hash_exists(h):
            log.info("already indexed: %s", path)
            return VideoResult(path, "already_indexed", reason="already indexed")

        # 2. Camera metadata (resolve_metadata logs the unresolved case itself, 1.5).
        try:
            meta = resolve_metadata(path, log)
        except Exception as exc:  # unexpected I/O problem reading the sidecar
            log.error("ingest failed: %s: %s", path, exc)
            return VideoResult(path, "failed", reason=str(exc))
        if meta is None:
            return VideoResult(path, "unresolved_metadata", reason="unresolved camera metadata")

        # 3. Open the source (1.11, 2.9, 2.10).
        try:
            source = self.open_source(path)
        except UnknownFrameRate:
            log.error("unknown frame rate: %s", path)
            return VideoResult(path, "unreadable", reason="unknown frame rate")
        except UnreadableVideo as exc:
            log.error("unreadable video: %s (%s)", path, exc)
            return VideoResult(path, "unreadable", reason="unreadable video")
        except Exception as exc:
            log.error("unreadable video: %s (%s)", path, exc)
            return VideoResult(path, "unreadable", reason=f"unreadable video: {exc}")

        try:
            return self._ingest_open(path, h, meta, source)
        finally:
            try:
                source.close()
            except Exception:  # closing must never hide the real outcome
                log.debug("source close failed: %s", path, exc_info=True)

    def _ingest_open(self, path: Path, h: str, meta: Any, source: VideoSource) -> VideoResult:
        st = _VideoState()
        counts: dict[str, int] = {"sampled": 0, "passed": 0}
        try:
            with self.db.transaction():  # 4. BEGIN IMMEDIATE ... 9. COMMIT
                try:
                    self._steps(path, h, meta, source, st, counts)
                except _COMPENSATED:
                    self._kill_job(st)  # rollback step 1: kill ffmpeg (before ROLLBACK)
                    raise
        except _COMPENSATED as exc:
            # rollback step 2 (ROLLBACK) already ran when the `with` exited; a failed COMMIT
            # leaves the transaction open, so roll back explicitly too.
            self._rollback(path, exc, st)
            if isinstance(exc, KeyboardInterrupt):
                raise
            if isinstance(exc, UnreadableVideo):
                return VideoResult(path, "unreadable", video_id=None, reason="unreadable video")
            return VideoResult(path, "failed", video_id=None, reason=_reason(exc))

        self._seen_hashes.add(h)
        log.info(
            "ingested: %s (video_id=%s, sampled=%d, passed=%d, vectors=%d)",
            path, st.video_id, counts["sampled"], counts["passed"], len(st.staged_ids),
        )
        return VideoResult(
            path, "ingested", video_id=st.video_id, sampled=counts["sampled"],
            passed=counts["passed"], vectors=len(st.staged_ids),
        )

    def _steps(
        self, path: Path, h: str, meta: Any, source: VideoSource, st: _VideoState,
        counts: dict[str, int],
    ) -> None:
        cfg = self.cfg
        fps = float(source.fps)

        # 4. camera + videos row (1.9, 1.10, 6.9), playback job started concurrently (6.5).
        self.db.upsert_camera(meta.camera_id, meta.label)
        st.video_id = video_id = self.db.insert_video(
            NewVideo(
                camera_id=meta.camera_id,
                src_path=str(path),
                src_hash=h,
                start_ts=meta.start_time,
                fps=fps,
                width=int(source.width),
                height=int(source.height),
                est_duration_s=max(0.0, float(source.est_frame_count) / fps),
            )
        )
        self._check("video_row", path)

        cfg.thumbs_dir.mkdir(parents=True, exist_ok=True)
        cfg.playback_dir.mkdir(parents=True, exist_ok=True)
        dst = cfg.playback_dir / playback_name(video_id)
        st.written_files += [part_path(dst), dst]  # recorded before anything is written
        st.job = self.start_playback(path, dst)
        self._check("playback_start", path)

        # 5. sample, gate, frames rows, thumbnails, detector, batched embedding.
        #    With a TrackSink (V2 9.2) frames are also taken at the tracking rate; the gate
        #    and everything V1 stores still use only the sample-rate frames (``on_sample``).
        detector = self._active_detector
        tracker = self.tracker if detector is not None else None
        sampler: Sampler | DualSampler
        if tracker is not None:
            sampler = DualSampler(cfg.sample_rate, effective_track_rate(cfg))
            selected: Iterable[tuple[Any, bool]] = sampler.select(source.frames())
        else:
            sampler = Sampler(cfg.sample_rate)
            selected = ((f, True) for f in sampler.select(source.frames()))
        gate = MotionGate(
            cfg.motion_threshold, cfg.keyframe_interval_s, cfg.gate_width,
            cfg.motion_pixel_delta, cfg.motion_method,  # type: ignore[arg-type]
            roi=self.rois.get(meta.camera_id),
        )
        gate.reset()
        if tracker is not None:
            st.camera_id = meta.camera_id
            st.track_ids_before = dict(self._track_ids.get(meta.camera_id, {}))
            st.start_t = to_aware_local(meta.start_time).timestamp()
            tracker.begin_video(meta.camera_id, st.start_t, video_id)
            st.tracking = True
            st.engine = self._engine(meta.camera_id)
            if st.engine is not None:
                st.engine_snap = st.engine.snapshot()
        queue: list[_Pending] = []
        batch = int(cfg.batch_size)
        passed = 0
        last_motion = False  # the last sample-rate frame showed motion (fraction >= threshold)

        for frame, on_sample in selected:
            self._check("frame", path)
            if not on_sample:
                # Tracking-rate frame between sample-rate frames: detect only while the
                # scene is active - the last sample-rate frame showed motion (its changed
                # fraction, whatever the gate's label: a keyframe can carry motion too), or
                # a track is alive. A still keyframe or the first frame starts nothing.
                if tracker is not None and (last_motion or tracker.active):
                    self._track_frame(tracker, frame, self._run_detector(
                        detector, path, frame.offset_s, frame.image), st)
                continue
            decision = gate.evaluate(frame)
            last_motion = decision.passed and decision.reason != "first" and (
                decision.fraction >= cfg.motion_threshold)
            if not decision.passed:
                # static / empty: no row, no thumbnail, no embedding (3.7, 3.9); a live track
                # keeps the detector running so a person standing still is not lost.
                if tracker is not None and tracker.active and frame.image.size:
                    self._track_frame(tracker, frame, self._run_detector(
                        detector, path, frame.offset_s, frame.image), st)
                continue
            passed += 1
            image = frame.image
            name = thumb_name(video_id, frame.index)
            frame_id = self.db.insert_frame(
                build_new_frame(video_id, meta.start_time, frame.index, frame.offset_s,
                                decision, name)
            )
            thumb = cfg.thumbs_dir / name
            st.written_files.append(thumb)
            self._check("thumbnail", path)
            write_thumbnail(image, thumb, cfg.thumb_width)

            queue.append(_Pending(frame_id, image, None))
            dets = self._run_detector(detector, path, frame.offset_s, image)
            keys: dict[int, int | None] = {}
            if tracker is not None:
                keys = {id(d): a.track_key for d, a in zip(dets, self._track_frame(tracker, frame, dets, st))}
            for det, box in self._crops(dets, image):
                x1, y1, x2, y2 = box
                queue.append(_Pending(frame_id, image[y1:y2, x1:x2], det, box, keys.get(id(det))))
            while len(queue) >= batch:
                self._flush(path, queue[:batch], st)
                del queue[:batch]

        # 6. remainder; zero passed frames (incl. zero decoded) -> unreadable (1.11, 2.10).
        if queue:
            self._flush(path, queue, st)
            queue.clear()
        counts["sampled"] = sampler.selected_count
        counts["passed"] = passed
        if passed == 0:
            raise UnreadableVideo(f"no frames passed the motion gate: {path}")
        if tracker is not None:
            # Inside the transaction: a failure later on rolls the rows and the tracker back.
            records = tracker.end_video(to_aware_local(meta.start_time).timestamp()
                                        + float(source.decoded_index_count) / fps)
            self._store_tracks(meta.camera_id, records, st)

        # 7. playback file, then finalize the videos row (3.6, 6.5).
        self._check("playback_wait", path)
        st.job.wait()
        final = Path(st.job.finalize())
        self._check("finalize", path)
        self.db.finalize_video(
            video_id,
            sampled=sampler.selected_count,
            passed=passed,
            duration_s=float(source.decoded_index_count) / fps,
            playback_path=final.name,
        )

        # 8. FAISS add + atomic save before COMMIT (key decision 1, 8.10).
        self._check("index_add", path)
        if st.staged_ids:
            st.index_touched = True
            self.index.add(np.asarray(st.staged_ids, dtype=np.int64), np.stack(st.staged_vecs))
        self._check("index_save", path)
        st.index_saved = True
        self.index.save(cfg.index_path)
        self._check("after_index_save", path)
        self._check("commit", path)
        # 9. COMMIT happens when the caller's `with db.transaction()` block exits.

    def _detect(
        self, detector: Detector | None, path: Path, offset_s: float, image: np.ndarray
    ) -> list[tuple[Detection, tuple[int, int, int, int]]]:
        """High detections with clamped boxes; inference errors -> warning, none (4.7, 4.8, 5.10)."""
        return self._crops(self._run_detector(detector, path, offset_s, image), image)

    @staticmethod
    def _run_detector(
        detector: Detector | None, path: Path, offset_s: float, image: np.ndarray
    ) -> list[Detection]:
        """All detections (high and low); inference errors -> warning, none (4.7, 4.8)."""
        if detector is None:
            return []
        try:
            return list(detector.detect(image))
        except Exception as exc:
            log.warning(
                "detector failed: %s at offset %.3f s: %s; storing frame with zero detections",
                path, offset_s, exc,
            )
            return []

    @staticmethod
    def _crops(
        dets: list[Detection], image: np.ndarray
    ) -> list[tuple[Detection, tuple[int, int, int, int]]]:
        """High detections with clamped boxes (5.10).

        Low detections (``Detection.high`` false, V2 Requirement 9.1) are dropped here: only
        high ones get crops and embeddings, so search results equal V1's.
        """
        h, w = image.shape[:2]
        out: list[tuple[Detection, tuple[int, int, int, int]]] = []
        for det in dets:
            if not det.high:
                continue
            box = clamp_box(*det.box, w, h)
            if box is None:
                continue  # crop smaller than 1 px: no crop embedding (5.10)
            out.append((det, box))
        return out

    def _flush(self, path: Path, items: list[_Pending], st: _VideoState) -> None:
        """Embed one batch; insert a ``vectors`` row per item and stage its vector (5.1, 5.2, 6.3)."""
        self._check("embed", path)
        vecs = np.asarray(self.encoder.encode_images([it.image for it in items]))
        dim = self.index.dim
        if vecs.ndim != 2 or vecs.shape != (len(items), dim):
            raise EmbeddingError(
                f"encoder returned shape {vecs.shape}, expected ({len(items)}, {dim})"
            )
        if not np.all(np.isfinite(vecs)):
            raise EmbeddingError("encoder returned non-finite values")
        vecs = vecs.astype(np.float32, copy=False)
        for it, vec in zip(items, vecs):
            if it.detection is None:
                nv = NewVector(frame_id=it.frame_id, kind="frame")
            else:
                nv = NewVector(
                    frame_id=it.frame_id, kind="crop", det_class=it.detection.cls,
                    det_conf=float(it.detection.conf), box=it.box,
                )
            vector_id = self.db.insert_vector(nv)
            st.staged_ids.append(vector_id)
            st.staged_vecs.append(vec)
            if it.track_key is not None:
                st.crop_keys.append((vector_id, it.track_key))

    def _store_tracks(self, camera_id: str, records: Sequence[TrackRecord], st: _VideoState) -> None:
        """Write drained track records (V2 Requirement 10.7): a new ``tracks`` row per new
        confirmed track, ``extend_track`` for one continuing from an earlier Segment, one
        ``track_obs`` row per observation (tagged with its own video, which for back-filled
        observations can be the previous Segment), then ``vectors.track_id`` for this video's
        crops of confirmed tracks. A crop of a track confirmed only in a later Segment stays
        unlinked."""
        ids = self._track_ids.setdefault(camera_id, {})
        for rec in records:
            obs = [NewTrackObs(o.tag[0], o.tag[1], round(o.t * 1000), o.conf, o.high, o.box)
                   for o in rec.observations if isinstance(o.tag, tuple) and o.tag[0] is not None]
            if obs:
                last_ms = max(o.abs_epoch_ms for o in obs)
                if rec.key in ids:
                    self.db.extend_track(ids[rec.key], last_ms, len(obs))
                else:
                    ids[rec.key] = self.db.insert_track(
                        camera_id, rec.cls, min(o.abs_epoch_ms for o in obs), last_ms, len(obs))
                self.db.insert_track_obs(ids[rec.key], obs)
        links = [(v, ids[k]) for v, k in st.crop_keys if k in ids]
        if links:
            self.db.link_vectors(links)
        if st.firings:
            mutes = self.db.active_mutes(0)
            for pa in st.firings:
                ms = round(pa.firing.t * 1000)
                muted = mutes.get(pa.rule.id, 0) > ms
                self.db.insert_alert(NewAlert(
                    kind="rule", trigger_epoch_ms=ms, state="suppressed" if muted else "pending",
                    rule_id=pa.rule.id, rule_name=pa.rule.name, severity=pa.rule.severity,
                    camera_id=camera_id, track_id=ids.get(pa.firing.track_key) if pa.firing.track_key is not None else None,
                    video_id=st.video_id, trigger_offset_s=pa.offset_s, det_class=pa.det_class,
                    n_tracks=pa.firing.n_tracks, snapshot_path=pa.snapshot))
                log.warning("alert: %s on %s at %.1f s (%s)%s", pa.rule.id, camera_id, pa.offset_s,
                            pa.det_class, " [muted]" if muted else "")
        for rec in records:
            if rec.ended:
                ids.pop(rec.key, None)  # no more observations will come for it

    def _engine(self, camera_id: str) -> RuleEngine | None:
        """This camera's RuleEngine (created once, cooldowns seeded from stored alerts)."""
        if camera_id not in self._engines:
            rules = [r for r in self.rules if r.camera == camera_id]
            if not rules:
                return None
            eng = RuleEngine(rules)
            for rid, ms in self.db.last_rule_fires(camera_id).items():
                eng.set_last_fire(rid, ms / 1000)
            self._engines[camera_id] = eng
        return self._engines[camera_id]

    def _track_frame(self, tracker: TrackSink, frame: Any, dets: list[Detection],
                     st: _VideoState) -> list[Any]:
        """Feed the tracker, then the camera's rules; a firing writes its snapshot now (the frame
        is in memory) and is stored with the video's track rows."""
        res = list(tracker.update(frame, dets) or ())
        eng = st.engine
        if eng is None or frame.image is None or not frame.image.size:
            return res
        h, w = frame.image.shape[:2]
        obs, by_key = [], {}
        for a in res:
            if getattr(a, "track_key", None) is None:
                continue
            obs.append(Obs(a.track_key, a.detection.cls, anchor(a.detection.box, w, h), a.confirmed))
            by_key[a.track_key] = a.detection
        t = st.start_t + float(frame.offset_s)
        for f in eng.step(t, datetime.fromtimestamp(t), obs):
            rule = next(r for r in eng.rules if r.id == f.rule_id)
            det = by_key.get(f.track_key)
            name = snapshot_name(st.video_id or 0, rule.id, round(t * 1000))
            dst = self.cfg.alerts_dir / name
            st.written_files.append(dst)
            self.cfg.alerts_dir.mkdir(parents=True, exist_ok=True)
            when = datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
            img = draw_snapshot(frame.image, det.box if det else None, f"{rule.name} | {when}",
                                zone=rule.zone_poly, line=rule.line_pts)
            if not cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                raise OSError(f"could not write alert snapshot {dst}")
            st.firings.append(_PendingAlert(rule, f, float(frame.offset_s),
                                            det.cls if det else None, name))
        return res

    # -- compensation --------------------------------------------------------------------------

    def _kill_job(self, st: _VideoState) -> None:
        if st.job is None or st.job_killed:
            return
        st.job_killed = True
        try:
            st.job.kill()
        except Exception:
            log.warning("could not stop playback job", exc_info=True)

    def _rollback(self, path: Path, exc: BaseException, st: _VideoState) -> None:
        """kill ffmpeg -> ROLLBACK -> index remove (+ save) -> delete files -> log (6.8, 7.7);
        the TrackSink is rolled back to its state before this video (V2 Requirement 10.6)."""
        self._kill_job(st)
        if st.tracking and self.tracker is not None:
            try:
                self.tracker.rollback_video()
            except Exception:
                log.error("track rollback failed for %s", path, exc_info=True)
            if st.camera_id is not None and st.track_ids_before is not None:
                self._track_ids[st.camera_id] = st.track_ids_before
            if st.engine is not None and st.engine_snap is not None:
                st.engine.restore(st.engine_snap)
        try:
            self.db.rollback()  # no-op unless a failed COMMIT left the transaction open
        except Exception:
            log.error("SQLite ROLLBACK failed for %s", path, exc_info=True)

        if st.index_touched or st.index_saved:
            try:
                if st.staged_ids:
                    self.index.remove(np.asarray(st.staged_ids, dtype=np.int64))
                if st.index_saved:
                    self.index.save(self.cfg.index_path)
            except Exception:
                # Orphan IDs left behind are removed by the next reconcile() (8.15).
                log.error("index compensation failed for %s; run reconcile", path, exc_info=True)

        for f in st.written_files:
            try:
                Path(f).unlink(missing_ok=True)
            except OSError as e:
                log.error("could not delete %s during rollback: %s", f, e)

        if isinstance(exc, UnreadableVideo):
            log.error("unreadable video: %s: %s; rolled back", path, exc)
        elif isinstance(exc, KeyboardInterrupt):
            log.error("ingest failed: %s: interrupted; rolled back", path)
        else:
            log.error("ingest failed: %s: %s", path, _reason(exc))

    # -- reconciliation ------------------------------------------------------------------------

    def reconcile(self) -> ReconcileReport:
        """Make the DB, the index, and the media directories agree (key decision 2).

        1. Remove FAISS IDs that have no ``vectors`` row.
        2. Delete ``videos`` rows whose status is not ``complete`` and their files.
        3. Delete complete videos with any vector missing from FAISS (and their files and
           remaining index entries), so the next run re-ingests them.
        4. Delete thumbnail/playback files (by our naming pattern) no row references.
        The index is saved if it changed.
        """
        cfg = self.cfg
        removed: list[int] = []
        deleted_videos: list[int] = []
        deleted_files: list[str] = []
        index_changed = False

        db_ids = set(self.db.vector_ids().tolist())
        orphans = [i for i in self.index.ids().tolist() if i not in db_ids]
        if orphans:
            self.index.remove(np.asarray(orphans, dtype=np.int64))
            removed.extend(sorted(orphans))
            index_changed = True

        for vid in self.db.video_ids(status="ingesting"):
            deleted_files += self._delete_video_files(vid)
            deleted_videos.append(vid)

        idx_ids = set(self.index.ids().tolist())
        for vid in self.db.video_ids(status="complete"):
            vids = self.db.vector_ids_for_video(vid).tolist()
            if all(i in idx_ids for i in vids):
                continue
            present = [i for i in vids if i in idx_ids]
            if present:
                self.index.remove(np.asarray(present, dtype=np.int64))
                removed.extend(present)
                index_changed = True
            deleted_files += self._delete_video_files(vid)
            deleted_videos.append(vid)

        referenced = self.db.referenced_files() | self.db.referenced_alert_files()
        for directory, pattern in ((cfg.thumbs_dir, _THUMB_RE), (cfg.playback_dir, _PLAYBACK_RE),
                                   (cfg.alerts_dir, ALERT_FILE_RE)):
            if not directory.is_dir():
                continue
            for f in sorted(directory.iterdir()):
                if f.is_file() and pattern.match(f.name) and f.name not in referenced:
                    if _unlink(f):
                        deleted_files.append(f.name)

        empty_tracks = self.db.delete_empty_tracks()  # their videos were deleted (V2 11.3)
        if empty_tracks:
            log.warning("reconcile: deleted %d tracks without observations", empty_tracks)

        if index_changed:
            self.index.save(cfg.index_path)
        report = ReconcileReport(removed, deleted_videos, deleted_files)
        if report.changed:
            log.warning(
                "reconcile: removed %d orphan index IDs, deleted %d videos, deleted %d files",
                len(removed), len(deleted_videos), len(deleted_files),
            )
        return report

    def _delete_video_files(self, video_id: int) -> list[str]:
        names = self.db.delete_video(video_id)
        pb = playback_name(video_id)
        names += [n for n in (pb, pb + ".part") if n not in names]
        out: list[str] = []
        for n in names:
            base = self.cfg.playback_dir if _PLAYBACK_RE.match(n) else self.cfg.thumbs_dir
            if _unlink(base / n):
                out.append(n)
        return out


def _unlink(p: Path) -> bool:
    """Delete ``p`` if present; True if a file was removed."""
    try:
        p.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as e:
        log.error("could not delete %s: %s", p, e)
        return False


def _reason(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
