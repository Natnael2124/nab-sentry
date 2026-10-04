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
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from nab_sentry.config import Config
from nab_sentry.embed.clip_encoder import EmbeddingError, Encoder
from nab_sentry.ingest.detector import Detection, Detector, clamp_box
from nab_sentry.ingest.metadata import resolve_metadata, src_hash, to_aware_local
from nab_sentry.ingest.motion import GateDecision, MotionGate
from nab_sentry.ingest.sampler import Sampler
from nab_sentry.ingest.sources import (
    FileSource,
    UnknownFrameRate,
    UnreadableVideo,
    VideoSource,
)
from nab_sentry.ingest.transcode import PlaybackJob, default_playback, part_path, write_thumbnail
from nab_sentry.logging_setup import get_logger
from nab_sentry.store.db import MetadataStore, NewFrame, NewVector, NewVideo
from nab_sentry.store.vector_index import VectorIndex

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
    ) -> None:
        self.cfg = cfg
        self.db = db
        self.index = index
        self.encoder = encoder
        self.detector = detector
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
        sampler = Sampler(cfg.sample_rate)
        gate = MotionGate(
            cfg.motion_threshold, cfg.keyframe_interval_s, cfg.gate_width,
            cfg.motion_pixel_delta, cfg.motion_method,  # type: ignore[arg-type]
        )
        gate.reset()
        detector = self._active_detector
        queue: list[_Pending] = []
        batch = int(cfg.batch_size)
        passed = 0

        for frame in sampler.select(source.frames()):
            self._check("frame", path)
            decision = gate.evaluate(frame)
            if not decision.passed:
                continue  # static / empty: no row, no thumbnail, no embedding (3.7, 3.9)
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
            for det, box in self._detect(detector, path, frame.offset_s, image):
                x1, y1, x2, y2 = box
                queue.append(_Pending(frame_id, image[y1:y2, x1:x2], det, box))
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
        """Kept detections with clamped boxes; inference errors -> warning, none (4.7, 4.8, 5.10)."""
        if detector is None:
            return []
        try:
            dets = list(detector.detect(image))
        except Exception as exc:
            log.warning(
                "detector failed: %s at offset %.3f s: %s; storing frame with zero detections",
                path, offset_s, exc,
            )
            return []
        h, w = image.shape[:2]
        out: list[tuple[Detection, tuple[int, int, int, int]]] = []
        for det in dets:
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
            st.staged_ids.append(self.db.insert_vector(nv))
            st.staged_vecs.append(vec)

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
        """kill ffmpeg -> ROLLBACK -> index remove (+ save) -> delete files -> log (6.8, 7.7)."""
        self._kill_job(st)
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

        referenced = self.db.referenced_files()
        for directory, pattern in ((cfg.thumbs_dir, _THUMB_RE), (cfg.playback_dir, _PLAYBACK_RE)):
            if not directory.is_dir():
                continue
            for f in sorted(directory.iterdir()):
                if f.is_file() and pattern.match(f.name) and f.name not in referenced:
                    if _unlink(f):
                        deleted_files.append(f.name)

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
