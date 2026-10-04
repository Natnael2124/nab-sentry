"""SQLite Metadata_Store: cameras, videos, frames, and vectors.

Requirements 1.9, 1.10, 6.1-6.3, 6.9, 8.3, 8.8, 8.9, 8.13, 18.4.

Every statement uses ``?`` or named placeholders; no caller-supplied value is ever formatted into
SQL text (18.4). The connection runs in autocommit mode (``isolation_level=None``) and explicit
transactions use ``BEGIN IMMEDIATE`` via :meth:`MetadataStore.transaction`. All access goes through
one re-entrant lock so a single store can be shared between threads (the API server).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:  # avoid an import cycle once engine.py imports MetadataStore (task 11.8)
    from nab_sentry.search.engine import SearchFilter

log = logging.getLogger(__name__)

SCHEMA_VERSION = "1"
EMBED_MODEL = "ViT-B-32/laion2b_s34b_b79k"
EMBED_DIM = "512"

DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cameras (
    camera_id  TEXT PRIMARY KEY CHECK (length(camera_id) BETWEEN 1 AND 64),
    label      TEXT NOT NULL CHECK (length(label) BETWEEN 1 AND 128),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
    video_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id      TEXT    NOT NULL REFERENCES cameras(camera_id),
    src_path       TEXT    NOT NULL,
    src_hash       TEXT    NOT NULL UNIQUE,
    start_ts       TEXT    NOT NULL,
    start_epoch_ms INTEGER NOT NULL,
    fps            REAL    NOT NULL CHECK (fps > 0),
    width          INTEGER NOT NULL,
    height         INTEGER NOT NULL,
    duration_s     REAL    NOT NULL CHECK (duration_s >= 0),
    sampled_count  INTEGER NOT NULL DEFAULT 0,
    passed_count   INTEGER NOT NULL DEFAULT 0 CHECK (passed_count <= sampled_count),
    playback_path  TEXT,
    status         TEXT    NOT NULL CHECK (status IN ('ingesting', 'complete')),
    ingested_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS frames (
    frame_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id     INTEGER NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    frame_idx    INTEGER NOT NULL CHECK (frame_idx >= 0),
    offset_s     REAL    NOT NULL CHECK (offset_s >= 0),
    abs_ts       TEXT    NOT NULL,
    abs_epoch_ms INTEGER NOT NULL,
    gate_reason  TEXT    NOT NULL CHECK (gate_reason IN ('first', 'motion', 'keyframe')),
    motion_frac  REAL    NOT NULL CHECK (motion_frac BETWEEN 0 AND 1),
    thumb_path   TEXT    NOT NULL,
    UNIQUE (video_id, frame_idx)
);
CREATE INDEX IF NOT EXISTS idx_frames_video_offset ON frames(video_id, offset_s);
CREATE INDEX IF NOT EXISTS idx_frames_epoch        ON frames(abs_epoch_ms);

CREATE TABLE IF NOT EXISTS vectors (
    vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id  INTEGER NOT NULL REFERENCES frames(frame_id) ON DELETE CASCADE,
    kind      TEXT    NOT NULL CHECK (kind IN ('frame', 'crop')),
    det_class TEXT    CHECK (det_class IN ('person','car','truck','bus','motorcycle','bicycle')),
    det_conf  REAL    CHECK (det_conf BETWEEN 0 AND 1),
    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
    CHECK (
        (kind = 'frame' AND det_class IS NULL AND det_conf IS NULL
            AND x1 IS NULL AND y1 IS NULL AND x2 IS NULL AND y2 IS NULL)
     OR (kind = 'crop' AND det_class IS NOT NULL AND det_conf IS NOT NULL
            AND x1 >= 0 AND y1 >= 0 AND x1 < x2 AND y1 < y2)
    )
);
CREATE INDEX IF NOT EXISTS idx_vectors_frame      ON vectors(frame_id);
CREATE INDEX IF NOT EXISTS idx_vectors_kind_class ON vectors(kind, det_class);
"""

# Fixed template: the clause text never changes, only the bound values (8.3, 8.8, 8.9, 8.13, 18.4).
ALLOWED_IDS_SQL = """
SELECT v.vector_id FROM vectors v
JOIN frames f ON f.frame_id = v.frame_id
JOIN videos d ON d.video_id = f.video_id
WHERE d.status = 'complete'
  AND (:camera IS NULL OR d.camera_id = :camera)
  AND (:start_ms IS NULL OR f.abs_epoch_ms >= :start_ms)
  AND (:end_ms IS NULL OR f.abs_epoch_ms <= :end_ms)
  AND (:cls IS NULL OR (v.kind = 'crop' AND v.det_class = :cls))
ORDER BY v.vector_id
"""

HIT_ROWS_SQL = """
SELECT v.vector_id, f.frame_id, d.video_id, d.camera_id, c.label, f.offset_s, f.thumb_path,
       d.duration_s, d.start_ts, d.start_epoch_ms
FROM temp.ids t
JOIN vectors v ON v.vector_id = t.id
JOIN frames  f ON f.frame_id  = v.frame_id
JOIN videos  d ON d.video_id  = f.video_id
JOIN cameras c ON c.camera_id = d.camera_id
"""

UpsertResult = Literal["created", "exists", "label_conflict"]
VectorKind = Literal["frame", "crop"]
GateReason = Literal["first", "motion", "keyframe"]


# ---------------------------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NewVideo:
    camera_id: str
    src_path: str
    src_hash: str
    start_ts: datetime  # aware; a naive value is taken as local time (1.12)
    fps: float
    width: int
    height: int
    est_duration_s: float


@dataclass(frozen=True)
class NewFrame:
    video_id: int
    frame_idx: int
    offset_s: float
    abs_time: datetime  # aware
    gate_reason: GateReason
    motion_frac: float
    thumb_path: str


@dataclass(frozen=True)
class NewVector:
    frame_id: int
    kind: VectorKind
    det_class: str | None = None
    det_conf: float | None = None
    box: tuple[int, int, int, int] | None = None  # (x1, y1, x2, y2) in source-frame pixels


@dataclass(frozen=True)
class HitRow:
    vector_id: int
    frame_id: int
    video_id: int
    camera_id: str
    camera_label: str
    offset_s: float
    thumb_path: str
    duration_s: float
    start_ts: datetime
    start_epoch_ms: int


@dataclass(frozen=True)
class Counts:
    cameras: int
    videos: int  # complete videos only
    frames: int
    vectors: int


# ---------------------------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------------------------


def as_aware(dt: datetime) -> datetime:
    """Return ``dt`` unchanged if aware, otherwise interpret it in the local time zone (1.12)."""
    return dt if dt.tzinfo is not None and dt.utcoffset() is not None else dt.astimezone()


def epoch_ms(dt: datetime) -> int:
    """Integer epoch milliseconds, matching the ``abs_epoch_ms`` rule in the design."""
    return round(as_aware(dt).timestamp() * 1000)


def iso_ms(dt: datetime) -> str:
    """ISO 8601 with millisecond precision and UTC offset."""
    return as_aware(dt).isoformat(timespec="milliseconds")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------------------------


class MetadataStore:
    def __init__(self, path: Path | str, *, busy_timeout_ms: int = 5000) -> None:
        self.path = path
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")  # int-coerced constant
        self.init_schema()

    # -- lifecycle -----------------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> MetadataStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(DDL)
            self._conn.executemany(
                "INSERT OR IGNORE INTO schema_meta(key, value) VALUES (?, ?)",
                [("schema_version", SCHEMA_VERSION), ("embed_model", EMBED_MODEL), ("dim", EMBED_DIM)],
            )

    # -- transactions --------------------------------------------------------------------------

    @property
    def in_transaction(self) -> bool:
        return self._conn.in_transaction

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """``BEGIN IMMEDIATE`` ... ``COMMIT``; ``ROLLBACK`` on any exception (incl. KeyboardInterrupt)."""
        with self._lock:
            if self._conn.in_transaction:
                raise RuntimeError("nested MetadataStore.transaction() is not supported")
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def rollback(self) -> None:
        """``ROLLBACK`` if a transaction is still open (e.g. after a failed ``COMMIT``); else no-op."""
        with self._lock:
            if self._conn.in_transaction:
                self._conn.execute("ROLLBACK")

    @contextmanager
    def _write(self) -> Iterator[None]:
        """Join the caller's transaction if one is open, otherwise run in a new one."""
        with self._lock:
            if self._conn.in_transaction:
                yield
            else:
                with self.transaction():
                    yield

    # -- writes --------------------------------------------------------------------------------

    def upsert_camera(self, camera_id: str, label: str) -> UpsertResult:
        """Create the camera row once; never relabel an existing row (1.9, 1.10)."""
        with self._write():
            cur = self._conn.execute(
                "INSERT INTO cameras(camera_id, label, created_at) VALUES (?, ?, ?) "
                "ON CONFLICT(camera_id) DO NOTHING",
                (camera_id, label, _now_iso()),
            )
            if cur.rowcount == 1:
                return "created"
            (stored,) = self._conn.execute(
                "SELECT label FROM cameras WHERE camera_id = ?", (camera_id,)
            ).fetchone()
        if stored == label:
            return "exists"
        log.warning(
            "camera label conflict: camera_id=%r stored_label=%r new_label=%r (keeping stored)",
            camera_id, stored, label,
        )
        return "label_conflict"

    def camera_label(self, camera_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT label FROM cameras WHERE camera_id = ?", (camera_id,)
            ).fetchone()
        return None if row is None else row[0]

    def hash_exists(self, src_hash: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM videos WHERE src_hash = ? LIMIT 1", (src_hash,)
            ).fetchone()
        return row is not None

    def insert_video(self, v: NewVideo) -> int:
        """Insert a ``videos`` row with ``status='ingesting'`` (6.9)."""
        with self._write():
            cur = self._conn.execute(
                "INSERT INTO videos(camera_id, src_path, src_hash, start_ts, start_epoch_ms, fps, "
                "width, height, duration_s, status, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ingesting', ?)",
                (
                    v.camera_id, v.src_path, v.src_hash, iso_ms(v.start_ts), epoch_ms(v.start_ts),
                    float(v.fps), int(v.width), int(v.height), float(v.est_duration_s), _now_iso(),
                ),
            )
            return int(cur.lastrowid)

    def insert_frame(self, f: NewFrame) -> int:
        with self._write():
            cur = self._conn.execute(
                "INSERT INTO frames(video_id, frame_idx, offset_s, abs_ts, abs_epoch_ms, "
                "gate_reason, motion_frac, thumb_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    int(f.video_id), int(f.frame_idx), float(f.offset_s), iso_ms(f.abs_time),
                    epoch_ms(f.abs_time), f.gate_reason, float(f.motion_frac), f.thumb_path,
                ),
            )
            return int(cur.lastrowid)

    def insert_vector(self, v: NewVector) -> int:
        x1 = y1 = x2 = y2 = None
        if v.box is not None:
            x1, y1, x2, y2 = (int(c) for c in v.box)
        conf = None if v.det_conf is None else float(v.det_conf)
        with self._write():
            cur = self._conn.execute(
                "INSERT INTO vectors(frame_id, kind, det_class, det_conf, x1, y1, x2, y2) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (int(v.frame_id), v.kind, v.det_class, conf, x1, y1, x2, y2),
            )
            return int(cur.lastrowid)

    def finalize_video(
        self, video_id: int, *, sampled: int, passed: int, duration_s: float, playback_path: str
    ) -> None:
        with self._write():
            cur = self._conn.execute(
                "UPDATE videos SET sampled_count = ?, passed_count = ?, duration_s = ?, "
                "playback_path = ?, status = 'complete' WHERE video_id = ?",
                (int(sampled), int(passed), float(duration_s), playback_path, int(video_id)),
            )
            if cur.rowcount != 1:
                raise KeyError(f"no videos row with video_id={video_id}")

    def delete_video(self, video_id: int) -> list[str]:
        """Delete a video (frames and vectors cascade); return its thumbnail and playback file names."""
        with self._write():
            files = [
                r[0]
                for r in self._conn.execute(
                    "SELECT thumb_path FROM frames WHERE video_id = ? ORDER BY frame_id", (int(video_id),)
                )
            ]
            row = self._conn.execute(
                "SELECT playback_path FROM videos WHERE video_id = ?", (int(video_id),)
            ).fetchone()
            if row is None:
                return []
            if row[0]:
                files.append(row[0])
            self._conn.execute("DELETE FROM videos WHERE video_id = ?", (int(video_id),))
        return files

    # -- reads ---------------------------------------------------------------------------------

    def vector_ids(self) -> np.ndarray:
        with self._lock:
            rows = self._conn.execute("SELECT vector_id FROM vectors ORDER BY vector_id").fetchall()
        return np.array([r[0] for r in rows], dtype=np.int64)

    def video_ids(self, *, status: str | None = None) -> list[int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT video_id FROM videos WHERE (? IS NULL OR status = ?) ORDER BY video_id",
                (status, status),
            ).fetchall()
        return [r[0] for r in rows]

    def vector_ids_for_video(self, video_id: int) -> np.ndarray:
        with self._lock:
            rows = self._conn.execute(
                "SELECT v.vector_id FROM vectors v JOIN frames f ON f.frame_id = v.frame_id "
                "WHERE f.video_id = ? ORDER BY v.vector_id",
                (int(video_id),),
            ).fetchall()
        return np.array([r[0] for r in rows], dtype=np.int64)

    def referenced_files(self) -> set[str]:
        """Every thumbnail and playback file name referenced by any ``frames`` or ``videos`` row."""
        with self._lock:
            thumbs = self._conn.execute("SELECT thumb_path FROM frames").fetchall()
            plays = self._conn.execute(
                "SELECT playback_path FROM videos WHERE playback_path IS NOT NULL"
            ).fetchall()
        return {r[0] for r in thumbs} | {r[0] for r in plays}

    def allowed_vector_ids(self, f: SearchFilter) -> np.ndarray:
        """Vector IDs of complete videos satisfying every set filter part, ANDed (8.3)."""
        params = {
            "camera": f.camera_id,
            "start_ms": None if f.start is None else epoch_ms(f.start),
            "end_ms": None if f.end is None else epoch_ms(f.end),
            "cls": f.cls,
        }
        with self._lock:
            rows = self._conn.execute(ALLOWED_IDS_SQL, params).fetchall()
        return np.array([r[0] for r in rows], dtype=np.int64)

    def hit_rows(self, ids: Sequence[int]) -> dict[int, HitRow]:
        """Look up display data for hit IDs via a temp table (no SQLite variable-count limit)."""
        id_list = [(int(i),) for i in ids]
        if not id_list:
            return {}
        with self._lock:
            self._conn.execute("CREATE TEMP TABLE IF NOT EXISTS ids(id INTEGER PRIMARY KEY)")
            self._conn.execute("DELETE FROM temp.ids")
            try:
                self._conn.executemany("INSERT OR IGNORE INTO temp.ids(id) VALUES (?)", id_list)
                rows = self._conn.execute(HIT_ROWS_SQL).fetchall()
            finally:
                self._conn.execute("DELETE FROM temp.ids")
        out: dict[int, HitRow] = {}
        for r in rows:
            out[r[0]] = HitRow(
                vector_id=r[0], frame_id=r[1], video_id=r[2], camera_id=r[3], camera_label=r[4],
                offset_s=r[5], thumb_path=r[6], duration_s=r[7],
                start_ts=datetime.fromisoformat(r[8]), start_epoch_ms=r[9],
            )
        return out

    def cameras(self) -> list[tuple[str, str]]:
        with self._lock:
            rows = self._conn.execute("SELECT camera_id, label FROM cameras ORDER BY camera_id").fetchall()
        return [(r[0], r[1]) for r in rows]

    def counts(self) -> Counts:
        with self._lock:
            row = self._conn.execute(
                "SELECT (SELECT COUNT(*) FROM cameras), "
                "(SELECT COUNT(*) FROM videos WHERE status = 'complete'), "
                "(SELECT COUNT(*) FROM frames), (SELECT COUNT(*) FROM vectors)"
            ).fetchone()
        return Counts(cameras=row[0], videos=row[1], frames=row[2], vectors=row[3])

    def playback_for(self, video_id: int) -> str | None:
        """Playback file name of a complete video, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT playback_path FROM videos WHERE video_id = ? AND status = 'complete'",
                (int(video_id),),
            ).fetchone()
        return None if row is None else row[0]
