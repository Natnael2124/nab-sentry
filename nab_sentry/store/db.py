"""SQLite Metadata_Store: cameras, videos, frames, vectors and (schema v2) tracks.

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
import time
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

SCHEMA_VERSION = "3"
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

# Schema migrations: MIGRATIONS[v] upgrades version v to v + 1 (V2 Requirement 11). A new
# database is created with the version-1 DDL above and then migrated like an old one, so every
# database runs the same migration code. Statements run one by one inside BEGIN IMMEDIATE
# (``executescript`` would commit early).
MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """CREATE TABLE tracks (
            track_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id      TEXT    NOT NULL REFERENCES cameras(camera_id),
            det_class      TEXT    NOT NULL
                CHECK (det_class IN ('person','car','truck','bus','motorcycle','bicycle')),
            first_epoch_ms INTEGER NOT NULL,
            last_epoch_ms  INTEGER NOT NULL CHECK (last_epoch_ms >= first_epoch_ms),
            n_obs          INTEGER NOT NULL CHECK (n_obs >= 1)
        )""",
        "CREATE INDEX idx_tracks_camera_time ON tracks(camera_id, first_epoch_ms)",
        """CREATE TABLE track_obs (
            track_id     INTEGER NOT NULL REFERENCES tracks(track_id) ON DELETE CASCADE,
            video_id     INTEGER NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
            offset_s     REAL    NOT NULL CHECK (offset_s >= 0),
            abs_epoch_ms INTEGER NOT NULL,
            conf         REAL    NOT NULL CHECK (conf BETWEEN 0 AND 1),
            high         INTEGER NOT NULL CHECK (high IN (0, 1)),
            x1 INTEGER NOT NULL, y1 INTEGER NOT NULL, x2 INTEGER NOT NULL, y2 INTEGER NOT NULL,
            CHECK (x1 >= 0 AND y1 >= 0 AND x1 < x2 AND y1 < y2),
            PRIMARY KEY (track_id, video_id, offset_s)
        )""",
        "CREATE INDEX idx_track_obs_video ON track_obs(video_id)",
        "ALTER TABLE vectors ADD COLUMN track_id INTEGER REFERENCES tracks(track_id) ON DELETE SET NULL",
        "CREATE INDEX idx_vectors_track ON vectors(track_id)",
    ),
    # v2 -> v3: the alert outbox (V2 Requirements 12.5, 14, 15), written by the Watcher inside
    # the Segment's transaction and delivered by the Notifier.
    2: (
        """CREATE TABLE alerts (
            alert_id              INTEGER PRIMARY KEY AUTOINCREMENT,
            kind                  TEXT    NOT NULL
                CHECK (kind IN ('rule', 'camera_offline', 'camera_recovered', 'disk_low')),
            rule_id               TEXT,
            rule_name             TEXT,
            severity              TEXT    CHECK (severity IN ('immediate', 'verify')),
            camera_id             TEXT    REFERENCES cameras(camera_id),
            track_id              INTEGER REFERENCES tracks(track_id) ON DELETE SET NULL,
            video_id              INTEGER REFERENCES videos(video_id) ON DELETE SET NULL,
            trigger_offset_s      REAL,
            trigger_epoch_ms      INTEGER NOT NULL,
            det_class             TEXT,
            n_tracks              INTEGER,
            state                 TEXT    NOT NULL
                CHECK (state IN ('pending', 'photo_sent', 'sent', 'failed', 'suppressed')),
            attempts              INTEGER NOT NULL DEFAULT 0,
            next_attempt_epoch_ms INTEGER NOT NULL DEFAULT 0,
            snapshot_path         TEXT,
            clip_path             TEXT,
            note                  TEXT,
            feedback              TEXT    CHECK (feedback IN ('real', 'false')),
            created_at            TEXT    NOT NULL,
            CHECK ((kind = 'rule') = (rule_id IS NOT NULL))
        )""",
        "CREATE INDEX idx_alerts_state ON alerts(state, next_attempt_epoch_ms)",
        "CREATE INDEX idx_alerts_rule ON alerts(rule_id, camera_id, trigger_epoch_ms)",
        """CREATE TABLE alert_messages (
            alert_id   INTEGER NOT NULL REFERENCES alerts(alert_id) ON DELETE CASCADE,
            chat_id    INTEGER NOT NULL,
            step       TEXT    NOT NULL CHECK (step IN ('photo', 'clip', 'text')),
            message_id INTEGER NOT NULL,
            PRIMARY KEY (alert_id, chat_id, step)
        )""",
        "CREATE TABLE rule_mutes (rule_id TEXT PRIMARY KEY, until_epoch_ms INTEGER NOT NULL)",
        "CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    ),
}


class SchemaVersionError(RuntimeError):
    """The database was written by a newer NAB Sentry (schema version above this code's)."""


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
class NewTrackObs:
    video_id: int
    offset_s: float
    abs_epoch_ms: int
    conf: float
    high: bool
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class NewAlert:
    kind: str  # rule | camera_offline | camera_recovered | disk_low
    trigger_epoch_ms: int
    state: str = "pending"
    rule_id: str | None = None
    rule_name: str | None = None
    severity: str | None = None
    camera_id: str | None = None
    track_id: int | None = None
    video_id: int | None = None
    trigger_offset_s: float | None = None
    det_class: str | None = None
    n_tracks: int | None = None
    snapshot_path: str | None = None
    note: str | None = None


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
        # busy_timeout first: switching to WAL and creating/migrating the schema need locks, and
        # several processes (watcher, API server) may open the same database at the same time.
        self._conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")  # int-coerced constant
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._enable_wal(busy_timeout_ms)
        self.init_schema()

    # -- lifecycle -----------------------------------------------------------------------------

    def _enable_wal(self, timeout_ms: int) -> None:
        """Switch to WAL unless it already is. SQLite fails a journal-mode change at once
        ("database is locked", no busy handler) while another connection has the file open, so
        retry until ``timeout_ms``; a database NAB Sentry created is WAL already (it persists)."""
        if self._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal":
            return
        deadline = time.monotonic() + timeout_ms / 1000
        while True:
            try:
                self._conn.execute("PRAGMA journal_mode = WAL")
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> MetadataStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(DDL)  # version-1 tables; no-op on an existing database
            self._conn.executemany(
                "INSERT OR IGNORE INTO schema_meta(key, value) VALUES (?, ?)",
                [("schema_version", "1"), ("embed_model", EMBED_MODEL), ("dim", EMBED_DIM)],
            )
            self.migrate()

    def schema_version(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
        return int(row[0]) if row else 0

    def migrate(self) -> list[int]:
        """Upgrade to ``SCHEMA_VERSION`` one step per transaction; returns the versions applied.

        Each step re-reads the version inside ``BEGIN IMMEDIATE``, so two processes opening the
        same old database cannot both apply it. A newer database raises SchemaVersionError.
        """
        target = int(SCHEMA_VERSION)
        applied: list[int] = []
        with self._lock:
            while True:
                with self._txn():
                    (cur,) = self._conn.execute(
                        "SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
                    version = int(cur)
                    if version > target:
                        raise SchemaVersionError(
                            f"{self.path}: schema version {version} is newer than this NAB Sentry "
                            f"({target}); upgrade the software instead of opening it")
                    if version == target:
                        return applied
                    for stmt in MIGRATIONS[version]:
                        self._conn.execute(stmt)
                    self._conn.execute(
                        "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
                        (str(version + 1),))
                log.info("migrated %s from schema version %d to %d", self.path, version, version + 1)
                applied.append(version + 1)

    # -- transactions --------------------------------------------------------------------------

    @property
    def in_transaction(self) -> bool:
        return self._conn.in_transaction

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """``BEGIN IMMEDIATE`` ... ``COMMIT``; ``ROLLBACK`` on any exception (incl. KeyboardInterrupt)."""
        with self._txn():
            yield

    @contextmanager
    def _txn(self) -> Iterator[None]:
        """The implementation of :meth:`transaction`; used directly by :meth:`migrate`, which
        runs inside ``__init__`` where a subclass's overrides may not be ready yet."""
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

    def insert_track(self, camera_id: str, det_class: str, first_epoch_ms: int,
                     last_epoch_ms: int, n_obs: int) -> int:
        with self._write():
            cur = self._conn.execute(
                "INSERT INTO tracks(camera_id, det_class, first_epoch_ms, last_epoch_ms, n_obs) "
                "VALUES (?, ?, ?, ?, ?)",
                (camera_id, det_class, int(first_epoch_ms), int(last_epoch_ms), int(n_obs)),
            )
            return int(cur.lastrowid)

    def extend_track(self, track_id: int, last_epoch_ms: int, added_obs: int) -> None:
        """A continuing track got ``added_obs`` more observations, the latest at ``last_epoch_ms``."""
        with self._write():
            cur = self._conn.execute(
                "UPDATE tracks SET last_epoch_ms = MAX(last_epoch_ms, ?), n_obs = n_obs + ? "
                "WHERE track_id = ?",
                (int(last_epoch_ms), int(added_obs), int(track_id)),
            )
            if cur.rowcount != 1:
                raise KeyError(f"no tracks row with track_id={track_id}")

    def insert_track_obs(self, track_id: int, obs: Sequence[NewTrackObs]) -> None:
        with self._write():
            self._conn.executemany(
                "INSERT INTO track_obs(track_id, video_id, offset_s, abs_epoch_ms, conf, high, "
                "x1, y1, x2, y2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(int(track_id), int(o.video_id), float(o.offset_s), int(o.abs_epoch_ms),
                  float(o.conf), int(bool(o.high)), *(int(c) for c in o.box)) for o in obs],
            )

    def link_vectors(self, pairs: Sequence[tuple[int, int]]) -> None:
        """Set ``vectors.track_id`` for ``(vector_id, track_id)`` pairs."""
        with self._write():
            self._conn.executemany("UPDATE vectors SET track_id = ? WHERE vector_id = ?",
                                   [(int(t), int(v)) for v, t in pairs])

    def delete_empty_tracks(self) -> int:
        """Delete tracks left without observations (their videos were deleted); returns count."""
        with self._write():
            cur = self._conn.execute(
                "DELETE FROM tracks WHERE NOT EXISTS "
                "(SELECT 1 FROM track_obs o WHERE o.track_id = tracks.track_id)")
            return int(cur.rowcount)

    # -- alerts (schema v3) -----------------------------------------------------------------------

    def insert_alert(self, a: NewAlert) -> int:
        with self._write():
            cur = self._conn.execute(
                "INSERT INTO alerts(kind, rule_id, rule_name, severity, camera_id, track_id, video_id, "
                "trigger_offset_s, trigger_epoch_ms, det_class, n_tracks, state, snapshot_path, note, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (a.kind, a.rule_id, a.rule_name, a.severity, a.camera_id, a.track_id, a.video_id,
                 a.trigger_offset_s, int(a.trigger_epoch_ms), a.det_class, a.n_tracks, a.state,
                 a.snapshot_path, a.note, _now_iso()),
            )
            return int(cur.lastrowid)

    def last_rule_fires(self, camera_id: str) -> dict[str, int]:
        """rule_id -> latest trigger_epoch_ms of its alerts on ``camera_id`` (cooldown seed)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT rule_id, MAX(trigger_epoch_ms) FROM alerts "
                "WHERE kind = 'rule' AND camera_id = ? GROUP BY rule_id", (camera_id,)).fetchall()
        return {r[0]: int(r[1]) for r in rows}

    def active_mutes(self, now_epoch_ms: int) -> dict[str, int]:
        """rule_id -> muted-until epoch ms, for mutes still running at ``now_epoch_ms``."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT rule_id, until_epoch_ms FROM rule_mutes WHERE until_epoch_ms > ?",
                (int(now_epoch_ms),)).fetchall()
        return {r[0]: int(r[1]) for r in rows}

    def set_mute(self, rule_id: str, until_epoch_ms: int | None) -> None:
        """Mute ``rule_id`` until the given time; ``None`` removes the mute."""
        with self._write():
            if until_epoch_ms is None:
                self._conn.execute("DELETE FROM rule_mutes WHERE rule_id = ?", (rule_id,))
            else:
                self._conn.execute(
                    "INSERT INTO rule_mutes(rule_id, until_epoch_ms) VALUES (?, ?) "
                    "ON CONFLICT(rule_id) DO UPDATE SET until_epoch_ms = excluded.until_epoch_ms",
                    (rule_id, int(until_epoch_ms)))

    # -- outbox (read and updated by the Notifier) ------------------------------------------------

    ALERT_COLUMNS = ("alert_id", "kind", "rule_id", "rule_name", "severity", "camera_id", "track_id",
                     "video_id", "trigger_offset_s", "trigger_epoch_ms", "det_class", "n_tracks",
                     "state", "attempts", "next_attempt_epoch_ms", "snapshot_path", "clip_path",
                     "note", "feedback", "created_at")
    _ALERT_UPDATABLE = frozenset({"state", "attempts", "next_attempt_epoch_ms", "clip_path", "note",
                                  "feedback"})

    def _alert_rows(self, sql: str, params: tuple) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(zip(self.ALERT_COLUMNS, r)) for r in rows]

    def due_alerts(self, now_epoch_ms: int, limit: int = 20) -> list[dict]:
        """Alerts the Notifier should work on now (pending / photo_sent, retry time reached)."""
        cols = ", ".join(self.ALERT_COLUMNS)
        return self._alert_rows(
            f"SELECT {cols} FROM alerts WHERE state IN ('pending', 'photo_sent') "
            "AND next_attempt_epoch_ms <= ? ORDER BY alert_id LIMIT ?", (int(now_epoch_ms), int(limit)))

    def get_alert(self, alert_id: int) -> dict | None:
        cols = ", ".join(self.ALERT_COLUMNS)
        rows = self._alert_rows(f"SELECT {cols} FROM alerts WHERE alert_id = ?", (int(alert_id),))
        return rows[0] if rows else None

    def alerts_between(self, from_epoch_ms: int, to_epoch_ms: int) -> list[dict]:
        cols = ", ".join(self.ALERT_COLUMNS)
        return self._alert_rows(
            f"SELECT {cols} FROM alerts WHERE trigger_epoch_ms >= ? AND trigger_epoch_ms < ? "
            "ORDER BY trigger_epoch_ms", (int(from_epoch_ms), int(to_epoch_ms)))

    def update_alert(self, alert_id: int, **fields: object) -> None:
        bad = set(fields) - self._ALERT_UPDATABLE
        if bad:
            raise ValueError(f"not updatable: {sorted(bad)}")
        if not fields:
            return
        names = sorted(fields)  # column names come from the whitelist above, never from callers
        sets = ", ".join(f"{n} = ?" for n in names)
        with self._write():
            cur = self._conn.execute(f"UPDATE alerts SET {sets} WHERE alert_id = ?",
                                     (*(fields[n] for n in names), int(alert_id)))
            if cur.rowcount != 1:
                raise KeyError(f"no alerts row with alert_id={alert_id}")

    def sent_steps(self, alert_id: int) -> dict[tuple[int, str], int]:
        """(chat_id, step) -> Telegram message_id already accepted for this alert."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT chat_id, step, message_id FROM alert_messages WHERE alert_id = ?",
                (int(alert_id),)).fetchall()
        return {(int(r[0]), r[1]): int(r[2]) for r in rows}

    def record_message(self, alert_id: int, chat_id: int, step: str, message_id: int) -> None:
        with self._write():
            self._conn.execute(
                "INSERT OR REPLACE INTO alert_messages(alert_id, chat_id, step, message_id) "
                "VALUES (?, ?, ?, ?)", (int(alert_id), int(chat_id), step, int(message_id)))

    def alert_for_message(self, chat_id: int, message_id: int) -> int | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT alert_id FROM alert_messages WHERE chat_id = ? AND message_id = ?",
                (int(chat_id), int(message_id))).fetchone()
        return None if row is None else int(row[0])

    def clip_segments(self, camera_id: str, from_epoch_ms: int, to_epoch_ms: int) -> list[tuple[str, int, float]]:
        """(src_path, start_epoch_ms, duration_s) of complete videos of ``camera_id`` overlapping
        the window, in time order."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT src_path, start_epoch_ms, duration_s FROM videos WHERE camera_id = ? "
                "AND status = 'complete' AND start_epoch_ms < ? "
                "AND start_epoch_ms + CAST(duration_s * 1000 AS INTEGER) > ? ORDER BY start_epoch_ms",
                (camera_id, int(to_epoch_ms), int(from_epoch_ms))).fetchall()
        return [(r[0], int(r[1]), float(r[2])) for r in rows]

    def latest_video(self, camera_id: str) -> tuple[str, int, float] | None:
        """(src_path, start_epoch_ms, duration_s) of the camera's newest complete video."""
        with self._lock:
            row = self._conn.execute(
                "SELECT src_path, start_epoch_ms, duration_s FROM videos WHERE camera_id = ? "
                "AND status = 'complete' ORDER BY start_epoch_ms DESC LIMIT 1", (camera_id,)).fetchone()
        return None if row is None else (row[0], int(row[1]), float(row[2]))

    def kv_get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def kv_set(self, key: str, value: str) -> None:
        with self._write():
            self._conn.execute("INSERT INTO kv(key, value) VALUES (?, ?) "
                               "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    def referenced_alert_files(self) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT snapshot_path FROM alerts WHERE snapshot_path IS NOT NULL "
                "UNION SELECT clip_path FROM alerts WHERE clip_path IS NOT NULL").fetchall()
        return {r[0] for r in rows}

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
