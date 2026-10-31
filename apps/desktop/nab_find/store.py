"""Case storage: one folder per case with ``case.db`` (SQLite) and ``thumbs/`` (design: store.py).

SQLite holds everything, embeddings included (512 float32 = 2 KB each), so a crash can never leave
an index out of step with the database: the in-memory search index is rebuilt from SQLite when a
case opens (about 1 s per 100k vectors) and appended to after every commit. Search is an exact dot
product in NumPy, fast enough for a case (8 busy hours is tens of thousands of vectors).

One ``CaseStore`` per open case, shared by the worker thread (writes) and the server threads
(reads); a lock serialises access to the connection and the in-memory index.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

DIM = 512
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS videos (
  video_id INTEGER PRIMARY KEY, path TEXT NOT NULL, camera TEXT NOT NULL, start_ts REAL NOT NULL,
  start_estimated INTEGER NOT NULL, duration_s REAL NOT NULL, fps REAL NOT NULL, width INTEGER NOT NULL,
  height INTEGER NOT NULL, keyframe_gap_s REAL, status TEXT NOT NULL DEFAULT 'ready', error TEXT);
CREATE TABLE IF NOT EXISTS spans (
  video_id INTEGER NOT NULL, pass TEXT NOT NULL, start_s REAL NOT NULL, end_s REAL NOT NULL);
CREATE TABLE IF NOT EXISTS motion (video_id INTEGER NOT NULL, start_s REAL NOT NULL, end_s REAL NOT NULL);
CREATE TABLE IF NOT EXISTS frames (
  frame_id INTEGER PRIMARY KEY, video_id INTEGER NOT NULL, offset_s REAL NOT NULL, thumb TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS visits (
  visit_id INTEGER PRIMARY KEY, video_id INTEGER NOT NULL, cls TEXT NOT NULL, t0 REAL NOT NULL,
  t1 REAL NOT NULL, n_obs INTEGER NOT NULL, max_speed REAL NOT NULL, path_json TEXT NOT NULL,
  tags_json TEXT NOT NULL DEFAULT '{}', thumb TEXT);
CREATE TABLE IF NOT EXISTS vectors (
  vector_id INTEGER PRIMARY KEY, kind TEXT NOT NULL, video_id INTEGER NOT NULL, offset_s REAL NOT NULL,
  frame_id INTEGER, visit_id INTEGER, x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER, thumb TEXT,
  vec BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS ai_checks (
  id INTEGER PRIMARY KEY, query TEXT NOT NULL, model TEXT NOT NULL, created REAL NOT NULL,
  result_json TEXT NOT NULL, tokens_in INTEGER NOT NULL, tokens_out INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS vectors_visit ON vectors(visit_id);
CREATE INDEX IF NOT EXISTS spans_video ON spans(video_id, pass);
"""
KIND_FRAME, KIND_CROP = 0, 1


@dataclass
class Video:
    video_id: int
    path: str
    camera: str
    start_ts: float
    start_estimated: bool
    duration_s: float
    fps: float
    width: int
    height: int
    keyframe_gap_s: float | None
    status: str
    error: str | None

    @property
    def end_ts(self) -> float:
        return self.start_ts + self.duration_s


@dataclass
class FrameRow:
    video_id: int
    offset_s: float
    thumb: str
    vec: np.ndarray


@dataclass
class CropRow:
    video_id: int
    offset_s: float
    visit_key: int  # processor's track key; mapped to visit_id in the same commit
    box: tuple[int, int, int, int]
    thumb: str
    vec: np.ndarray


@dataclass
class VisitRow:
    key: int
    video_id: int
    cls: str
    t0: float
    t1: float
    n_obs: int
    max_speed: float
    path: list[tuple[float, float, float]]  # (offset_s, cx, cy) normalised
    tags: dict[str, Any] = field(default_factory=dict)
    thumb: str | None = None


@dataclass
class Chunk:
    """Everything one processing chunk produced; committed in one transaction."""

    video_id: int
    pass_name: str
    span: tuple[float, float]
    frames: list[FrameRow] = field(default_factory=list)
    crops: list[CropRow] = field(default_factory=list)
    visits: list[VisitRow] = field(default_factory=list)
    motion: list[tuple[float, float]] = field(default_factory=list)


class VectorTable:
    """In-memory copy of the ``vectors`` table for search: ids, unit vectors and filter columns."""

    def __init__(self) -> None:
        self.n = 0
        self.ids = np.empty(0, np.int64)
        self.vecs = np.empty((0, DIM), np.float32)
        self.kind = np.empty(0, np.int8)
        self.video = np.empty(0, np.int64)
        self.offset = np.empty(0, np.float64)
        self.visit = np.empty(0, np.int64)  # -1: none

    def _grow(self, need: int) -> None:
        cap = len(self.ids)
        if need <= cap:
            return
        new = max(need, cap * 2, 1024)
        for name in ("ids", "kind", "video", "offset", "visit"):
            a = getattr(self, name)
            b = np.empty(new, a.dtype)
            b[:self.n] = a[:self.n]
            setattr(self, name, b)
        v = np.empty((new, DIM), np.float32)
        v[:self.n] = self.vecs[:self.n]
        self.vecs = v

    def add(self, ids, vecs, kind, video, offset, visit) -> None:
        k = len(ids)
        if k == 0:
            return
        self._grow(self.n + k)
        s = slice(self.n, self.n + k)
        self.ids[s], self.vecs[s], self.kind[s] = ids, vecs, kind
        self.video[s], self.offset[s], self.visit[s] = video, offset, visit
        self.n += k

    def view(self) -> dict[str, np.ndarray]:
        n = self.n
        return {"ids": self.ids[:n], "vecs": self.vecs[:n], "kind": self.kind[:n], "video": self.video[:n],
                "offset": self.offset[:n], "visit": self.visit[:n]}


def _vec_blob(v: np.ndarray) -> bytes:
    return np.ascontiguousarray(v, dtype=np.float32).tobytes()


class CaseStore:
    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self.thumbs = self.folder / "thumbs"
        self.previews = self.folder / "previews"
        self.lock = threading.RLock()
        self._con = sqlite3.connect(self.folder / "case.db", check_same_thread=False, isolation_level=None)
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA synchronous=NORMAL")
        self._con.executescript(SCHEMA)
        self.table = VectorTable()
        self._load_vectors()

    # ------------------------------------------------------------- create / open
    @classmethod
    def create(cls, root: Path, name: str) -> "CaseStore":
        folder = Path(root) / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        (folder / "thumbs").mkdir(parents=True)
        st = cls(folder)
        st.set_meta(name=name, created=time.time(), status="new", range=None)
        return st

    def close(self) -> None:
        with self.lock:
            self._con.close()

    def delete(self) -> None:
        self.close()
        shutil.rmtree(self.folder, ignore_errors=True)

    @property
    def case_id(self) -> str:
        return self.folder.name

    # ------------------------------------------------------------- meta
    def meta(self) -> dict[str, Any]:
        with self.lock:
            return {k: json.loads(v) for k, v in self._con.execute("SELECT key, value FROM meta")}

    def set_meta(self, **kv: Any) -> None:
        with self.lock:
            self._con.executemany("INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE "
                                  "SET value = excluded.value", [(k, json.dumps(v)) for k, v in kv.items()])

    # ------------------------------------------------------------- videos
    def add_video(self, path: str, camera: str, start_ts: float, estimated: bool, duration_s: float, fps: float,
                  width: int, height: int, keyframe_gap_s: float | None, status: str = "ready",
                  error: str | None = None) -> int:
        with self.lock:
            cur = self._con.execute(
                "INSERT INTO videos(path, camera, start_ts, start_estimated, duration_s, fps, width, height, "
                "keyframe_gap_s, status, error) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (path, camera, start_ts, int(estimated), duration_s, fps, width, height, keyframe_gap_s, status,
                 error))
            return int(cur.lastrowid)

    def videos(self) -> list[Video]:
        with self.lock:
            rows = self._con.execute("SELECT * FROM videos ORDER BY camera, start_ts").fetchall()
        return [Video(r[0], r[1], r[2], r[3], bool(r[4]), r[5], r[6], r[7], r[8], r[9], r[10], r[11]) for r in rows]

    def video(self, video_id: int) -> Video | None:
        return next((v for v in self.videos() if v.video_id == video_id), None)

    def update_video(self, video_id: int, **kv: Any) -> None:
        allowed = {"camera", "start_ts", "start_estimated", "status", "error"}
        sets = {k: (int(v) if isinstance(v, bool) else v) for k, v in kv.items() if k in allowed}
        if not sets:
            return
        with self.lock:
            self._con.execute(f"UPDATE videos SET {', '.join(f'{k} = ?' for k in sets)} WHERE video_id = ?",
                              (*sets.values(), video_id))

    def remove_video(self, video_id: int) -> None:
        with self.lock:
            self._con.execute("BEGIN")
            for t in ("spans", "motion", "frames", "visits", "vectors", "videos"):
                self._con.execute(f"DELETE FROM {t} WHERE video_id = ?", (video_id,))
            self._con.execute("COMMIT")
            self.table = VectorTable()
            self._load_vectors()

    # ------------------------------------------------------------- work done
    def spans(self, video_id: int, pass_name: str) -> list[tuple[float, float]]:
        with self.lock:
            return [(a, b) for a, b in self._con.execute(
                "SELECT start_s, end_s FROM spans WHERE video_id = ? AND pass = ? ORDER BY start_s",
                (video_id, pass_name))]

    def motion(self, video_id: int) -> list[tuple[float, float]]:
        with self.lock:
            return [(a, b) for a, b in self._con.execute(
                "SELECT start_s, end_s FROM motion WHERE video_id = ? ORDER BY start_s", (video_id,))]

    def reset_work(self) -> None:
        """Forget all processing results (the range changed): videos stay."""
        with self.lock:
            self._con.execute("BEGIN")
            for t in ("spans", "motion", "frames", "visits", "vectors"):
                self._con.execute(f"DELETE FROM {t}")
            self._con.execute("COMMIT")
            self.table = VectorTable()
        shutil.rmtree(self.thumbs, ignore_errors=True)
        self.thumbs.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- commit
    def commit(self, chunk: Chunk) -> dict[int, int]:
        """Write a chunk in one transaction, then add its vectors to the search table.
        Returns the processor's track key -> visit_id map for visits written."""
        with self.lock:
            con = self._con
            saved = dict(getattr(self, "_visit_key_map", {}))
            con.execute("BEGIN")
            try:
                keymap = self._visit_ids(chunk.visits)
                new_ids, new_vecs, kinds, vids, offs, visits = [], [], [], [], [], []
                for f in chunk.frames:
                    fid = con.execute("INSERT INTO frames(video_id, offset_s, thumb) VALUES (?,?,?)",
                                      (f.video_id, f.offset_s, f.thumb)).lastrowid
                    vid = con.execute("INSERT INTO vectors(kind, video_id, offset_s, frame_id, thumb, vec) "
                                      "VALUES ('frame',?,?,?,?,?)",
                                      (f.video_id, f.offset_s, fid, f.thumb, _vec_blob(f.vec))).lastrowid
                    new_ids.append(vid), new_vecs.append(f.vec), kinds.append(KIND_FRAME)
                    vids.append(f.video_id), offs.append(f.offset_s), visits.append(-1)
                for c in chunk.crops:
                    visit_id = keymap.get(c.visit_key) or self._visit_key_map[(c.video_id, c.visit_key)]
                    vid = con.execute(
                        "INSERT INTO vectors(kind, video_id, offset_s, visit_id, x1, y1, x2, y2, thumb, vec) "
                        "VALUES ('crop',?,?,?,?,?,?,?,?,?)",
                        (c.video_id, c.offset_s, visit_id, *c.box, c.thumb, _vec_blob(c.vec))).lastrowid
                    new_ids.append(vid), new_vecs.append(c.vec), kinds.append(KIND_CROP)
                    vids.append(c.video_id), offs.append(c.offset_s), visits.append(visit_id)
                con.executemany("INSERT INTO motion(video_id, start_s, end_s) VALUES (?,?,?)",
                                [(chunk.video_id, a, b) for a, b in chunk.motion])
                con.execute("INSERT INTO spans(video_id, pass, start_s, end_s) VALUES (?,?,?,?)",
                            (chunk.video_id, chunk.pass_name, *chunk.span))
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                self._visit_key_map = saved
                raise
            if new_ids:
                self.table.add(np.asarray(new_ids, np.int64), np.stack(new_vecs).astype(np.float32),
                               np.asarray(kinds, np.int8), np.asarray(vids, np.int64), np.asarray(offs),
                               np.asarray(visits, np.int64))
            return keymap

    _visit_key_map: dict[tuple[int, int], int]

    def _visit_ids(self, visits: Sequence[VisitRow]) -> dict[int, int]:
        """Insert or update visits (inside the open transaction); key -> visit_id."""
        if not hasattr(self, "_visit_key_map"):
            self._visit_key_map = {}
        out = {}
        for v in visits:
            row = (v.video_id, v.cls, v.t0, v.t1, v.n_obs, v.max_speed, json.dumps(v.path), json.dumps(v.tags),
                   v.thumb)
            known = self._visit_key_map.get((v.video_id, v.key))
            if known is None:
                known = self._con.execute(
                    "INSERT INTO visits(video_id, cls, t0, t1, n_obs, max_speed, path_json, tags_json, thumb) "
                    "VALUES (?,?,?,?,?,?,?,?,?)", row).lastrowid
                self._visit_key_map[(v.video_id, v.key)] = known
            else:
                self._con.execute(
                    "UPDATE visits SET video_id=?, cls=?, t0=?, t1=?, n_obs=?, max_speed=?, path_json=?, "
                    "tags_json=?, thumb=COALESCE(?, thumb) WHERE visit_id=?", (*row, known))
            out[v.key] = int(known)
        return out

    def forget_track_keys(self) -> None:
        """Track keys restart with every processor run: never map a new key onto an old visit."""
        self._visit_key_map = {}

    def set_visit_tags(self, visit_id: int, tags: dict[str, Any]) -> None:
        with self.lock:
            self._con.execute("UPDATE visits SET tags_json = ? WHERE visit_id = ?", (json.dumps(tags), visit_id))

    def _load_vectors(self) -> None:
        rows = self._con.execute(
            "SELECT vector_id, kind, video_id, offset_s, COALESCE(visit_id, -1), vec FROM vectors ORDER BY vector_id"
        ).fetchall()
        if not rows:
            return
        self.table.add(np.asarray([r[0] for r in rows], np.int64),
                       np.frombuffer(b"".join(r[5] for r in rows), np.float32).reshape(len(rows), DIM),
                       np.asarray([KIND_FRAME if r[1] == "frame" else KIND_CROP for r in rows], np.int8),
                       np.asarray([r[2] for r in rows], np.int64), np.asarray([r[3] for r in rows]),
                       np.asarray([r[4] for r in rows], np.int64))

    # ------------------------------------------------------------- reads
    def query(self, sql: str, args: Iterable[Any] = ()) -> list[tuple]:
        with self.lock:
            return self._con.execute(sql, tuple(args)).fetchall()

    def vectors_of_visit(self, visit_id: int) -> np.ndarray:
        rows = self.query("SELECT vec FROM vectors WHERE visit_id = ?", (visit_id,))
        return np.frombuffer(b"".join(r[0] for r in rows), np.float32).reshape(len(rows), DIM)

    def counts(self) -> dict[str, int]:
        q = self.query
        return {"frames": q("SELECT COUNT(*) FROM frames")[0][0], "visits": q("SELECT COUNT(*) FROM visits")[0][0],
                "vectors": q("SELECT COUNT(*) FROM vectors")[0][0]}

    def add_ai_check(self, query: str, model: str, result: Any, tokens_in: int, tokens_out: int) -> int:
        with self.lock:
            return int(self._con.execute(
                "INSERT INTO ai_checks(query, model, created, result_json, tokens_in, tokens_out) "
                "VALUES (?,?,?,?,?,?)", (query, model, time.time(), json.dumps(result), tokens_in, tokens_out)
            ).lastrowid)


def list_cases(root: Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if (p / "case.db").is_file()), reverse=True)
