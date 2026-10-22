"""Schema migration v1 -> current (v3) (V2 Requirement 11.1, task 11.2): Property 19.

Migrating any v1 database keeps every v1 row, adds an all-NULL ``vectors.track_id`` and the
track tables, passes SQLite's integrity and foreign-key checks, and is idempotent. A database
from a newer version is refused, and two stores opening the same old file migrate it once.
"""

from __future__ import annotations

import sqlite3
import tempfile
import threading
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nab_sentry.store.db import DDL, SCHEMA_VERSION, MetadataStore, SchemaVersionError

V1_TABLES = ("cameras", "videos", "frames", "vectors")
V1_VECTOR_COLS = "vector_id, frame_id, kind, det_class, det_conf, x1, y1, x2, y2"
CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]


def make_v1(path: Path, data) -> None:
    """A database exactly as V1 wrote it: the version-1 DDL, schema_version '1', rows."""
    con = sqlite3.connect(path)
    con.executescript(DDL)
    con.executemany("INSERT INTO schema_meta(key, value) VALUES (?, ?)",
                    [("schema_version", "1"), ("embed_model", "ViT-B-32/laion2b_s34b_b79k"), ("dim", "512")])
    cams, videos = data
    for i, cam in enumerate(cams):
        con.execute("INSERT INTO cameras VALUES (?, ?, '2026-01-01T00:00:00+00:00')", (cam, f"label {i}"))
    for v_i, (cam_i, frames) in enumerate(videos):
        vid = con.execute(
            "INSERT INTO videos(camera_id, src_path, src_hash, start_ts, start_epoch_ms, fps, width, "
            "height, duration_s, sampled_count, passed_count, playback_path, status, ingested_at) "
            "VALUES (?, ?, ?, '2026-01-01T00:00:00.000+00:00', 1767225600000, 10, 64, 48, ?, ?, ?, ?, "
            "'complete', '2026-01-01T00:00:00+00:00')",
            (cams[cam_i % len(cams)], f"v{v_i}.mp4", f"h{v_i}", float(len(frames)), len(frames),
             len(frames), f"v{v_i}.mp4")).lastrowid
        for f_i, crops in enumerate(frames):
            fid = con.execute(
                "INSERT INTO frames(video_id, frame_idx, offset_s, abs_ts, abs_epoch_ms, gate_reason, "
                "motion_frac, thumb_path) VALUES (?, ?, ?, '2026-01-01T00:00:00.000+00:00', ?, 'motion', 0.5, ?)",
                (vid, f_i, float(f_i), 1767225600000 + 1000 * f_i, f"v{vid}_f{f_i:07d}.jpg")).lastrowid
            con.execute("INSERT INTO vectors(frame_id, kind) VALUES (?, 'frame')", (fid,))
            for cls, conf in crops:
                con.execute("INSERT INTO vectors(frame_id, kind, det_class, det_conf, x1, y1, x2, y2) "
                            "VALUES (?, 'crop', ?, ?, 1, 2, 30, 40)", (fid, cls, conf))
    con.commit()
    con.close()


def dump(con: sqlite3.Connection) -> dict:
    out = {t: con.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall() for t in V1_TABLES[:3]}
    out["vectors"] = con.execute(f"SELECT {V1_VECTOR_COLS} FROM vectors ORDER BY 1").fetchall()
    return out


v1_data = st.tuples(
    st.lists(st.from_regex(r"CAM-[A-Z0-9]{1,6}", fullmatch=True), min_size=1, max_size=3, unique=True),
    st.lists(st.tuples(st.integers(0, 2),
                       st.lists(st.lists(st.tuples(st.sampled_from(CLASSES), st.floats(0, 1)), max_size=3),
                                max_size=5)),
             max_size=4),
)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(v1_data)
def test_property_19_migration_keeps_v1_rows_and_is_idempotent(data):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "v1.db"
        make_v1(path, data)
        con = sqlite3.connect(path)
        before = dump(con)
        con.close()

        db = MetadataStore(path)
        try:
            c = db._conn
            assert db.schema_version() == int(SCHEMA_VERSION)
            assert dump(c) == before
            assert c.execute("SELECT COUNT(*) FROM vectors WHERE track_id IS NOT NULL").fetchone()[0] == 0
            assert {"tracks", "track_obs", "alerts", "alert_messages", "rule_mutes", "kv"} <= {r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'")}
            assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert c.execute("PRAGMA foreign_key_check").fetchall() == []
            assert db.migrate() == []  # idempotent
        finally:
            db.close()
        db2 = MetadataStore(path)  # reopening is a no-op too
        try:
            assert db2.schema_version() == int(SCHEMA_VERSION) and dump(db2._conn) == before
        finally:
            db2.close()


def test_new_database_is_created_at_current_version(tmp_path: Path):
    db = MetadataStore(tmp_path / "new.db")
    try:
        assert db.schema_version() == int(SCHEMA_VERSION)
        assert [r[1] for r in db._conn.execute("PRAGMA table_info(vectors)")][-1] == "track_id"
    finally:
        db.close()


def test_newer_database_is_refused(tmp_path: Path):
    path = tmp_path / "future.db"
    MetadataStore(path).close()
    con = sqlite3.connect(path)
    con.execute("UPDATE schema_meta SET value = ? WHERE key = 'schema_version'", (str(int(SCHEMA_VERSION) + 1),))
    con.commit()
    con.close()
    with pytest.raises(SchemaVersionError, match="newer"):
        MetadataStore(path)


def test_two_processes_opening_an_old_database_migrate_it_once(tmp_path: Path):
    path = tmp_path / "v1.db"
    make_v1(path, (["CAM-A"], [(0, [[("person", 0.9)]])]))
    errors: list[BaseException] = []
    stores: list[MetadataStore] = []
    barrier = threading.Barrier(4)

    def open_store():
        try:
            barrier.wait()
            stores.append(MetadataStore(path))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=open_store) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert errors == []
        c = stores[0]._conn
        assert stores[0].schema_version() == int(SCHEMA_VERSION)
        # The ALTER TABLE ran exactly once (a second run would have failed: duplicate column).
        assert [r[1] for r in c.execute("PRAGMA table_info(vectors)")].count("track_id") == 1
    finally:
        for s in stores:
            s.close()


def test_track_rows_cascade_with_their_video(tmp_path: Path):
    from nab_sentry.store.db import NewTrackObs

    path = tmp_path / "t.db"
    make_v1(path, (["CAM-A"], [(0, [[("person", 0.9)]]), (0, [[("person", 0.8)]])]))
    db = MetadataStore(path)
    try:
        tid = db.insert_track("CAM-A", "person", 1000, 2000, 2)
        db.insert_track_obs(tid, [NewTrackObs(1, 0.0, 1000, 0.9, True, (1, 2, 30, 40)),
                                  NewTrackObs(2, 0.0, 2000, 0.2, False, (2, 2, 31, 40))])
        crop = db._conn.execute("SELECT vector_id FROM vectors WHERE kind = 'crop' ORDER BY 1").fetchone()[0]
        db.link_vectors([(crop, tid)])
        db.extend_track(tid, 1500, 0)  # never moves last_epoch_ms backwards
        assert db._conn.execute("SELECT last_epoch_ms FROM tracks").fetchone()[0] == 2000
        db.delete_video(1)
        assert db._conn.execute("SELECT COUNT(*) FROM track_obs").fetchone()[0] == 1
        assert db._conn.execute("SELECT track_id FROM vectors WHERE vector_id = ?", (crop,)).fetchone() is None
        assert db.delete_empty_tracks() == 0
        db.delete_video(2)
        assert db.delete_empty_tracks() == 1
    finally:
        db.close()
