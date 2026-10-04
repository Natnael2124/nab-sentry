"""Sanity checks for nab_sentry.store.db (detailed property/unit tests live in tests/test_store_*.py)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from nab_sentry.search.engine import SearchFilter
from nab_sentry.store.db import MetadataStore, NewFrame, NewVector, NewVideo

T0 = datetime(2025, 1, 1, 8, 0, 0, tzinfo=timezone(timedelta(hours=11)))


@pytest.fixture
def store(tmp_path):
    s = MetadataStore(tmp_path / "db" / "nab_sentry.db")
    yield s
    s.close()


def _video(store: MetadataStore, cam: str, h: str, start: datetime = T0) -> int:
    store.upsert_camera(cam, f"label {cam}")
    return store.insert_video(
        NewVideo(cam, f"/videos/{h}.mp4", h, start, fps=10.0, width=640, height=360, est_duration_s=60.0)
    )


def _frame(store: MetadataStore, vid: int, idx: int, start: datetime = T0) -> int:
    off = idx / 10.0
    return store.insert_frame(
        NewFrame(vid, idx, off, start + timedelta(seconds=off), "motion", 0.5, f"v{vid}_f{idx:07d}.jpg")
    )


def test_pragmas_and_schema(store):
    conn = store._conn
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"schema_meta", "cameras", "videos", "frames", "vectors"} <= tables
    meta = dict(conn.execute("SELECT key, value FROM schema_meta"))
    assert meta["schema_version"] == "1" and meta["dim"] == "512"


def test_upsert_camera_results(store):
    assert store.upsert_camera("CAM-1", "Gate") == "created"
    assert store.upsert_camera("CAM-1", "Gate") == "exists"
    assert store.upsert_camera("CAM-1", "Other") == "label_conflict"
    assert store.cameras() == [("CAM-1", "Gate")]


def test_foreign_key_rejected(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_frame(NewFrame(999, 0, 0.0, T0, "first", 0.0, "x.jpg"))


def test_transaction_rolls_back(store):
    with pytest.raises(RuntimeError):
        with store.transaction():
            _video(store, "CAM-1", "h1")
            raise RuntimeError("boom")
    assert store.counts().cameras == 0
    assert not store.hash_exists("h1")
    assert not store.in_transaction


def test_finalize_filters_hits_and_delete(store):
    with store.transaction():
        v1 = _video(store, "CAM-1", "h1")
        f0 = _frame(store, v1, 0)
        f1 = _frame(store, v1, 600)  # +60 s
        a = store.insert_vector(NewVector(f0, "frame"))
        b = store.insert_vector(NewVector(f0, "crop", "person", 0.9, (1, 2, 30, 40)))
        c = store.insert_vector(NewVector(f1, "crop", "car", 0.8, (0, 0, 10, 10)))
    # not complete yet -> excluded from filtered search, playback hidden
    assert store.allowed_vector_ids(SearchFilter(camera_id="CAM-1")).tolist() == []
    assert store.playback_for(v1) is None
    store.finalize_video(v1, sampled=60, passed=2, duration_s=60.0, playback_path=f"v{v1}.mp4")

    assert store.vector_ids().tolist() == [a, b, c]
    assert store.allowed_vector_ids(SearchFilter(camera_id="CAM-1")).tolist() == [a, b, c]
    assert store.allowed_vector_ids(SearchFilter(camera_id="NOPE")).tolist() == []
    assert store.allowed_vector_ids(SearchFilter(cls="person")).tolist() == [b]
    assert store.allowed_vector_ids(SearchFilter(start=T0 + timedelta(seconds=30))).tolist() == [c]
    assert store.allowed_vector_ids(SearchFilter(end=T0)).tolist() == [a, b]  # inclusive end
    assert store.allowed_vector_ids(SearchFilter(start=T0, cls="car", camera_id="CAM-1")).tolist() == [c]
    # SQL syntax in a filter value is a literal (18.4)
    assert store.allowed_vector_ids(SearchFilter(camera_id="x' OR 1=1; --")).tolist() == []
    assert store.counts().vectors == 3

    rows = store.hit_rows([c, a, 12345])
    assert set(rows) == {a, c}
    assert rows[c].camera_label == "label CAM-1" and rows[c].offset_s == pytest.approx(60.0)
    assert rows[a].start_ts == T0 and rows[a].duration_s == 60.0
    assert store.playback_for(v1) == f"v{v1}.mp4"

    files = store.delete_video(v1)
    assert files == [f"v{v1}_f0000000.jpg", f"v{v1}_f0000600.jpg", f"v{v1}.mp4"]
    c_ = store.counts()
    assert (c_.videos, c_.frames, c_.vectors) == (0, 0, 0)


def test_search_filter_is_empty():
    assert SearchFilter().is_empty()
    assert not SearchFilter(cls="car").is_empty()
