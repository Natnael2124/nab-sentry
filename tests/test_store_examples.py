"""Example-based unit tests for the Metadata_Store (Requirements 6.1, 6.9)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from nab_sentry.store.db import MetadataStore, NewFrame, NewVector, NewVideo, epoch_ms, iso_ms

START = datetime(2024, 5, 1, 12, 0, 0, 250000, tzinfo=timezone(timedelta(hours=2)))


@pytest.fixture
def store(tmp_path: Path):
    s = MetadataStore(tmp_path / "meta.sqlite3")
    yield s
    s.close()


def _video(src_hash: str = "a" * 64, camera_id: str = "cam1", **kw) -> NewVideo:
    fields = dict(
        camera_id=camera_id, src_path=f"C:/footage/{src_hash[:8]}.mp4", src_hash=src_hash,
        start_ts=START, fps=25.0, width=640, height=480, est_duration_s=12.5,
    )
    fields.update(kw)
    return NewVideo(**fields)


def _frame(video_id: int, idx: int, thumb: str) -> NewFrame:
    return NewFrame(
        video_id=video_id, frame_idx=idx, offset_s=idx / 25.0,
        abs_time=START + timedelta(seconds=idx / 25.0), gate_reason="motion",
        motion_frac=0.1, thumb_path=thumb,
    )


def _populate(store: MetadataStore, src_hash: str, tag: str, *, playback: str | None) -> int:
    """One video with three frames, each with a frame vector and one crop vector."""
    vid = store.insert_video(_video(src_hash))
    for idx in (0, 5, 10):
        fid = store.insert_frame(_frame(vid, idx, f"{tag}_{idx}.jpg"))
        store.insert_vector(NewVector(frame_id=fid, kind="frame"))
        store.insert_vector(
            NewVector(frame_id=fid, kind="crop", det_class="person", det_conf=0.9, box=(1, 2, 30, 40))
        )
    if playback is not None:
        store.finalize_video(vid, sampled=10, passed=3, duration_s=12.4, playback_path=playback)
    return vid


def _count(store: MetadataStore, table: str, video_id: int) -> int:
    queries = {
        "videos": "SELECT COUNT(*) FROM videos WHERE video_id = ?",
        "frames": "SELECT COUNT(*) FROM frames WHERE video_id = ?",
        "vectors": "SELECT COUNT(*) FROM vectors v JOIN frames f ON f.frame_id = v.frame_id "
        "WHERE f.video_id = ?",
    }
    return store._conn.execute(queries[table], (video_id,)).fetchone()[0]


# -- 6.1: foreign keys reject a missing parent -------------------------------------------------


def test_video_with_missing_camera_is_rejected(store: MetadataStore) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_video(_video(camera_id="nope"))
    assert store._conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0


def test_frame_with_missing_video_is_rejected(store: MetadataStore) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_frame(_frame(9999, 0, "x.jpg"))
    assert store._conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0] == 0


def test_vector_with_missing_frame_is_rejected(store: MetadataStore) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_vector(NewVector(frame_id=9999, kind="frame"))
    assert store._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 0


def test_inserts_with_existing_parents_succeed(store: MetadataStore) -> None:
    store.upsert_camera("cam1", "Front gate")
    vid = store.insert_video(_video())
    fid = store.insert_frame(_frame(vid, 0, "t0.jpg"))
    vec = store.insert_vector(NewVector(frame_id=fid, kind="frame"))
    assert (vid, fid, vec) == (1, 1, 1)


# -- 6.9: videos row contents at ingest start --------------------------------------------------


def test_videos_row_holds_start_fields(store: MetadataStore) -> None:
    store.upsert_camera("cam1", "Front gate")
    v = _video("b" * 64)
    vid = store.insert_video(v)
    row = store._conn.execute(
        "SELECT camera_id, src_path, src_hash, start_ts, start_epoch_ms, duration_s, status, "
        "playback_path FROM videos WHERE video_id = ?",
        (vid,),
    ).fetchone()
    assert row[0] == "cam1"
    assert row[1] == v.src_path
    assert row[2] == "b" * 64
    assert row[3] == iso_ms(START) == "2024-05-01T12:00:00.250+02:00"
    assert datetime.fromisoformat(row[3]) == START
    assert row[4] == epoch_ms(START)
    assert row[5] == pytest.approx(12.5)
    assert row[6] == "ingesting"
    assert row[7] is None
    # Not complete yet, so not counted and no playback file exposed.
    assert store.counts().videos == 0
    assert store.playback_for(vid) is None


# -- delete_video cascades and returns file names ----------------------------------------------


def test_delete_video_cascades_and_returns_file_names(store: MetadataStore) -> None:
    store.upsert_camera("cam1", "Front gate")
    keep = _populate(store, "c" * 64, "keep", playback="keep.mp4")
    gone = _populate(store, "d" * 64, "gone", playback="gone.mp4")
    before = store.counts()
    assert (before.videos, before.frames, before.vectors) == (2, 6, 12)

    files = store.delete_video(gone)

    assert files == ["gone_0.jpg", "gone_5.jpg", "gone_10.jpg", "gone.mp4"]
    assert [_count(store, t, gone) for t in ("videos", "frames", "vectors")] == [0, 0, 0]
    # The other video's rows are untouched.
    assert [_count(store, t, keep) for t in ("videos", "frames", "vectors")] == [1, 3, 6]
    after = store.counts()
    assert (after.cameras, after.videos, after.frames, after.vectors) == (1, 1, 3, 6)
    assert store.playback_for(keep) == "keep.mp4"
    assert store.hash_exists("d" * 64) is False


def test_delete_ingesting_video_returns_only_thumbs(store: MetadataStore) -> None:
    store.upsert_camera("cam1", "Front gate")
    vid = _populate(store, "e" * 64, "partial", playback=None)
    assert store.delete_video(vid) == ["partial_0.jpg", "partial_5.jpg", "partial_10.jpg"]
    assert store.counts().frames == 0 and store.counts().vectors == 0


def test_delete_missing_video_returns_empty(store: MetadataStore) -> None:
    assert store.delete_video(12345) == []
