"""Activity log in the ingest pipeline (Step 3, Requirement 19, task 18.2)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from nab_sentry.config import Config
from nab_sentry.ingest.pipeline import IngestPipeline
from nab_sentry.ingest.retention import apply_retention
from nab_sentry.live.camera_config import CameraSpec
from nab_sentry.store.db import MetadataStore
from nab_sentry.store.vector_index import VectorIndex
from nab_sentry.track.live import LiveTracks
from tests.fakes import FakeEncoder, FakePlayback, FakeSource
from tests.test_pipeline_tracks import FPS, H, W, WalkingPerson, frames, name

CAM = CameraSpec("CAM-A", "Back yard", "dshow:x",
                 zones={"lawn": ((0.0, 0.0), (0.4, 0.0), (0.4, 1.0), (0.0, 1.0))},
                 lines={"path": ((0.3, 0.0), (0.3, 1.0))})


def ingest(root: Path, videos):
    cfg = Config(root=root, enable_detector=True, sample_rate=1.0, track_rate=5.0, track_min_hits=3)
    cfg.videos_dir.mkdir(parents=True, exist_ok=True)
    for n in videos:
        (cfg.videos_dir / n).write_bytes(n.encode())
    db = MetadataStore(cfg.db_path)
    pipe = IngestPipeline(cfg, db, VectorIndex(), FakeEncoder(), WalkingPerson(),
                          open_source=lambda p: FakeSource(videos[Path(p).name], fps=FPS, width=W, height=H, path=p),
                          start_playback=FakePlayback(), tracker=LiveTracks.from_config(cfg), cameras=[CAM])
    assert {r.status for r in pipe.ingest_paths([cfg.videos_dir]).results} == {"ingested"}
    return cfg, db, pipe


def test_one_activity_per_track_with_zone_and_crossing(tmp_path: Path):
    cfg, db, _ = ingest(tmp_path, {name(0): frames(0, 25)})
    try:
        [a] = db.activities_between(0, 2**62)
        assert a["camera_id"] == "CAM-A" and a["det_class"] == "person"
        zones, crossings = json.loads(a["zones"]), json.loads(a["crossings"])
        assert zones["lawn"] > 0 and [c["direction"] for c in crossings] == ["in"]  # walks right
        assert "Back yard" in a["summary"] and "on the lawn" in a["summary"] and "crossed the path (in)" in a["summary"]
        assert a["thumb_path"] and (cfg.thumbs_dir / a["thumb_path"]).exists()  # best crop's frame
        assert isinstance(json.loads(a["tags"]), dict)
    finally:
        db.close()


def test_activity_grows_across_segments(tmp_path: Path):
    cfg, db, _ = ingest(tmp_path, {name(0): frames(0, 20), name(4): frames(20, 20)})
    try:
        [a] = db.activities_between(0, 2**62)
        n_obs = db._conn.execute("SELECT n_obs FROM tracks WHERE track_id = ?", (a["track_id"],)).fetchone()[0]
        assert n_obs == 40 and a["last_epoch_ms"] - a["first_epoch_ms"] >= 7000
        assert db._conn.execute("SELECT COUNT(*) FROM activities").fetchone()[0] == 1
    finally:
        db.close()


def test_retention_removes_the_activity_with_its_track(tmp_path: Path):
    cfg, db, pipe = ingest(tmp_path, {name(0): frames(0, 20)})
    try:
        assert len(db.activities_between(0, 2**62)) == 1
        apply_retention(Config(root=tmp_path, retention_days=1), db, pipe.index, ["CAM-A"],
                        datetime(2026, 1, 3).timestamp())
        assert db.activities_between(0, 2**62) == []
    finally:
        db.close()
