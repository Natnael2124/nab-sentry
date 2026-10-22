"""Retention (V2 Requirement 15.3, tasks 16.2-16.3): Property 22 - after retention,
``reconcile()`` finds nothing to repair; only old footage of live cameras is deleted."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nab_sentry.config import Config
from nab_sentry.ingest.pipeline import IngestPipeline
from nab_sentry.ingest.retention import apply_retention
from nab_sentry.ingest.watch import Watcher
from nab_sentry.store.db import MetadataStore
from nab_sentry.store.vector_index import VectorIndex
from nab_sentry.track.live import LiveTracks
from tests.fakes import FakeEncoder, FakePlayback, FakeSource
from tests.test_pipeline_alerts import yard
from tests.test_pipeline_tracks import FPS, H, W, WalkingPerson, frames, name

LIVE_OLD, LIVE_NEW, IMPORTED = name(0), name(12 * 3600), name(0, cam="CAM-B")


def build(root: Path, retention_days: float, alert_days: float):
    cfg = Config(root=root, enable_detector=True, sample_rate=1.0, track_rate=5.0, track_min_hits=3,
                 retention_days=retention_days, alert_retention_days=alert_days)
    cfg.videos_dir.mkdir(parents=True)
    vids = {LIVE_OLD: frames(0, 20), LIVE_NEW: frames(20, 20), IMPORTED: frames(40, 20)}
    for n in vids:
        (cfg.videos_dir / n).write_bytes(n.encode())
    db = MetadataStore(cfg.db_path)
    index = VectorIndex()
    pipe = IngestPipeline(cfg, db, index, FakeEncoder(), WalkingPerson(),
                          open_source=lambda p: FakeSource(vids[Path(p).name], fps=FPS, width=W, height=H, path=p),
                          start_playback=FakePlayback(), tracker=LiveTracks.from_config(cfg),
                          rules=[yard(cooldown_s=0.0), yard(id="b", camera="CAM-B", cooldown_s=0.0)])
    assert {r.status for r in pipe.ingest_paths([cfg.videos_dir]).results} == {"ingested"}
    return cfg, db, index, pipe


def on_disk(cfg: Config) -> set[str]:
    return {p.name for d in (cfg.thumbs_dir, cfg.playback_dir, cfg.alerts_dir, cfg.videos_dir)
            if d.is_dir() for p in d.iterdir()}


def test_only_old_live_footage_is_deleted(tmp_path: Path):
    cfg, db, index, pipe = build(tmp_path, retention_days=1.0, alert_days=30.0)
    try:
        now = datetime(2026, 1, 2, 6, 0).timestamp()  # LIVE_OLD ended > 1 day ago, LIVE_NEW not
        before = {r[1] for r in db._conn.execute("SELECT video_id, src_path FROM videos")}
        rep = apply_retention(cfg, db, index, ["CAM-A"], now)
        left = {Path(r[0]).name for r in db._conn.execute("SELECT src_path FROM videos")}
        assert left == {LIVE_NEW, IMPORTED} and len(before) == 3
        assert LIVE_OLD in rep.files and not (cfg.videos_dir / LIVE_OLD).exists()
        assert (cfg.videos_dir / IMPORTED).exists()  # not a live camera: never deleted
        assert rep.vectors > 0 and set(index.ids().tolist()) == set(db.vector_ids().tolist())
        # Alert rows stay; their media is younger than alert_retention_days.
        assert db._conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 3
        assert pipe.reconcile().changed is False
    finally:
        db.close()


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.sampled_from([1.0, 2.0]), st.sampled_from([1.0, 30.0]), st.floats(0, 3 * 86400))
def test_property_22_reconcile_finds_nothing_after_retention(days, alert_days, later):
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, index, pipe = build(Path(tmp), days, alert_days)
        try:
            now = datetime(2026, 1, 1).timestamp() + later
            apply_retention(cfg, db, index, ["CAM-A", "CAM-B"], now)
            assert pipe.reconcile().changed is False
            assert set(index.ids().tolist()) == set(db.vector_ids().tolist())
            assert set(VectorIndex.load(cfg.index_path).ids().tolist()) == set(db.vector_ids().tolist())
            referenced = db.referenced_files() | db.referenced_alert_files()
            stored_sources = {Path(r[0]).name for r in db._conn.execute("SELECT src_path FROM videos")}
            assert on_disk(cfg) == referenced | stored_sources
            # No track without observations, no observation without a video.
            assert db._conn.execute("SELECT COUNT(*) FROM tracks t WHERE NOT EXISTS "
                                    "(SELECT 1 FROM track_obs o WHERE o.track_id = t.track_id)").fetchone()[0] == 0
        finally:
            db.close()


def test_alert_media_expires_but_rows_stay(tmp_path: Path):
    cfg, db, index, _ = build(tmp_path, retention_days=3650, alert_days=1.0)
    try:
        rep = apply_retention(cfg, db, index, ["CAM-A"], datetime(2026, 1, 3).timestamp())
        assert len(rep.alert_files) == 3 and not list(cfg.alerts_dir.glob("a_*"))
        assert db._conn.execute("SELECT COUNT(*) FROM alerts WHERE snapshot_path IS NULL").fetchone()[0] == 3
    finally:
        db.close()


def test_watcher_runs_maintenance_hourly():
    calls: list[float] = []
    t = {"now": 0.0}
    w = Watcher(None, Path("."), lambda c: 10.0, scan=lambda d: [], clock=lambda: t["now"],  # type: ignore[arg-type]
                maintenance=calls.append, maintenance_every_s=3600)
    for now in (0, 100, 3599, 3600, 3700, 7300):
        t["now"] = float(now)
        w.maintain()
    assert calls == [0.0, 3600.0, 7300.0]
