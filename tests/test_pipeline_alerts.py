"""Rule alerts in the ingest transaction (V2 Requirement 12.5, tasks 13.4-13.6): Property 18."""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from nab_sentry.config import Config
from nab_sentry.ingest.pipeline import IngestPipeline
from nab_sentry.rules.config import Rule
from nab_sentry.store.db import MetadataStore
from nab_sentry.store.vector_index import VectorIndex
from nab_sentry.track.live import LiveTracks
from tests.fakes import FailureInjector, FakeEncoder, FakePlayback, FakeSource
from tests.test_pipeline_tracks import FPS, H, W, WalkingPerson, frames, name

LEFT_HALF = ((0.0, 0.0), (0.6, 0.0), (0.6, 1.0), (0.0, 1.0))


def yard(**kw) -> Rule:
    base = dict(id="yard", name="Person in the yard", camera="CAM-A", classes=frozenset({"person"}),
                trigger="enter", zone="yard", zone_poly=LEFT_HALF, debounce=2, cooldown_s=300.0)
    base.update(kw)
    return Rule(**base)


def ingest(root: Path, videos, rules, *, person_from=0, checkpoint=None, retry=None, mute=None):
    cfg = Config(root=root, enable_detector=True, sample_rate=1.0, track_rate=5.0, track_min_hits=3)
    cfg.videos_dir.mkdir(parents=True, exist_ok=True)
    for n in videos:
        (cfg.videos_dir / n).write_bytes(n.encode())
    db = MetadataStore(cfg.db_path)
    try:
        if mute:
            db.set_mute(*mute)
        pipe = IngestPipeline(
            cfg, db, VectorIndex(), FakeEncoder(), WalkingPerson(person_from),
            open_source=lambda p: FakeSource(videos[Path(p).name], fps=FPS, width=W, height=H, path=p),
            start_playback=FakePlayback(), tracker=LiveTracks.from_config(cfg), checkpoint=checkpoint,
            rules=rules)
        res = pipe.ingest_paths([cfg.videos_dir]).results
        if retry is not None and {r.path.name: r.status for r in res}[retry] != "ingested":
            assert pipe.ingest_video(cfg.videos_dir / retry).status == "ingested"
        alerts = db._conn.execute(
            "SELECT a.rule_id, a.state, a.trigger_offset_s, v.src_path, a.det_class, a.snapshot_path, "
            "a.track_id, t.det_class FROM alerts a LEFT JOIN videos v USING (video_id) "
            "LEFT JOIN tracks t USING (track_id) ORDER BY a.alert_id").fetchall()
        files = sorted(p.name for p in cfg.alerts_dir.glob("*")) if cfg.alerts_dir.is_dir() else []
        return alerts, files, cfg, pipe
    finally:
        db.close()


def test_rule_fires_once_with_snapshot_and_track(tmp_path: Path):
    alerts, files, cfg, _ = ingest(tmp_path, {name(0): frames(0, 20)}, [yard()])
    [(rid, state, offset, src, cls, snap, track_id, track_cls)] = alerts
    assert (rid, state, cls, track_cls) == ("yard", "pending", "person", "person")
    assert Path(src).name == name(0) and track_id is not None
    assert offset < 2.0  # confirmed after 3 hits, then debounce 2: well inside the first seconds
    assert files == [snap]


def test_cooldown_spans_segments_and_survives_a_restart(tmp_path: Path):
    vids = {name(0): frames(0, 20), name(4): frames(20, 20)}
    alerts, _, _, _ = ingest(tmp_path, vids, [yard()])
    assert len(alerts) == 1
    # A new pipeline (process restart) on the same store: the cooldown is seeded from the
    # stored alert, so a new person 20 s later does not alert again.
    more = {name(24): frames(40, 20)}
    alerts2, _, _, _ = ingest(tmp_path, more, [yard()])
    assert len(alerts2) == 1
    alerts3, _, _, _ = ingest(tmp_path, more, [yard(cooldown_s=5.0)])  # already indexed: no-op
    assert len(alerts3) == 1


def test_muted_rule_stores_a_suppressed_alert(tmp_path: Path):
    alerts, files, _, _ = ingest(tmp_path, {name(0): frames(0, 20)}, [yard()],
                                 mute=("yard", 4_102_444_800_000))  # muted until 2100
    assert [a[1] for a in alerts] == ["suppressed"] and len(files) == 1


def test_other_class_and_outside_zone_do_not_fire(tmp_path: Path):
    right = ((0.95, 0.0), (1.0, 0.0), (1.0, 1.0), (0.95, 1.0))
    alerts, files, _, _ = ingest(tmp_path, {name(0): frames(0, 20)},
                                 [yard(id="far", zone_poly=right), yard(id="cars", classes=frozenset({"car"}))])
    assert alerts == [] and files == []


def test_cross_rule_fires_in_direction(tmp_path: Path):
    """The person walks right 1 px per frame; a vertical line at x = 0.3 crossed left to right.
    A line from top to bottom: its left (as seen on screen) is the right-hand half."""
    down_line = ((0.3, 0.0), (0.3, 1.0))
    rin = yard(id="in", trigger="cross", zone=None, zone_poly=None, line="l", line_pts=down_line, direction=1)
    rout = yard(id="out", trigger="cross", zone=None, zone_poly=None, line="l", line_pts=down_line, direction=-1)
    alerts, _, _, _ = ingest(tmp_path, {name(0): frames(0, 25)}, [rin, rout])
    assert [a[0] for a in alerts] == ["in"]


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.sampled_from(["frame", "thumbnail", "embed", "playback_wait", "finalize", "index_add",
                        "index_save", "after_index_save", "commit"]), st.integers(1, 3))
def test_property_18_failure_then_retry_equals_clean_run(step, nth):
    """The person appears in Segment 2, which fails at ``step`` and is retried: alerts and alert
    files equal a clean run, and the failed attempt left none behind."""
    vids = {name(0): frames(0, 20), name(4): frames(20, 20)}
    with tempfile.TemporaryDirectory() as tmp:
        clean, clean_files, _, _ = ingest(Path(tmp) / "c", vids, [yard()], person_from=22)
        inj = FailureInjector(step, nth=nth if step in ("frame", "thumbnail") else 1, video=name(4))
        failed, failed_files, _, _ = ingest(Path(tmp) / "f", vids, [yard()], person_from=22,
                                            checkpoint=inj, retry=name(4))
    assume(inj.fired == 1)
    norm = lambda rows: [(r[0], r[1], r[2], Path(r[3]).name, r[4], r[7]) for r in rows]  # noqa: E731
    assert len(clean) == 1 and norm(failed) == norm(clean)
    # File names embed the video id, which differs after a rollback; compare counts and the
    # referenced set instead.
    assert len(failed_files) == len(clean_files) == 1
    assert failed_files == [failed[0][5]]


def test_reconcile_deletes_unreferenced_alert_files(tmp_path: Path):
    _, files, cfg, pipe = ingest(tmp_path, {name(0): frames(0, 20)}, [yard()])
    stray = cfg.alerts_dir / "a_v99_yard_123.jpg"
    stray.write_bytes(b"x")
    keep = cfg.alerts_dir / "notes.txt"  # not our naming pattern: never touched
    keep.write_text("keep")
    db = MetadataStore(cfg.db_path)
    try:
        pipe.db, pipe.index = db, VectorIndex.load(cfg.index_path)
        report = pipe.reconcile()
    finally:
        db.close()
    assert "a_v99_yard_123.jpg" in report.deleted_files and not stray.exists()
    assert keep.exists() and (cfg.alerts_dir / files[0]).exists()
