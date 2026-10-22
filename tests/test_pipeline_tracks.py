"""Track storage in the ingest transaction (V2 Requirement 10.7, tasks 11.3-11.4)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from nab_sentry.config import Config
from nab_sentry.ingest.detector import Detection
from nab_sentry.ingest.pipeline import IngestPipeline
from nab_sentry.store.db import MetadataStore
from nab_sentry.store.vector_index import VectorIndex
from nab_sentry.track.live import LiveTracks
from tests.fakes import FailureInjector, FakeEncoder, FakePlayback, FakeSource

W, H = 96, 64
FPS = 5.0
START = 1767225600  # 2026-01-01T00:00:00Z


def name(sec: int, cam: str = "CAM-A") -> str:
    return f"{cam}_20260101T{sec // 3600:02d}{sec // 60 % 60:02d}{sec % 60:02d}.mp4"


def frames(first: int, n: int) -> list[np.ndarray]:
    """A bright bar sweeping 4 px per frame: about 40% of pixels change between sample frames
    1 s apart (motion), about 8% between consecutive frames (no scene cut). The global frame
    number is stored in pixel (0, 0)."""
    out = []
    for i in range(first, first + n):
        img = np.full((H, W, 3), 60, np.uint8)
        x = (4 * i) % (W + 20) - 20
        img[:, max(0, x):max(0, x + 20)] = 220
        img[0, 0] = (i % 256, i // 256, 0)
        out.append(img)
    return out


class WalkingPerson:
    """A person walking right 1 px per frame (from pixel (0, 0)), plus a low car box."""

    def __init__(self, person_from: int = 0) -> None:
        self.person_from = person_from

    def detect(self, frame: np.ndarray) -> list[Detection]:
        i = int(frame[0, 0, 0]) + 256 * int(frame[0, 0, 1])
        out = [Detection("car", 0.2, (60, 30, 90, 60), high=False)]
        if i >= self.person_from:
            x = 5 + (i - self.person_from) % 40
            out.insert(0, Detection("person", 0.9, (x, 5, x + 20, 50)))
        return out


def ingest(root: Path, videos: dict[str, list[np.ndarray]], *, person_from: int = 0,
           checkpoint=None, retry: str | None = None, cfg_kw: dict | None = None):
    cfg = Config(root=root, enable_detector=True, sample_rate=1.0, track_rate=5.0,
                 track_min_hits=3, **(cfg_kw or {}))
    cfg.videos_dir.mkdir(parents=True)
    for n in videos:
        (cfg.videos_dir / n).write_bytes(n.encode())
    db = MetadataStore(cfg.db_path)
    try:
        pipe = IngestPipeline(
            cfg, db, VectorIndex(), FakeEncoder(), WalkingPerson(person_from),
            open_source=lambda p: FakeSource(videos[Path(p).name], fps=FPS, width=W, height=H, path=p),
            start_playback=FakePlayback(), tracker=LiveTracks.from_config(cfg), checkpoint=checkpoint)
        res = pipe.ingest_paths([cfg.videos_dir]).results
        if retry is not None:
            if {r.path.name: r.status for r in res}[retry] != "ingested":
                assert pipe.ingest_video(cfg.videos_dir / retry).status == "ingested"
        else:
            assert {r.status for r in res} == {"ingested"}, res
        return dump(db), db, pipe
    finally:
        db.close()


def dump(db: MetadataStore) -> dict:
    c = db._conn
    return {
        "tracks": c.execute("SELECT track_id, camera_id, det_class, first_epoch_ms, last_epoch_ms, n_obs "
                            "FROM tracks ORDER BY 1").fetchall(),
        "obs": c.execute("SELECT o.track_id, v.src_path, o.offset_s, o.abs_epoch_ms, o.high, o.x1, o.y1, o.x2, o.y2 "
                         "FROM track_obs o JOIN videos v USING (video_id) ORDER BY 1, 4").fetchall(),
        "links": c.execute("SELECT v.vector_id, v.det_class, v.track_id FROM vectors v WHERE kind = 'crop' "
                           "ORDER BY 1").fetchall(),
        "videos": c.execute("SELECT video_id, src_path, start_epoch_ms FROM videos ORDER BY 1").fetchall(),
    }


def check_invariants(d: dict) -> None:
    by_track: dict[int, list] = {}
    for o in d["obs"]:
        by_track.setdefault(o[0], []).append(o)
    starts = {Path(p).name: ms for _, p, ms in d["videos"]}
    for tid, cam, cls, first, last, n in d["tracks"]:
        obs = by_track[tid]
        assert n == len(obs)
        assert first == min(o[3] for o in obs) and last == max(o[3] for o in obs)
        for o in obs:  # abs time = its own video's start + offset
            assert o[3] == round(starts[Path(o[1]).name] + o[2] * 1000)
    classes = {t[0]: t[2] for t in d["tracks"]}
    for _vid, cls, tid in d["links"]:
        if tid is not None:
            assert classes[tid] == cls


def test_one_video_stores_one_person_track_and_links_its_crops(tmp_path: Path):
    d, _, _ = ingest(tmp_path, {name(0): frames(0, 25)})
    check_invariants(d)
    assert [t[2] for t in d["tracks"]] == ["person"]  # the low car never starts a track
    n = d["tracks"][0][5]
    assert n == 25  # every frame at 5 fps (track_rate == fps) while the scene moves
    assert {o[4] for o in d["obs"]} == {1}
    assert all(tid == d["tracks"][0][0] for _, cls, tid in d["links"] if cls == "person")


def test_track_continues_across_segments_as_one_row(tmp_path: Path):
    vids = {name(0): frames(0, 20), name(4): frames(20, 20)}  # two contiguous 4 s Segments
    d, _, _ = ingest(tmp_path, vids)
    check_invariants(d)
    assert len(d["tracks"]) == 1
    assert {Path(o[1]).name for o in d["obs"]} == set(vids)
    assert d["tracks"][0][5] == 40


def test_backfilled_observations_reference_the_previous_segment(tmp_path: Path):
    """The person appears 2 frames before the end of Segment 1 (tentative there) and is
    confirmed in Segment 2: those 2 observations belong to Segment 1's video."""
    vids = {name(0): frames(0, 20), name(4): frames(20, 20)}
    d, _, _ = ingest(tmp_path, vids, person_from=18)
    check_invariants(d)
    [(tid, *_rest)] = d["tracks"]
    seg1 = [o for o in d["obs"] if Path(o[1]).name == name(0)]
    assert [o[2] for o in seg1] == [3.6, 3.8]
    # Segment 1's crop of the then-tentative person stays unlinked (known limit).
    assert d["tracks"][0][5] == 22


def test_gap_between_segments_starts_a_new_track(tmp_path: Path):
    vids = {name(0): frames(0, 20), name(60): frames(20, 20)}  # 56 s gap
    d, _, _ = ingest(tmp_path, vids)
    check_invariants(d)
    assert len(d["tracks"]) == 2


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.sampled_from(["frame", "thumbnail", "embed", "playback_wait", "finalize", "index_add",
                        "index_save", "after_index_save", "commit"]),
       st.integers(1, 3), st.sampled_from([0, 18]))
def test_failure_then_retry_stores_exactly_the_clean_rows(step, nth, person_from):
    """Task 11.4: Segment 2 fails at any checkpoint and is retried; tracks, observations and
    crop links equal a clean run (and the failed attempt left nothing behind)."""
    vids = {name(0): frames(0, 20), name(4): frames(20, 20)}
    with tempfile.TemporaryDirectory() as tmp:
        clean, _, _ = ingest(Path(tmp) / "clean", vids, person_from=person_from)
        inj = FailureInjector(step, nth=nth if step in ("frame", "thumbnail") else 1, video=name(4))
        failed, _, _ = ingest(Path(tmp) / "failed", vids, person_from=person_from,
                              checkpoint=inj, retry=name(4))
    assume(inj.fired == 1)  # e.g. a 3rd thumbnail may not exist in this Segment
    check_invariants(failed)
    # video_id / vector_id / track_id values can differ after a rollback (AUTOINCREMENT), so
    # compare by content.
    def norm(d):
        tmap = {t[0]: i for i, t in enumerate(d["tracks"])}
        return ([t[1:] for t in d["tracks"]],
                [(tmap[o[0]], Path(o[1]).name, *o[2:]) for o in d["obs"]],
                sorted((cls, None if tid is None else tmap[tid]) for _, cls, tid in d["links"]))
    assert norm(failed) == norm(clean)


def test_reconcile_deletes_tracks_of_deleted_videos(tmp_path: Path):
    vids = {name(0): frames(0, 20), name(60): frames(20, 20)}
    _, _, pipe = ingest(tmp_path, vids)
    cfg = pipe.cfg
    db = MetadataStore(cfg.db_path)
    try:
        assert db._conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 2
        db._conn.execute("UPDATE videos SET status = 'ingesting' WHERE src_path LIKE ?", (f"%{name(60)}",))
        pipe.db = db
        pipe.index = VectorIndex.load(cfg.index_path)
        pipe.reconcile()  # deletes the "unfinished" video, cascading its observations
        assert db._conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 1
        assert db._conn.execute("SELECT COUNT(*) FROM track_obs").fetchone()[0] == 20
    finally:
        db.close()
