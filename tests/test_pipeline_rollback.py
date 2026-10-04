"""Property 24: metadata store and vector index stay one-to-one under failures (task 10.7).

For a random set of 1-4 fake videos and a random injected failure (an ordinary per-step error,
a playback-job failure, a ``KeyboardInterrupt``, or a ``SimulatedCrash`` followed by a
"restart" = reopen the DB + reload ``vectors.faiss`` from disk + ``reconcile()``):

- DB vector IDs == in-memory index IDs == on-disk index IDs, and the thumbnail/playback files
  on disk are exactly the ones the DB references (so the failed video left no files);
- the failed video has no ``videos`` row;
- every other ingested video's rows, vectors, and files equal those of a clean run;
- a later clean run ingests the failed (and any unprocessed) videos, ending in the clean state.

**Validates: Requirements 6.7, 6.8, 6.10, 7.7, 8.10**
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from nab_sentry.config import Config
from nab_sentry.ingest.detector import Detection
from nab_sentry.ingest.pipeline import CHECKPOINTS, IngestPipeline
from nab_sentry.store.db import MetadataStore
from nab_sentry.store.vector_index import VectorIndex
from tests.fakes import (
    FailureInjector,
    FakeDetector,
    FakeEncoder,
    FakePlayback,
    FakeSource,
    SimulatedCrash,
)

W, H, FPS, N = 64, 48, 4.0, 24  # 6 s of video per file

NAMES = (
    "CAM-A_20250101T080000.mp4",
    "CAM-B_20250101T090000.mp4",
    "CAM-C_20250102T100000.mp4",
    "CAM-D_20250103T110000.mp4",
)
PER_FRAME_STEPS = ("frame", "thumbnail", "embed")


def _frames(seed: int) -> list[np.ndarray]:
    """A white square that jumps every 4 frames, so the gate passes some frames, not all."""
    out = []
    for i in range(N):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        x = (seed * 7 + (i // 4) * 9) % (W - 12)
        img[10:22, x:x + 12] = 255
        out.append(img)
    return out


def _detections(frame: np.ndarray, i: int) -> list[Detection]:
    """Content-based (not call-index-based) so a video's detections don't depend on history.

    One ``FakeDetector`` is shared by all videos of a run, so its call index ``i`` is shifted by
    whatever earlier (possibly failed) videos consumed; keying on ``i`` made a later video's
    crops differ from a clean run even though the pipeline rolled back correctly.
    """
    hit = hashlib.sha256(np.ascontiguousarray(frame).tobytes()).digest()[0] % 2 == 0
    return [Detection("person", 0.9, (2, 3, 20, 30))] if hit else []


def _digest(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


class Env:
    def __init__(self, root: Path, seeds: dict[str, int], detector: bool,
                 playback: FakePlayback | None = None) -> None:
        self.cfg = Config(root=root, sample_rate=2.0, keyframe_interval_s=2.0,
                          enable_detector=detector, batch_size=3)
        self.videos_dir = root / "videos"
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.seeds = seeds
        for name, seed in seeds.items():
            (self.videos_dir / name).write_bytes(f"video-{seed}".encode())
        self.detector_on = detector
        self.playback = playback or FakePlayback()
        self.db = MetadataStore(self.cfg.db_path)
        self.index = VectorIndex()

    def open_source(self, path: Path) -> FakeSource:
        return FakeSource(_frames(self.seeds[Path(path).name]), fps=FPS, width=W, height=H,
                          path=path)

    def pipeline(self, checkpoint: FailureInjector | None = None,
                 playback: FakePlayback | None = None) -> IngestPipeline:
        detector = FakeDetector(_detections) if self.detector_on else None
        return IngestPipeline(
            self.cfg, self.db, self.index, FakeEncoder(batch_size=self.cfg.batch_size), detector,
            open_source=self.open_source, start_playback=playback or FakePlayback(),
            checkpoint=checkpoint,
        )

    def run(self, **kw: object):
        return self.pipeline(**kw).ingest_paths([self.videos_dir])  # type: ignore[arg-type]

    def restart(self) -> None:
        """Process restart: reopen the DB and reload the index from disk."""
        self.db.close()
        self.db = MetadataStore(self.cfg.db_path)
        p = self.cfg.index_path
        self.index = VectorIndex.load(p) if p.exists() else VectorIndex()

    def close(self) -> None:
        self.db.close()

    # -- invariant ---------------------------------------------------------------------------

    def assert_consistent(self) -> None:
        db_ids = set(self.db.vector_ids().tolist())
        assert set(self.index.ids().tolist()) == db_ids, "in-memory index != vectors table"
        p = self.cfg.index_path
        disk = set(VectorIndex.load(p).ids().tolist()) if p.exists() else set()
        assert disk == db_ids, "on-disk index != vectors table"
        thumbs = {f.name for f in self.cfg.thumbs_dir.glob("*")} if self.cfg.thumbs_dir.is_dir() else set()
        plays = {f.name for f in self.cfg.playback_dir.glob("*")} if self.cfg.playback_dir.is_dir() else set()
        assert thumbs | plays == self.db.referenced_files(), "media files != DB references"
        # Every thumbnail is a JPEG; every complete video's playback file exists.
        conn = self.db._conn
        for (t,) in conn.execute("SELECT thumb_path FROM frames"):
            assert (self.cfg.thumbs_dir / t).read_bytes()[:2] == b"\xff\xd8"
        for (pb,) in conn.execute("SELECT playback_path FROM videos WHERE status='complete'"):
            assert pb and (self.cfg.playback_dir / pb).is_file()
        assert not list(conn.execute("SELECT 1 FROM videos WHERE status != 'complete'"))

    # -- content snapshot (independent of row IDs) -------------------------------------------

    def snapshot(self) -> dict[str, tuple]:
        conn = self.db._conn
        fx = self.index.faiss_index
        out: dict[str, tuple] = {}
        videos = conn.execute(
            "SELECT d.video_id, d.src_path, d.camera_id, c.label, d.start_ts, d.fps, d.width, "
            "d.height, d.duration_s, d.sampled_count, d.passed_count, d.playback_path, d.status "
            "FROM videos d JOIN cameras c ON c.camera_id = d.camera_id"
        ).fetchall()
        for vid, src, cam, label, start, fps, w, h, dur, sc, pc, pb, status in videos:
            frames = []
            for fid, fidx, off, abs_ts, reason, mf, thumb in conn.execute(
                "SELECT frame_id, frame_idx, offset_s, abs_ts, gate_reason, motion_frac, "
                "thumb_path FROM frames WHERE video_id = ? ORDER BY frame_idx", (vid,)
            ).fetchall():
                vecs = tuple(
                    (kind, cls, conf, x1, y1, x2, y2, fx.reconstruct(int(vec_id)).tobytes())
                    for vec_id, kind, cls, conf, x1, y1, x2, y2 in conn.execute(
                        "SELECT vector_id, kind, det_class, det_conf, x1, y1, x2, y2 "
                        "FROM vectors WHERE frame_id = ? ORDER BY vector_id", (fid,)
                    ).fetchall()
                )
                frames.append((fidx, off, abs_ts, reason, mf,
                               _digest(self.cfg.thumbs_dir / thumb), vecs))
            out[Path(src).name] = (
                cam, label, start, fps, w, h, dur, sc, pc, status,
                _digest(self.cfg.playback_dir / pb) if pb else None, tuple(frames),
            )
        return out


@dataclass(frozen=True)
class Failure:
    kind: str  # "error" | "keyboard" | "crash" | "playback_job"
    step: str
    nth: int
    target: int  # index into the sorted list of videos


@st.composite
def scenarios(draw):
    n = draw(st.integers(1, 4))
    seeds = draw(st.lists(st.integers(0, 50), min_size=n, max_size=n, unique=True))
    detector = draw(st.booleans())
    kind = draw(st.sampled_from(["error", "error", "keyboard", "crash", "playback_job"]))
    if kind == "crash":
        step = draw(st.sampled_from(("after_index_save",) * 3 + CHECKPOINTS))
    else:
        step = draw(st.sampled_from(CHECKPOINTS))
    nth = draw(st.integers(1, 4)) if step in PER_FRAME_STEPS else 1
    target = draw(st.integers(0, n - 1))
    return dict(zip(NAMES[:n], seeds)), detector, Failure(kind, step, nth, target)


def _clean_snapshot(root: Path, seeds: dict[str, int], detector: bool) -> dict[str, tuple]:
    env = Env(root, seeds, detector)
    try:
        rep = env.run()
        assert all(r.status == "ingested" for r in rep.results)
        env.assert_consistent()
        return env.snapshot()
    finally:
        env.close()


@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(scenarios())
# Regression: the first video fails after one detector call; the next video must still match.
@example(({NAMES[0]: 0, NAMES[1]: 1}, True, Failure("error", "frame", 2, 0)))
def test_store_and_index_stay_one_to_one_under_failures(scenario) -> None:
    """**Validates: Requirements 6.7, 6.8, 6.10, 7.7, 8.10**"""
    seeds, detector, fail = scenario
    names = sorted(seeds)
    target = names[fail.target]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        clean = _clean_snapshot(Path(tmp) / "clean", seeds, detector)
        assert set(clean) == set(names)

        env = Env(Path(tmp) / "ws", seeds, detector)
        try:
            playback = None
            inj = None
            if fail.kind == "playback_job":
                playback = FakePlayback(fail_for=[target])
            else:
                err = {"error": None, "keyboard": KeyboardInterrupt,
                       "crash": SimulatedCrash}[fail.kind]
                inj = FailureInjector(fail.step, nth=fail.nth, video=target, error=err)

            raised: BaseException | None = None
            report = None
            try:
                report = env.run(checkpoint=inj, playback=playback)
            except (KeyboardInterrupt, SimulatedCrash) as exc:
                raised = exc

            fired = inj.fired if inj is not None else 1
            if fail.kind in ("keyboard", "crash") and fired:
                assert raised is not None, "KeyboardInterrupt / crash must propagate"
            else:
                assert raised is None

            if fail.kind == "crash" and fired:
                env.restart()
                env.pipeline().reconcile()
            env.assert_consistent()

            snap = env.snapshot()
            if fired:
                assert target not in snap, "failed video left rows behind"
                if raised is None:
                    statuses = {r.path.name: r.status for r in report.results}
                    assert statuses[target] == "failed"
                    expected = set(names) - {target}
                else:  # the run stopped at the target; later videos were never processed
                    expected = {n for n in names if n < target}
            else:
                expected = set(names)
            assert set(snap) == expected
            for name in expected:
                assert snap[name] == clean[name], f"{name} differs from a clean run"

            # A later clean run ingests the failed / unprocessed videos (7.7).
            again = env.run()
            assert {r.path.name: r.status for r in again.results} == {
                n: ("already_indexed" if n in expected else "ingested") for n in names
            }
            env.assert_consistent()
            assert env.snapshot() == clean
        finally:
            env.close()


@pytest.mark.parametrize("step", CHECKPOINTS)
def test_crash_at_every_step_is_repaired_by_reconcile(tmp_path: Path, step: str) -> None:
    """Deterministic companion: a crash at any step on the second video is fully repaired."""
    seeds = {NAMES[0]: 1, NAMES[1]: 2}
    clean = _clean_snapshot(tmp_path / "clean", seeds, detector=True)
    env = Env(tmp_path / "ws", seeds, detector=True)
    try:
        with pytest.raises(SimulatedCrash):
            env.run(checkpoint=FailureInjector(step, video=NAMES[1], error=SimulatedCrash))
        env.restart()
        env.pipeline().reconcile()
        env.assert_consistent()
        assert env.snapshot() == {NAMES[0]: clean[NAMES[0]]}
        env.run()
        env.assert_consistent()
        assert env.snapshot() == clean
    finally:
        env.close()
