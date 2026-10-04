"""Fast sanity checks for the ingest pipeline (task 10.3).

The full properties (7, 11, 17, 24, 26, 27) are tested in tasks 10.4-10.10; these tests only
exercise the main paths: success, re-ingest skip, rollback on failure, crash + reconcile,
reconcile of orphans, and the optional detector.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from nab_sentry.config import Config, ConfigError
from nab_sentry.ingest.detector import Detection
from nab_sentry.ingest.pipeline import IngestPipeline
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

W, H, FPS, N = 64, 48, 4.0, 24  # 6 s of video


def _frames(seed: int) -> list[np.ndarray]:
    """A white square that jumps every 4 frames, so motion passes some frames, not all."""
    out = []
    for i in range(N):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        x = (seed * 7 + (i // 4) * 9) % (W - 12)
        img[10:22, x:x + 12] = 255
        out.append(img)
    return out


class Env:
    def __init__(self, root: Path, **cfg_kw: object) -> None:
        kw = dict(sample_rate=2.0, keyframe_interval_s=2.0, enable_detector=False, batch_size=3)
        kw.update(cfg_kw)
        self.cfg = Config(root=root, **kw)  # type: ignore[arg-type]
        self.videos_dir = root / "videos"
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.sources: dict[str, int] = {}
        self.db = MetadataStore(self.cfg.db_path)
        self.index = VectorIndex()
        self.playback = FakePlayback()

    def add_video(self, name: str, seed: int, content: bytes | None = None) -> Path:
        p = self.videos_dir / name
        p.write_bytes(content if content is not None else f"video-{seed}".encode())
        self.sources[name] = seed
        return p

    def open_source(self, path: Path) -> FakeSource:
        seed = self.sources[Path(path).name]
        return FakeSource(_frames(seed), fps=FPS, width=W, height=H, path=path)

    def pipeline(self, *, detector=None, checkpoint=None) -> IngestPipeline:
        return IngestPipeline(
            self.cfg, self.db, self.index, FakeEncoder(batch_size=self.cfg.batch_size),
            detector, open_source=self.open_source, start_playback=self.playback,
            checkpoint=checkpoint,
        )

    def disk_index_ids(self) -> set[int]:
        if not self.cfg.index_path.exists():
            return set()
        return set(VectorIndex.load(self.cfg.index_path).ids().tolist())

    def assert_consistent(self) -> None:
        db_ids = set(self.db.vector_ids().tolist())
        assert set(self.index.ids().tolist()) == db_ids
        assert self.disk_index_ids() == db_ids
        thumbs = {p.name for p in self.cfg.thumbs_dir.glob("*")}
        plays = {p.name for p in self.cfg.playback_dir.glob("*")}
        assert thumbs | plays == self.db.referenced_files()


@pytest.fixture
def env(tmp_path: Path):
    e = Env(tmp_path / "ws")
    yield e
    e.db.close()


def test_successful_ingest_writes_rows_files_and_index(env: Env) -> None:
    env.add_video("CAM-A_20250101T080000.mp4", seed=1)
    report = env.pipeline().ingest_paths([env.videos_dir])

    assert [r.status for r in report.results] == ["ingested"]
    r = report.results[0]
    assert 1 <= r.passed <= r.sampled == 12  # 6 s at 2 fps -> 12 or 13 targets, 12 frames
    c = env.db.counts()
    assert (c.cameras, c.videos, c.frames, c.vectors) == (1, 1, r.passed, r.passed)
    assert r.vectors == r.passed  # frame vectors only (detector disabled)
    assert len(list(env.cfg.thumbs_dir.glob("*.jpg"))) == r.passed
    assert env.db.playback_for(r.video_id) == f"v{r.video_id}.mp4"
    assert (env.cfg.playback_dir / f"v{r.video_id}.mp4").is_file()
    env.assert_consistent()


def test_reingest_and_duplicate_content_are_skipped(env: Env, caplog) -> None:
    env.add_video("CAM-A_20250101T080000.mp4", seed=1)
    env.add_video("CAM-B_20250101T090000.mp4", seed=2, content=b"video-1")  # same bytes
    first = env.pipeline().ingest_paths([env.videos_dir])
    assert sorted(r.status for r in first.results) == ["already_indexed", "ingested"]
    before = env.db.counts()

    caplog.set_level(logging.INFO, logger="nab_sentry")
    second = env.pipeline().ingest_paths([env.videos_dir])
    assert [r.status for r in second.results] == ["already_indexed"] * 2
    assert env.db.counts() == before
    assert sum("already indexed" in m for m in caplog.messages) == 2
    env.assert_consistent()


def test_config_error_before_any_write(env: Env) -> None:
    env.add_video("CAM-A_20250101T080000.mp4", seed=1)
    bad = Env(env.cfg.root, sample_rate=99.0)
    try:
        with pytest.raises(ConfigError, match="sample_rate"):
            bad.pipeline().ingest_paths([env.videos_dir])
        assert bad.db.counts().vectors == 0
        assert not bad.cfg.index_path.exists()
    finally:
        bad.db.close()


@pytest.mark.parametrize(
    "step", ["thumbnail", "embed", "playback_wait", "index_save", "after_index_save", "commit"]
)
def test_failure_rolls_back_only_the_failed_video(env: Env, step: str, caplog) -> None:
    env.add_video("CAM-A_20250101T080000.mp4", seed=1)
    env.add_video("CAM-B_20250101T090000.mp4", seed=2)
    inj = FailureInjector(step, video="CAM-B_20250101T090000.mp4")
    report = env.pipeline(checkpoint=inj).ingest_paths([env.videos_dir])

    assert inj.fired == 1
    assert [r.status for r in report.results] == ["ingested", "failed"]
    assert env.db.counts().videos == 1
    assert env.db.camera_label("CAM-B") is None
    assert "CAM-B_20250101T090000.mp4" in " ".join(caplog.messages)
    env.assert_consistent()

    # A later run ingests the failed video again (7.7).
    again = env.pipeline().ingest_paths([env.videos_dir])
    assert [r.status for r in again.results] == ["already_indexed", "ingested"]
    env.assert_consistent()


def test_crash_after_index_save_is_repaired_by_reconcile(env: Env) -> None:
    env.add_video("CAM-A_20250101T080000.mp4", seed=1)
    env.add_video("CAM-B_20250101T090000.mp4", seed=2)
    inj = FailureInjector.crash_after_index_save(video="CAM-B_20250101T090000.mp4")
    with pytest.raises(SimulatedCrash):
        env.pipeline(checkpoint=inj).ingest_paths([env.videos_dir])

    # "Restart": reopen the DB and reload the index from disk.
    env.db.close()
    env.db = MetadataStore(env.cfg.db_path)
    env.index = VectorIndex.load(env.cfg.index_path)
    assert env.disk_index_ids() > set(env.db.vector_ids().tolist())  # orphans on disk

    rep = env.pipeline().reconcile()
    assert rep.removed_index_ids and rep.deleted_files
    env.assert_consistent()
    assert env.db.counts().videos == 1

    again = env.pipeline().ingest_paths([env.videos_dir])
    assert [r.status for r in again.results] == ["already_indexed", "ingested"]
    env.assert_consistent()


def test_reconcile_removes_orphans_and_incomplete_videos(env: Env) -> None:
    env.add_video("CAM-A_20250101T080000.mp4", seed=1)
    env.add_video("CAM-B_20250101T090000.mp4", seed=2)
    pipe = env.pipeline()
    pipe.ingest_paths([env.videos_dir])
    vid_a, vid_b = env.db.video_ids(status="complete")

    # Orphan FAISS ID, orphan thumbnail, and video B's vectors missing from the index.
    env.index.add(np.array([999_999]), np.eye(1, 512, dtype=np.float32))
    (env.cfg.thumbs_dir / "v4242_f0000000.jpg").write_bytes(b"x")
    env.index.remove(env.db.vector_ids_for_video(vid_b)[:1])

    rep = pipe.reconcile()
    assert 999_999 in rep.removed_index_ids
    assert rep.deleted_video_ids == [vid_b]
    assert "v4242_f0000000.jpg" in rep.deleted_files
    assert env.db.video_ids() == [vid_a]
    env.assert_consistent()
    assert not pipe.reconcile().changed


def test_detector_crops_and_detector_errors(env: Env) -> None:
    env2 = Env(env.cfg.root.parent / "ws2", enable_detector=True)
    try:
        env2.add_video("CAM-A_20250101T080000.mp4", seed=1)
        dets = [Detection("person", 0.9, (2, 3, 20, 30)), Detection("car", 0.5, (60, 40, 64, 48))]
        detector = FakeDetector(lambda frame, i: dets, raise_on={0})
        report = env2.pipeline(detector=detector).ingest_paths([env2.videos_dir])
        r = report.results[0]
        assert r.status == "ingested"
        # First passed frame: detector raised -> no crops; every other frame: 2 crops.
        assert r.vectors == r.passed + 2 * (r.passed - 1)
        env2.assert_consistent()
    finally:
        env2.db.close()
