"""Unit tests for the ingest pipeline's skip paths (task 10.10).

Requirements 1.5 (unresolved metadata), 1.11 / 2.10 (unreadable / zero passed frames),
2.9 (unknown frame rate), 7.1 (Src_Hash before any write), 7.8 (unhashable file).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

import nab_sentry.ingest.pipeline as pipeline_mod
from nab_sentry.config import Config
from nab_sentry.ingest.pipeline import IngestPipeline
from nab_sentry.ingest.sources import UnknownFrameRate, UnreadableVideo
from nab_sentry.store.db import MetadataStore
from nab_sentry.store.vector_index import VectorIndex
from tests.fakes import FakeEncoder, FakePlayback, FakeSource

W, H, FPS, N = 64, 48, 4.0, 24
GOOD_NAME = "CAM-A_20250101T080000.mp4"


def _moving_frames() -> list[np.ndarray]:
    out = []
    for i in range(N):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        x = ((i // 4) * 9) % (W - 12)
        img[10:22, x:x + 12] = 255
        out.append(img)
    return out


class SpyStore(MetadataStore):
    """Records the order of hash computation (via ``events``) and every write call."""

    WRITES = ("transaction", "upsert_camera", "insert_video", "insert_frame",
              "insert_vector", "finalize_video")

    def __init__(self, path: Path, events: list[str]) -> None:
        super().__init__(path)
        self.events = events

    def __getattribute__(self, name: str):
        attr = super().__getattribute__(name)
        if name in SpyStore.WRITES:
            events = super().__getattribute__("events")

            def wrapper(*a, **kw):
                events.append(f"write:{name}")
                return attr(*a, **kw)

            return wrapper
        return attr


class Env:
    def __init__(self, root: Path) -> None:
        self.cfg = Config(root=root, sample_rate=2.0, keyframe_interval_s=2.0,
                          enable_detector=False, batch_size=3)
        self.videos_dir = root / "videos"
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.events: list[str] = []
        self.db = SpyStore(self.cfg.db_path, self.events)
        self.index = VectorIndex()
        self.playback = FakePlayback()
        self.opened: list[Path] = []
        self.source_factory = lambda path: FakeSource(
            _moving_frames(), fps=FPS, width=W, height=H, path=path)

    def add_video(self, name: str, content: bytes = b"video-bytes") -> Path:
        p = self.videos_dir / name
        p.write_bytes(content)
        return p

    def open_source(self, path: Path):
        self.opened.append(Path(path))
        return self.source_factory(path)

    def pipeline(self) -> IngestPipeline:
        return IngestPipeline(
            self.cfg, self.db, self.index, FakeEncoder(batch_size=self.cfg.batch_size), None,
            open_source=self.open_source, start_playback=self.playback,
        )

    def writes(self) -> list[str]:
        return [e for e in self.events if e.startswith("write:")]

    def assert_nothing_written(self) -> None:
        c = self.db.counts()
        assert (c.cameras, c.videos, c.frames, c.vectors) == (0, 0, 0, 0)
        assert self.index.ids().size == 0
        assert not self.cfg.index_path.exists()
        for d in (self.cfg.thumbs_dir, self.cfg.playback_dir):
            assert not d.exists() or not any(d.iterdir())


@pytest.fixture
def env(tmp_path: Path):
    e = Env(tmp_path / "ws")
    yield e
    e.db.close()


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.DEBUG, logger="nab_sentry")
    return caplog


# -- 1.5: unresolved metadata --------------------------------------------------------------------


def test_unresolved_metadata_is_logged_and_skipped_without_writes(env: Env, logs) -> None:
    path = env.add_video("random_clip.mp4")  # no sidecar, name does not match the pattern
    result = env.pipeline().ingest_video(path)

    assert result.status == "unresolved_metadata"
    assert result.video_id is None
    assert env.opened == []  # never decoded
    assert env.writes() == []
    assert env.playback.jobs == []
    assert any(
        "unresolved camera metadata" in r.getMessage() and str(path) in r.getMessage()
        for r in logs.records
    )
    env.assert_nothing_written()


# -- 1.11 / 2.9: unreadable video and unknown frame rate -----------------------------------------


@pytest.mark.parametrize(
    ("error", "reason", "logged"),
    [
        (UnreadableVideo("cannot open"), "unreadable video", "unreadable video"),
        (UnknownFrameRate("fps=0"), "unknown frame rate", "unknown frame rate"),
    ],
)
def test_open_failure_is_logged_and_skipped(env: Env, logs, error, reason, logged) -> None:
    path = env.add_video(GOOD_NAME)

    def boom(p: Path):
        raise error

    env.source_factory = boom
    result = env.pipeline().ingest_video(path)

    assert result.status == "unreadable"
    assert result.reason == reason
    assert env.opened == [path]
    assert env.writes() == []
    assert env.playback.jobs == []
    assert any(logged in r.getMessage() and str(path) in r.getMessage() for r in logs.records)
    env.assert_nothing_written()


# -- 7.1: Src_Hash before any write ---------------------------------------------------------------


def test_src_hash_computed_before_any_write(env: Env, monkeypatch) -> None:
    real = pipeline_mod.src_hash

    def spy(path, *a, **kw):
        env.events.append("hash")
        return real(path, *a, **kw)

    monkeypatch.setattr(pipeline_mod, "src_hash", spy)
    path = env.add_video(GOOD_NAME)
    result = env.pipeline().ingest_video(path)

    assert result.status == "ingested"
    assert env.events.count("hash") == 1
    assert env.writes(), "a successful ingest writes rows"
    assert env.events[0] == "hash"
    assert env.events.index("hash") < env.events.index("write:transaction")


# -- 7.8: unhashable file -------------------------------------------------------------------------


def test_unhashable_file_is_logged_and_skipped(env: Env, logs, monkeypatch) -> None:
    def unreadable(path, *a, **kw):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(pipeline_mod, "src_hash", unreadable)
    path = env.add_video(GOOD_NAME)
    result = env.pipeline().ingest_video(path)

    assert result.status == "failed"
    assert "Permission denied" in (result.reason or "")
    assert env.opened == []
    assert env.writes() == []
    assert any(
        r.levelno >= logging.ERROR and str(path) in r.getMessage() for r in logs.records
    )
    env.assert_nothing_written()


def test_unhashable_file_does_not_abort_the_run(env: Env, monkeypatch) -> None:
    real = pipeline_mod.src_hash
    bad = env.add_video("CAM-A_20250101T070000.mp4", content=b"bad")
    good = env.add_video(GOOD_NAME, content=b"good")

    def flaky(path, *a, **kw):
        if Path(path) == bad:
            raise OSError("read error")
        return real(path, *a, **kw)

    monkeypatch.setattr(pipeline_mod, "src_hash", flaky)
    report = env.pipeline().ingest_paths([env.videos_dir])

    assert {r.path: r.status for r in report.results} == {bad: "failed", good: "ingested"}
    assert env.db.counts().videos == 1


# -- 1.11 / 2.10: zero passed frames are rolled back ----------------------------------------------


@pytest.mark.parametrize(
    "make_source",
    [
        pytest.param(lambda p: FakeSource([], fps=FPS, width=W, height=H, path=p), id="no-frames"),
        pytest.param(
            lambda p: FakeSource(n_frames=N, empty=range(N), fps=FPS, width=W, height=H, path=p),
            id="all-empty",
        ),
        pytest.param(
            lambda p: FakeSource(n_frames=N, undecodable=range(N), fps=FPS, width=W, height=H,
                                 path=p),
            id="all-undecodable",
        ),
    ],
)
def test_zero_passed_frames_is_unreadable_and_rolled_back(env: Env, logs, make_source) -> None:
    # A committed video first, so "unchanged" is checked against a non-empty state.
    env.add_video(GOOD_NAME, content=b"good")
    assert env.pipeline().ingest_video(env.videos_dir / GOOD_NAME).status == "ingested"
    before_counts = env.db.counts()
    before_ids = set(env.index.ids().tolist())
    before_disk = set(VectorIndex.load(env.cfg.index_path).ids().tolist())
    before_files = {p.name for d in (env.cfg.thumbs_dir, env.cfg.playback_dir)
                    for p in d.iterdir()}

    path = env.add_video("CAM-B_20250101T090000.mp4", content=b"zero")
    env.source_factory = make_source
    result = env.pipeline().ingest_video(path)

    assert result.status == "unreadable"
    assert result.video_id is None
    assert env.db.counts() == before_counts
    assert env.db.camera_label("CAM-B") is None
    assert set(env.index.ids().tolist()) == before_ids
    assert set(VectorIndex.load(env.cfg.index_path).ids().tolist()) == before_disk
    after_files = {p.name for d in (env.cfg.thumbs_dir, env.cfg.playback_dir)
                   for p in d.iterdir()}
    assert after_files == before_files
    assert env.playback.jobs[-1].killed
    assert any("unreadable video" in r.getMessage() and str(path) in r.getMessage()
               for r in logs.records)
