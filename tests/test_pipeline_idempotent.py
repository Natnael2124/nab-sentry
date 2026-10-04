"""Property 26: re-ingest is idempotent and incremental ingest matches full ingest (task 10.8).

**Validates: Requirements 7.3, 7.4, 7.6**
"""

from __future__ import annotations

import logging
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.test_pipeline_sanity import Env


@dataclass(frozen=True)
class FakeFile:
    name: str
    seed: int
    content: bytes


@st.composite
def video_sets(draw) -> tuple[list[FakeFile], list[bool]]:
    """Distinct contents, each copied 1-3 times (duplicates share camera, bytes, and frames)."""
    n_contents = draw(st.integers(1, 4))
    files: list[FakeFile] = []
    for k in range(n_contents):
        cam = draw(st.sampled_from(["A", "B", "C"]))
        copies = draw(st.integers(1, 3))
        for _ in range(copies):
            hour = len(files)  # unique timestamp per file, no overlaps
            name = f"CAM-{cam}_202501{1 + hour // 24:02d}T{hour % 24:02d}0000.mp4"
            files.append(FakeFile(name, seed=k, content=f"video-{k}".encode()))
    subset = draw(st.lists(st.booleans(), min_size=len(files), max_size=len(files)))
    return files, subset


@contextmanager
def workspace(files: list[FakeFile]):
    with tempfile.TemporaryDirectory() as tmp:
        env = Env(Path(tmp) / "ws")
        try:
            for f in files:
                env.add_video(f.name, seed=f.seed, content=f.content)
            yield env
        finally:
            env.db.close()


def snapshot(env: Env) -> tuple:
    c = env.db.counts()
    return (
        c.cameras, c.videos, c.frames, c.vectors,
        env.index.ntotal,
        sum(1 for p in env.cfg.thumbs_dir.glob("*") if p.is_file()),
        sum(1 for p in env.cfg.playback_dir.glob("*") if p.is_file()),
    )


def file_paths(env: Env, files: list[FakeFile]) -> list[Path]:
    return [env.videos_dir / f.name for f in files]


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(video_sets())
def test_reingest_is_idempotent_and_incremental_matches_full(data) -> None:
    files, subset = data
    part = [f for f, keep in zip(files, subset) if keep]
    n_contents = len({f.content for f in files})

    # (a) Full ingest into an empty workspace, then re-ingest the same set.
    with workspace(files) as full:
        first = full.pipeline().ingest_paths(file_paths(full, files))
        statuses = [r.status for r in first.results]
        assert statuses.count("ingested") == n_contents
        assert statuses.count("already_indexed") == len(files) - n_contents
        full_snap = snapshot(full)
        assert full_snap[1] == n_contents

        logger = logging.getLogger("nab_sentry")
        records: list[str] = []

        class _Grab(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record.getMessage())

        handler, old_level = _Grab(level=logging.INFO), logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            second = full.pipeline().ingest_paths(file_paths(full, files))
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)

        assert [r.status for r in second.results] == ["already_indexed"] * len(files)
        assert sum("already indexed" in m for m in records) == len(files)
        assert snapshot(full) == full_snap
        full.assert_consistent()

    # (b) Subset first, then the full set, in a separate empty workspace.
    with workspace(files) as inc:
        if part:
            inc.pipeline().ingest_paths(file_paths(inc, part))
        inc.pipeline().ingest_paths(file_paths(inc, files))
        assert snapshot(inc) == full_snap
        inc.assert_consistent()
