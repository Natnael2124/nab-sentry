"""Property 11, pipeline half: ``ingest_paths`` refuses an invalid Config before any write.

The ``validate()`` half lives in ``tests/test_config.py``; the generator here mirrors it.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nab_sentry.config import Config, ConfigError
from nab_sentry.ingest.pipeline import IngestPipeline
from nab_sentry.store.db import MetadataStore
from nab_sentry.store.vector_index import VectorIndex
from tests.fakes import FakeEncoder, FakePlayback, FakeSource

# name -> (lo, hi, integer) for the parameters named in Property 11.
RANGES: dict[str, tuple[float, float, bool]] = {
    "sample_rate": (0.1, 30, False),
    "motion_threshold": (0.0, 1.0, False),
    "keyframe_interval_s": (1, 600, False),
    "gate_width": (64, 1920, True),
    "det_conf": (0.0, 1.0, False),
    "det_max_per_frame": (1, 100, True),
    "batch_size": (1, 64, True),
}


def valid_value(name: str) -> st.SearchStrategy[object]:
    lo, hi, integer = RANGES[name]
    if integer:
        return st.integers(min_value=int(lo), max_value=int(hi))
    return st.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False)


def invalid_value(name: str) -> st.SearchStrategy[object]:
    lo, hi, integer = RANGES[name]
    if integer:
        below = st.integers(max_value=int(lo) - 1)
        above = st.integers(min_value=int(hi) + 1)
    else:
        below = st.floats(max_value=lo, exclude_max=True, allow_nan=False, allow_infinity=False)
        above = st.floats(min_value=hi, exclude_min=True, allow_nan=False, allow_infinity=False)
    return st.one_of(below, above, st.just(math.nan), st.text(max_size=12))


@st.composite
def bad_config_kwargs(draw: st.DrawFn) -> tuple[dict[str, object], set[str]]:
    """Kwargs for the seven parameters, with a non-empty subset made invalid."""
    names = sorted(RANGES)
    bad = draw(st.sets(st.sampled_from(names), min_size=1))
    kwargs = {n: draw(invalid_value(n) if n in bad else valid_value(n)) for n in names}
    return kwargs, bad


class SpyEncoder(FakeEncoder):
    def __init__(self) -> None:
        super().__init__(batch_size=4)
        self.calls = 0

    def encode_images(self, images):  # type: ignore[override]
        self.calls += 1
        return super().encode_images(images)


class SpyPlayback(FakePlayback):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __call__(self, src, dst):  # type: ignore[override]
        self.calls += 1
        return super().__call__(src, dst)


def _frames() -> list[np.ndarray]:
    out = []
    for i in range(8):
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        img[10:22, (i * 6) % 50:(i * 6) % 50 + 12] = 255
        out.append(img)
    return out


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(bad_config_kwargs())
def test_property_11_ingest_refuses_invalid_config_before_any_write(
    case: tuple[dict[str, object], set[str]],
) -> None:
    """Feature: nab-sentry, Property 11: Config validation names every out-of-range parameter.

    Pipeline half: ``ingest_paths`` raises ``ConfigError`` naming exactly the invalid subset
    before creating any row, file, or index entry.

    **Validates: Requirements 2.8, 3.8, 4.9**
    """
    kwargs, bad = case
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "ws"
        cfg = Config(root=root, enable_detector=False, **kwargs)  # type: ignore[arg-type]
        videos = root / "videos"
        videos.mkdir(parents=True)
        (videos / "CAM-A_20250101T080000.mp4").write_bytes(b"video-1")

        opened: list[Path] = []

        def open_source(path: Path) -> FakeSource:
            opened.append(Path(path))
            return FakeSource(_frames(), fps=4.0, width=64, height=48, path=path)

        encoder = SpyEncoder()
        playback = SpyPlayback()
        db = MetadataStore(cfg.db_path)  # creating the store makes the DB file; that's fine
        try:
            index = VectorIndex()
            pipe = IngestPipeline(
                cfg, db, index, encoder, None, open_source=open_source, start_playback=playback,
            )
            with pytest.raises(ConfigError) as exc:
                pipe.ingest_paths([videos])

            assert {i.name for i in exc.value.issues} == bad
            assert sorted(i.name for i in exc.value.issues) == sorted(bad)  # each named once
            for name in bad:
                assert name in str(exc.value)

            # Nothing written: no rows, no index entries, no index file, no media files.
            c = db.counts()
            assert (c.cameras, c.videos, c.frames, c.vectors) == (0, 0, 0, 0)
            assert db.video_ids() == []
            assert db.vector_ids().size == 0
            assert index.ids().size == 0
            assert not cfg.index_path.exists()
            for d in (cfg.thumbs_dir, cfg.playback_dir):
                assert not d.exists() or not any(d.iterdir())

            # No collaborator was touched.
            assert opened == []
            assert encoder.calls == 0
            assert playback.calls == 0 and playback.jobs == []
        finally:
            db.close()
