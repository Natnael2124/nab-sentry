"""Property 27: ingest order does not change the indexed records (task 10.9).

**Validates: Requirements 7.5**

Scope note: Requirement 7.5 compares (camera ID, Absolute_Timestamp, Vector_Kind) records. Under
Requirement 7.3 a file whose bytes duplicate an earlier file is skipped, so if two files had the
same bytes but *different* camera/start metadata (different names), the surviving metadata would
depend on order and 7.3 and 7.5 could not both hold. The generator therefore includes duplicate
content only as byte-identical copies with the same file name in another folder (same metadata);
otherwise every file has distinct content. Videos share cameras and also span several cameras.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nab_sentry.ingest.detector import Detection
from tests.fakes import FakeDetector
from tests.test_pipeline_sanity import Env

CAMERAS = ["CAM-A", "CAM-B", "CAM-C"]


def _detect(frame: np.ndarray, _idx: int) -> list[Detection]:
    """Content-based (not call-order-based) detector: box around the white square."""
    cols = np.flatnonzero(np.asarray(frame).max(axis=(0, 2)) > 200)
    if cols.size == 0:
        return []
    return [Detection("person", 0.8, (int(cols[0]), 10, int(cols[-1]) + 1, 22))]


# One video spec: (camera, start minute offset, has a byte-identical copy in copies/).
video_specs = st.lists(
    st.tuples(st.sampled_from(CAMERAS), st.integers(0, 600), st.booleans()),
    min_size=1, max_size=5, unique_by=lambda t: (t[0], t[1]),
)


def _name(camera: str, minute: int) -> str:
    return f"{camera}_20250101T{8 + minute // 60:02d}{minute % 60:02d}00.mp4"


def _build(env: Env, specs: list[tuple[str, int, bool]]) -> list[Path]:
    paths: list[Path] = []
    for seed, (cam, minute, dup) in enumerate(specs):
        p = env.add_video(_name(cam, minute), seed=seed)
        paths.append(p)
        if dup:
            copy = env.videos_dir / "copies" / p.name
            copy.parent.mkdir(exist_ok=True)
            copy.write_bytes(p.read_bytes())
            paths.append(copy)
    return paths


def _records(env: Env) -> Counter:
    rows = env.db._conn.execute(
        "SELECT v.camera_id, f.abs_epoch_ms, x.kind, x.det_class, x.x1, x.y1, x.x2, x.y2"
        " FROM vectors x JOIN frames f ON f.frame_id = x.frame_id"
        " JOIN videos v ON v.video_id = f.video_id WHERE v.status = 'complete'"
    ).fetchall()
    return Counter(tuple(r) for r in rows)


def _ingest(specs, order: list[int] | None, detector: bool) -> Counter:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        env = Env(Path(d) / "ws", enable_detector=detector)
        try:
            paths = _build(env, specs)
            det = FakeDetector(_detect) if detector else None
            pipe = env.pipeline(detector=det)
            if order is None:
                pipe.ingest_paths(paths)  # sorted order
            else:
                for i in order:  # permuted order, one file at a time (bypasses the sort)
                    pipe.ingest_video(paths[i])
            env.assert_consistent()
            return _records(env)
        finally:
            env.db.close()


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data(), specs=video_specs, detector=st.booleans())
def test_ingest_order_does_not_change_records(data, specs, detector) -> None:
    # Feature: nab-sentry, Property 27: Ingest order does not change the indexed records
    n_files = len(specs) + sum(dup for *_, dup in specs)
    order = data.draw(st.permutations(range(n_files)), label="order")

    baseline = _ingest(specs, None, detector)
    permuted = _ingest(specs, list(order), detector)

    assert baseline, "every generated video should produce at least one record"
    assert permuted == baseline
