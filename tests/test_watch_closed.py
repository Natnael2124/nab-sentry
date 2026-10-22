"""Watch mode (V2 Requirement 8): Property 8 for ``closed_segments`` plus Watcher retries."""

from __future__ import annotations

from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.ingest.pipeline import VideoResult
from nab_sentry.ingest.watch import (
    SegmentFile,
    Watcher,
    closed_segments,
    parse_listing,
)

SEG_S = 10.0


def ts(sec: int) -> str:
    h, m, s = sec // 3600, sec // 60 % 60, sec % 60
    return f"20260101T{h:02d}{m:02d}{s:02d}"


def seg(cam: str, sec: int, size: int = 100, mtime: float = 0.0) -> SegmentFile:
    return SegmentFile(Path(f"{cam}_{ts(sec)}.mkv"), cam, ts(sec), size, mtime)


@st.composite
def recordings(draw):
    """Several cameras, each a run of consecutive Segments; the newest one may still grow."""
    files, now = [], 10_000.0
    for cam in draw(st.lists(st.sampled_from(["CAM01", "CAM02", "CAM03"]), min_size=1, max_size=3, unique=True)):
        n = draw(st.integers(1, 6))
        start = draw(st.integers(0, 3000))
        for i in range(n):
            newest = i == n - 1
            age = draw(st.floats(0, 4)) if newest else draw(st.floats(15, 100))
            files.append(seg(cam, start + 10 * i, draw(st.integers(1, 10**6)), now - age))
    return files, now


@given(recordings())
def test_newest_growing_file_is_never_returned(rec):
    files, now = rec
    closed = closed_segments(files, now, lambda c: SEG_S, {}, set())
    for cam in {f.camera_id for f in files}:
        newest = max((f for f in files if f.camera_id == cam), key=lambda f: f.ts)
        assert newest not in closed
        older = [f for f in files if f.camera_id == cam and f is not newest]
        assert all(f in closed for f in older)


@given(recordings())
def test_order_per_camera_and_no_repeats_across_polls(rec):
    files, now = rec
    done: set[Path] = set()
    first = closed_segments(files, now, lambda c: SEG_S, {}, done)
    for cam in {f.camera_id for f in first}:
        mine = [f.ts for f in first if f.camera_id == cam]
        assert mine == sorted(mine)
    done |= {f.path for f in first}
    sizes = {f.path: f.size for f in files}
    later = closed_segments(files, now + 60, lambda c: SEG_S, sizes, done)
    assert not ({f.path for f in later} & {f.path for f in first})


def test_idle_file_closes_only_after_two_polls_with_same_size():
    f = seg("CAM01", 0, size=500, mtime=0.0)
    assert closed_segments([f], 100.0, lambda c: SEG_S, {}, set()) == []           # first sighting
    assert closed_segments([f], 100.0, lambda c: SEG_S, {f.path: 400}, set()) == []  # still growing
    assert closed_segments([f], 100.0, lambda c: SEG_S, {f.path: 500}, set()) == [f]
    assert closed_segments([f], 14.0, lambda c: SEG_S, {f.path: 500}, set()) == []  # mtime too fresh


def test_parse_listing_filters_names_and_extensions():
    entries = [(Path("CAM01_20260101T000000.mkv"), 1, 0.0), (Path("CAM01_20260101T000000.json"), 1, 0.0),
               (Path("random.mkv"), 1, 0.0), (Path("CAM-2_20260101T000010.MP4"), 1, 0.0)]
    assert [s.camera_id for s in parse_listing(entries)] == ["CAM01", "CAM-2"]


class FakePipeline:
    def __init__(self, outcomes):
        self.outcomes, self.calls = outcomes, []

    def ingest_video(self, path):
        self.calls.append(path.name)
        status = self.outcomes.get(path.name, ["ingested"]).pop(0) if self.outcomes.get(path.name) else "ingested"
        return VideoResult(path, status)


def test_watcher_retries_failures_then_gives_up():
    a, b, c = seg("CAM01", 0), seg("CAM01", 10), seg("CAM01", 20, mtime=999.0)
    pipe = FakePipeline({a.path.name: ["failed", "failed", "failed"]})
    w = Watcher(pipe, Path("."), lambda cam: SEG_S, scan=lambda d: [a, b, c], clock=lambda: 1000.0)
    for _ in range(4):
        w.poll_once()
    # a: 3 attempts (1 + 2 retries) then given up; b once; c (newest, fresh) never.
    assert pipe.calls.count(a.path.name) == 3
    assert pipe.calls.count(b.path.name) == 1
    assert c.path.name not in pipe.calls
    assert a.path in w.done and b.path in w.done
