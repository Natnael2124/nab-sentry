"""Property 40: Search API results are bounded, ordered, and time-consistent.

**Validates: Requirements 10.4, 10.5, 10.6**
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nab_sentry.api.app import Services, create_app
from nab_sentry.config import Config
from nab_sentry.embed.clip_encoder import EMBED_DIM, l2_normalize
from nab_sentry.querylog import QueryLog
from nab_sentry.search.engine import SearchEngine
from nab_sentry.store.db import MetadataStore, NewFrame, NewVector, NewVideo
from nab_sentry.store.vector_index import VectorIndex
from tests.fakes import BLUE_DIRECTION, RED_DIRECTION, FakeEncoder

EVENT_OUT_FIELDS = {
    "camera_id", "camera_label", "video_id", "start_offset_s", "end_offset_s", "start_time",
    "end_time", "score", "relative_score", "hit_count", "thumbnail_url", "video_url",
}
FPS = 10.0
ONE_MS = timedelta(milliseconds=1)

# Labels: some share words with queries so the Label_Boost path is exercised.
LABELS = ["Reception", "Red Gate", "Blue Car Park", "Loading Bay"]
QUERIES = ["red", "blue", "red gate", "blue car", "a person walking", "loading bay", "reception"]

tzs = st.integers(min_value=-14 * 60, max_value=14 * 60).map(
    lambda m: timezone(timedelta(minutes=m)))
start_times = st.builds(
    # Millisecond precision: Video_Start_Time is stored at ms precision (start_epoch_ms).
    lambda base, tz, ms: base.replace(microsecond=0, tzinfo=tz) + timedelta(milliseconds=ms),
    st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2040, 12, 31)),
    tzs,
    st.integers(min_value=0, max_value=999),
)
vector_kinds = st.sampled_from(["red", "blue", "random"])


@st.composite
def videos(draw):
    duration = draw(st.floats(min_value=1.0, max_value=600.0, allow_nan=False))
    max_idx = int(duration * FPS)
    idxs = draw(st.lists(st.integers(0, max_idx), min_size=1, max_size=12, unique=True))
    frames = [(i, draw(vector_kinds), draw(st.integers(0, 2**31))) for i in sorted(idxs)]
    return {
        "camera": draw(st.integers(0, 3)),
        "start": draw(start_times),
        "duration": duration,
        "frames": frames,
    }


def _vector(kind: str, seed: int) -> np.ndarray:
    if kind == "red":
        return RED_DIRECTION
    if kind == "blue":
        return BLUE_DIRECTION
    v = np.random.default_rng(seed).standard_normal(EMBED_DIM)
    return l2_normalize(v).astype(np.float32)


def _build(root: Path, dataset: list[dict]) -> tuple[Services, dict[int, tuple[datetime, float]]]:
    cfg = Config(root=root)
    cfg.thumbs_dir.mkdir(parents=True)
    cfg.playback_dir.mkdir(parents=True)
    db = MetadataStore(":memory:")
    index = VectorIndex()
    truth: dict[int, tuple[datetime, float]] = {}
    ids, vecs = [], []
    for n, v in enumerate(dataset):
        cam = f"CAM{v['camera']:02d}"
        db.upsert_camera(cam, LABELS[v["camera"]])
        vid = db.insert_video(NewVideo(cam, f"{n}.mp4", f"h{n}", v["start"], FPS, 16, 12,
                                       v["duration"]))
        for idx, kind, seed in v["frames"]:
            off = idx / FPS
            fid = db.insert_frame(NewFrame(vid, idx, off, v["start"] + timedelta(seconds=off),
                                           "motion", 0.5, f"v{vid}_f{idx:07d}.jpg"))
            ids.append(db.insert_vector(NewVector(fid, "frame")))
            vecs.append(_vector(kind, seed))
        db.finalize_video(vid, sampled=len(v["frames"]), passed=len(v["frames"]),
                          duration_s=v["duration"], playback_path=f"v{vid}.mp4")
        truth[vid] = (v["start"], v["duration"])
    index.add(np.array(ids, dtype=np.int64), np.stack(vecs).astype(np.float32))
    enc = FakeEncoder(colour_mode=True)
    engine = SearchEngine(cfg, db, index, enc)
    return Services(cfg, db, index, enc, False, engine, QueryLog(cfg.query_log_path)), truth


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    dataset=st.lists(videos(), min_size=1, max_size=5),
    q=st.sampled_from(QUERIES),
    limit=st.one_of(st.none(), st.integers(min_value=1, max_value=100)),
    camera=st.one_of(st.none(), st.integers(0, 3).map(lambda i: f"CAM{i:02d}")),
)
def test_search_results_bounded_ordered_time_consistent(dataset, q, limit, camera):
    with tempfile.TemporaryDirectory() as tmp:
        services, truth = _build(Path(tmp), dataset)
        try:
            params = {"q": q}
            if limit is not None:
                params["limit"] = str(limit)
            if camera is not None:
                params["camera"] = camera
            r = TestClient(create_app(services, web_dir=Path(tmp) / "noweb")).get(
                "/api/search", params=params)
        finally:
            services.db.close()

    assert r.status_code == 200
    events = r.json()
    assert isinstance(events, list)
    assert len(events) <= (20 if limit is None else limit)  # 10.4 (default 20)

    for ev in events:
        assert set(ev) >= EVENT_OUT_FIELDS  # 10.5
        if camera is not None:
            assert ev["camera_id"] == camera
        assert ev["video_id"] in truth
        video_start, duration = truth[ev["video_id"]]
        s, e = ev["start_offset_s"], ev["end_offset_s"]
        assert 0.0 <= s <= e <= duration + 1e-9  # 10.6 start <= end; offsets within video
        st_ = datetime.fromisoformat(ev["start_time"])
        et_ = datetime.fromisoformat(ev["end_time"])
        assert st_.tzinfo is not None and et_.tzinfo is not None
        assert abs(st_ - (video_start + timedelta(seconds=s))) <= ONE_MS  # 10.6
        assert abs(et_ - (video_start + timedelta(seconds=e))) <= ONE_MS  # 10.6
        assert ev["hit_count"] >= 1
        assert ev["video_url"] == f"/media/video/{ev['video_id']}"

    # 10.4: descending score (Event_Score + Label_Boost); equal scores by earliest absolute start.
    for a, b in zip(events, events[1:]):
        assert a["score"] >= b["score"]
        if a["score"] == b["score"]:
            assert datetime.fromisoformat(a["start_time"]) <= (
                datetime.fromisoformat(b["start_time"]) + ONE_MS)
