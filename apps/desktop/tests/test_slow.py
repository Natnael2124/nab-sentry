"""Slow tests: real FFmpeg, real models (models/desktop), the V2 test video. Run with ``-m slow``."""

from __future__ import annotations

from datetime import datetime

import cv2
import numpy as np
import pytest

from conftest import MODELS, VIDEO, need
from nab_find.library import add_videos
from nab_find.media import FrameReader, cut_clip, grab_jpeg, probe
from nab_find.processor import Processor, merge
from nab_find.search import Searcher
from nab_find.store import CaseStore

pytestmark = pytest.mark.slow
T0 = datetime(2026, 10, 7, 23, 0, 10).timestamp()  # the video's start, from its file name


@pytest.fixture(scope="module")
def models():
    need(MODELS / "clip_image.xml")
    need(VIDEO)
    from nab_find.models import load_models

    return load_models(MODELS, prefer="CPU")


def test_reader_modes_and_clip(tmp_path):
    need(VIDEO)
    info = probe(VIDEO)
    assert 387 < info.duration_s < 388 and info.width == 1376 and 4 < info.keyframe_gap_s < 5
    with FrameReader(VIDEO, info, 60, 80, mode="rate", rate=2.0) as r:
        offs = [o for o, _ in r.frames()]
    assert len(offs) == 40 and offs[0] == 60 and offs[-1] == pytest.approx(79.5)
    with FrameReader(VIDEO, info, 60, 120, mode="keyframes") as r:
        kf = [o for o, img in r.frames()]
    assert 12 <= len(kf) <= 15 and all(60 <= o < 121 for o in kf) and kf == sorted(kf)
    out = cut_clip(VIDEO, 100, 5, tmp_path / "c.mp4", overlay="Front  07 Oct 23:01:50")
    cap = cv2.VideoCapture(str(out))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert 4.5 < n / fps < 5.5
    img = cv2.imdecode(np.frombuffer(grab_jpeg(VIDEO, 100), np.uint8), 1)
    assert max(img.shape[:2]) == 384


def test_process_range_search_and_resume(tmp_path, models):
    st = CaseStore.create(tmp_path, "slow")
    assert add_videos(st, [VIDEO])[0].video_id == 1
    st.set_meta(range=[T0 + 20, T0 + 200])
    p = Processor(st, models)
    commits = {"n": 0}
    orig = st.commit

    def stop_after_two(chunk):
        out = orig(chunk)
        commits["n"] += 1
        if commits["n"] == 2:
            p.cancel()
        return out

    st.commit = stop_after_two
    assert p.run() == "cancelled"
    st.commit = orig
    assert Processor(st, models).run() == "done"
    assert merge(st.spans(1, "quick"), gap=0.01) == [(20, 200)]
    offs = [r[0] for r in st.query("SELECT offset_s FROM frames")]
    assert len(offs) == len(set(offs)) and min(offs) >= 20 and max(offs) <= 200
    assert st.counts()["visits"] > 10
    s = Searcher(st, models.clip)
    # "police officers at a traffic stop" is labelled 23:00:34-23:00:42 (offset 24-32 s)
    top = s.search("police officers at a traffic stop", limit=3)
    assert any(m.start_ts <= T0 + 32 and m.end_ts >= T0 + 24 for m in top)
    people = s.search("a person", __import__("nab_find.search", fromlist=["Filters"]).Filters(kind="people"))
    assert people and all(m.cls == "person" for m in people)
    visit = people[0].visit_id
    assert all(m.visit_id != visit for m in s.similar(visit))
