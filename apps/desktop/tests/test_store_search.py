"""Case store, search and the processor with fake models and a fake frame source."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FakeClip, unit
from nab_find import processor as proc_mod
from nab_find.processor import Processor
from nab_find.search import Filters, Searcher
from nab_find.store import CaseStore, Chunk, CropRow, FrameRow, VisitRow

from nab_sentry.ingest.detector import Detection

T0 = 1_800_000_000.0  # a clock time for video starts


def vec(i: int) -> np.ndarray:
    v = np.zeros(512, np.float32)
    v[i] = 1
    return v


def new_store(tmp_path) -> CaseStore:
    st = CaseStore.create(tmp_path, "case")
    st.add_video("a.mp4", "Front", T0, False, 600, 25, 1280, 720, 1.0)
    return st


def test_commit_reload_and_counts(tmp_path):
    st = new_store(tmp_path)
    keymap = st.commit(Chunk(1, "detail", (0, 60),
                             frames=[FrameRow(1, 5.0, "f.jpg", vec(0))],
                             crops=[CropRow(1, 6.0, 7, (1, 2, 30, 80), "c.jpg", vec(1))],
                             visits=[VisitRow(7, 1, "person", 5, 9, 4, 0.1, [(5, 0.5, 0.5)], {"phrase": "person"}, "c.jpg")],
                             motion=[(4, 10)]))
    assert st.counts() == {"frames": 1, "visits": 1, "vectors": 2}
    assert st.spans(1, "detail") == [(0, 60)] and st.motion(1) == [(4, 10)]
    folder = st.folder
    st.close()
    st2 = CaseStore(folder)
    t = st2.table.view()
    assert t["ids"].tolist() == [1, 2] and t["visit"].tolist() == [-1, keymap[7]]
    assert np.allclose(t["vecs"][1], vec(1))


def test_commit_rollback_keeps_store_unchanged(tmp_path):
    st = new_store(tmp_path)
    bad = Chunk(1, "detail", (0, 60), crops=[CropRow(1, 6.0, 99, (0, 0, 9, 9), "c.jpg", vec(1))],
                visits=[VisitRow(5, 1, "person", 0, 1, 1, 0, [], {}, None)])
    with pytest.raises(KeyError):  # crop of an unknown visit key
        st.commit(bad)
    assert st.counts() == {"frames": 0, "visits": 0, "vectors": 0}
    assert st.table.n == 0 and st.spans(1, "detail") == []
    # key 5 must not map to the rolled-back visit row
    st.commit(Chunk(1, "detail", (0, 1), visits=[VisitRow(5, 1, "car", 0, 1, 1, 0, [], {}, "x.jpg")]))
    assert st.query("SELECT cls FROM visits") == [("car",)]


def test_visit_upsert_across_chunks(tmp_path):
    st = new_store(tmp_path)
    st.commit(Chunk(1, "detail", (0, 60), visits=[VisitRow(3, 1, "person", 10, 20, 5, 0, [], {}, "a.jpg")]))
    st.commit(Chunk(1, "detail", (60, 120), visits=[VisitRow(3, 1, "person", 10, 70, 9, 0, [], {}, None)]))
    assert st.query("SELECT t0, t1, n_obs, thumb FROM visits") == [(10, 70, 9, "a.jpg")]


def test_reset_and_remove(tmp_path):
    st = new_store(tmp_path)
    st.commit(Chunk(1, "quick", (0, 60), frames=[FrameRow(1, 5.0, "f.jpg", vec(0))]))
    st.reset_work()
    assert st.table.n == 0 and st.counts()["frames"] == 0 and len(st.videos()) == 1
    st.commit(Chunk(1, "quick", (0, 60), frames=[FrameRow(1, 5.0, "f.jpg", vec(0))]))
    st.remove_video(1)
    assert st.videos() == [] and st.table.n == 0


# ---------------------------------------------------------------- search


def seeded(tmp_path) -> CaseStore:
    st = new_store(tmp_path)
    st.add_video("b.mp4", "Back", T0 + 1000, False, 600, 25, 1280, 720, 1.0)
    st.commit(Chunk(1, "quick", (0, 600), frames=[
        FrameRow(1, 10.0, "f10.jpg", unit(vec(0) + 0.1 * vec(5))),
        FrameRow(1, 14.0, "f14.jpg", unit(vec(0) + 0.2 * vec(5))),  # 4 s later: same moment
        FrameRow(1, 100.0, "f100.jpg", unit(vec(0) + 0.5 * vec(5))),
    ]))
    st.commit(Chunk(1, "detail", (0, 600), crops=[
        CropRow(1, 30.0, 1, (0, 0, 50, 100), "p1.jpg", unit(vec(1) + 0.05 * vec(2))),
        CropRow(1, 33.0, 1, (0, 0, 50, 100), "p1b.jpg", unit(vec(1) + 0.1 * vec(2))),
        CropRow(1, 300.0, 2, (0, 0, 50, 100), "p2.jpg", unit(vec(1) + 0.3 * vec(2))),
        CropRow(1, 400.0, 3, (0, 0, 80, 40), "car.jpg", unit(vec(3))),
    ], visits=[VisitRow(1, 1, "person", 30, 40, 5, 0.1, [], {"phrase": "person in a red top"}, "p1.jpg"),
               VisitRow(2, 1, "person", 300, 302, 2, 0.6, [], {}, "p2.jpg"),
               VisitRow(3, 1, "car", 390, 900, 50, 0.0, [], {}, "car.jpg")]))
    st.commit(Chunk(2, "quick", (0, 600), frames=[FrameRow(2, 50.0, "b50.jpg", unit(vec(0) + 0.3 * vec(5)))]))
    return st


def test_search_groups_close_hits(tmp_path):
    st = seeded(tmp_path)
    s = Searcher(st, FakeClip({"q": vec(0)}))
    res = s.search("q")
    frames = [m for m in res if m.visit_id is None]
    # 10 s and 14 s merge; 100 s and the other video stay separate
    assert sorted((m.video_id, round(m.offset_s)) for m in frames) == [(1, 10), (1, 100), (2, 50)]
    first = next(m for m in frames if m.offset_s == 10.0)
    assert first.hits == 2 and first.thumb == "f10.jpg" and first.start_ts == T0 + 8 and first.end_ts == T0 + 16
    assert res[0].offset_s == 10.0  # best match first


def test_search_filters(tmp_path):
    st = seeded(tmp_path)
    s = Searcher(st, FakeClip({"q": vec(0), "p": vec(1)}))
    assert {m.camera for m in s.search("q", Filters(camera="Back"))} == {"Back"}
    people = s.search("p", Filters(kind="people"))
    assert people and all(m.cls == "person" for m in people)
    assert people[0].label == "person in a red top"
    assert all(m.cls in {"car"} for m in s.search("p", Filters(kind="vehicles")))
    timed = s.search("q", Filters(start_ts=T0 + 50, end_ts=T0 + 200))
    assert [round(m.offset_s) for m in timed] == [100]


def test_similar_excludes_source_visit(tmp_path):
    st = seeded(tmp_path)
    s = Searcher(st, FakeClip())
    vid1 = st.query("SELECT visit_id FROM visits WHERE thumb = 'p1.jpg'")[0][0]
    vid2 = st.query("SELECT visit_id FROM visits WHERE thumb = 'p2.jpg'")[0][0]
    res = s.similar(vid1)
    assert res[0].visit_id == vid2 and all(m.visit_id != vid1 for m in res)


def test_visits_sorting_and_filters(tmp_path):
    st = seeded(tmp_path)
    s = Searcher(st, FakeClip())
    assert [v["thumb"] for v in s.visits(sort="longest")][0] == "car.jpg"
    assert [v["thumb"] for v in s.visits(sort="fastest")][0] == "p2.jpg"
    assert {v["cls"] for v in s.visits(Filters(kind="people"))} == {"person"}
    assert [v["thumb"] for v in s.visits(min_s=60)] == ["car.jpg"]


# ---------------------------------------------------------------- processor (fakes)


class FakeYolo:
    """A person in frames whose top-left pixel is 200, else nothing."""

    def detect(self, img):
        if img[0, 0, 0] == 200:
            return [Detection("person", 0.9, (100, 100, 160, 260), True)]
        return []


class FakeModels:
    def __init__(self):
        self.clip = FakeClip()
        self.yolo = FakeYolo()
        self.device = "CPU"


def fake_reader_factory(person_between: tuple[float, float]):
    class FakeReader:
        def __init__(self, path, info, start_s=0.0, end_s=None, mode="rate", rate=2.0, max_width=1280):
            self.a, self.b = start_s, info.duration_s if end_s is None else end_s
            self.step = 1.0 if mode == "keyframes" else 1.0 / rate

        def frames(self):
            t = self.a
            while t < self.b - 1e-9:
                img = np.full((360, 640, 3), 40, np.uint8)
                if person_between[0] <= t <= person_between[1]:
                    img[0, 0] = 200
                    x = int(100 + 10 * t) % 400
                    img[100:260, x:x + 60] = 220
                yield t, img
                t = round(t + self.step, 6)

    return FakeReader


def test_processor_end_to_end_with_fakes(tmp_path, monkeypatch):
    monkeypatch.setattr(proc_mod, "FrameReader", fake_reader_factory((100, 130)))
    st = new_store(tmp_path)
    events = []
    p = Processor(st, FakeModels(), on_event=events.append)
    assert p.run() == "done"
    assert events == ["quick_done", "done"]
    assert st.spans(1, "quick")[0][0] == 0 and st.spans(1, "quick")[-1][1] == 600
    motion = st.motion(1)
    assert motion and min(a for a, _ in motion) <= 100 and max(b for _, b in motion) >= 130
    # frames kept: activity frames plus one per 30 s otherwise
    assert st.counts()["frames"] >= 600 // 30
    visits = st.query("SELECT cls, t0, t1 FROM visits")
    assert visits and visits[0][0] == "person" and visits[0][1] <= 101 and visits[0][2] >= 129
    snap = p.progress.snapshot()
    assert snap["percent"] == 100 and snap["full_eta_s"] == 0


def test_processor_resumes_without_duplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(proc_mod, "FrameReader", fake_reader_factory((100, 130)))
    st = new_store(tmp_path)
    p = Processor(st, FakeModels())
    calls = {"n": 0}
    orig = st.commit

    def commit_then_cancel(chunk):
        out = orig(chunk)
        calls["n"] += 1
        if calls["n"] == 3:
            p.cancel()
        return out

    monkeypatch.setattr(st, "commit", commit_then_cancel)
    assert p.run() == "cancelled"
    done_first = st.spans(1, "quick")
    assert done_first and done_first[-1][1] < 600
    monkeypatch.setattr(st, "commit", orig)
    assert Processor(st, FakeModels()).run() == "done"
    offsets = [r[0] for r in st.query("SELECT offset_s FROM frames ORDER BY offset_s")]
    assert len(offsets) == len(set(offsets))
    from nab_find.processor import merge
    assert merge(st.spans(1, "quick"), gap=0.01) == [(0, 600)]


def test_processor_respects_range(tmp_path, monkeypatch):
    monkeypatch.setattr(proc_mod, "FrameReader", fake_reader_factory((100, 130)))
    st = new_store(tmp_path)
    st.set_meta(range=[T0 + 200, T0 + 300])
    Processor(st, FakeModels()).run()
    assert proc_mod.merge(st.spans(1, "quick"), gap=0.01) == [(200, 300)]
    assert st.query("SELECT COUNT(*) FROM visits")[0][0] == 0  # the person was outside the range
