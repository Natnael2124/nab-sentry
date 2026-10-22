"""ByteTrack-style tracker (V2 Requirement 10, task 10.2): Properties 9-12 and unit examples.

* Property 9  - assignment invariant: per frame each detection goes to at most one track and
  each track gets at most one detection.
* Property 10 - stability: objects moving linearly without overlapping each other keep one
  track key each.
* Property 11 - expiry: a track unmatched for more than ``buffer_s`` never receives a later
  detection.
* Property 12 - snapshot/restore: restoring a snapshot and replaying the same updates gives
  identical outputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.ingest.detector import Detection
from nab_sentry.ingest.sources import DecodedFrame
from nab_sentry.track.bytetrack import ByteTracker, iou

DT = 1 / 3  # tracking rate 3 fps


def det(x1, y1, x2, y2, cls="person", conf=0.9, high=True) -> Detection:
    return Detection(cls, conf, (int(x1), int(y1), int(x2), int(y2)), high=high)


# -- strategies ------------------------------------------------------------------------------

boxes = st.tuples(st.integers(0, 600), st.integers(0, 400), st.integers(4, 120), st.integers(4, 200)).map(
    lambda t: (t[0], t[1], t[0] + t[2], t[1] + t[3]))
detections = st.builds(
    lambda b, cls, conf, high: Detection(cls, conf, b, high=high),
    boxes, st.sampled_from(["person", "car"]), st.floats(0.1, 1.0), st.booleans())
streams = st.lists(st.tuples(st.floats(0.01, 3.0), st.lists(detections, max_size=6)), max_size=25)


def run(tracker: ByteTracker, stream, t0: float = 0.0):
    out, t = [], t0
    for dt, dets in stream:
        t += dt
        out.append([(a.track_key, a.confirmed) for a in tracker.step(t, dets)])
    return out


# -- Property 9 ------------------------------------------------------------------------------


@given(streams, st.floats(0.1, 0.9), st.integers(1, 4), st.floats(0.5, 5.0))
def test_property_9_one_to_one_assignment(stream, thr, hits, buffer_s):
    tr = ByteTracker(thr, hits, buffer_s)
    t = 0.0
    for dt, dets in stream:
        t += dt
        res = tr.step(t, dets)
        assert [a.detection for a in res] == list(dets)
        keys = [a.track_key for a in res if a.track_key is not None]
        assert len(keys) == len(set(keys))
        live = [x.key for x in tr.tracks]
        assert len(live) == len(set(live))
        for a in res:  # an assigned detection belongs to a live track of its own class
            if a.track_key is not None:
                [x] = [x for x in tr.tracks if x.key == a.track_key]
                assert x.cls == a.detection.cls and x.last_t == t
        # Low detections never start a track.
        new_keys = {x.key for x in tr.tracks if x.first_t == t}
        for a in res:
            if a.track_key in new_keys:
                assert a.detection.high


# -- Property 10 -----------------------------------------------------------------------------


@st.composite
def linear_scenes(draw):
    """1-4 objects in separate horizontal bands, constant velocity, <= 8 px per step."""
    n = draw(st.integers(1, 4))
    steps = draw(st.integers(4, 20))
    objs = []
    for k in range(n):
        y = 10 + k * 150  # bands 150 px apart, boxes 90 px tall: never overlap
        x = draw(st.integers(0, 300))
        vx = draw(st.integers(-8, 8))
        vy = draw(st.integers(-3, 3))
        cls = draw(st.sampled_from(["person", "car"]))
        objs.append((x, y, vx, vy, cls))
    return objs, steps


@given(linear_scenes(), st.integers(1, 4))
def test_property_10_linear_motion_keeps_one_key(scene, min_hits):
    objs, steps = scene
    tr = ByteTracker(0.3, min_hits, 2.0)
    keys: dict[int, set] = {k: set() for k in range(len(objs))}
    confirmed_at: dict[int, int] = {}
    for s in range(steps):
        dets = [det(x + vx * s, y + vy * s, x + vx * s + 40, y + vy * s + 90, cls)
                for x, y, vx, vy, cls in objs]
        res = tr.step(s * DT, dets)
        for k, a in enumerate(res):
            keys[k].add(a.track_key)
            if a.confirmed and k not in confirmed_at:
                confirmed_at[k] = s
    assert all(len(v) == 1 and None not in v for v in keys.values())
    assert len({next(iter(v)) for v in keys.values()}) == len(objs)  # distinct per object
    if steps >= min_hits:
        assert all(confirmed_at[k] == min_hits - 1 for k in keys)


# -- Property 11 -----------------------------------------------------------------------------


@given(st.integers(1, 6), st.floats(0.5, 5.0), st.floats(0.01, 10.0), st.integers(1, 3))
def test_property_11_expired_track_gets_no_later_detection(n_before, buffer_s, extra_gap, min_hits):
    tr = ByteTracker(0.3, min_hits, buffer_s)
    box = (100, 100, 140, 190)
    t = 0.0
    first_keys = set()
    for _ in range(n_before):
        first_keys |= {a.track_key for a in tr.step(t, [det(*box)])}
        t += DT
    t += buffer_s + extra_gap  # gap strictly longer than buffer_s since the last hit
    later = tr.step(t, [det(*box)])
    assert later[0].track_key not in first_keys
    assert all(x.key not in first_keys for x in tr.tracks)


# -- Property 12 -----------------------------------------------------------------------------


@given(streams, streams)
def test_property_12_snapshot_restore_replays_identically(before, after):
    tr = ByteTracker(0.3, 2, 2.0)
    run(tr, before)
    t0 = tr._now or 0.0
    snap = tr.snapshot()
    out1, drain1 = run(tr, after, t0), tr.drain()
    tr.restore(snap)
    out2, drain2 = run(tr, after, t0), tr.drain()
    assert out1 == out2 and drain1 == drain2


# -- drain: every observation exactly once ---------------------------------------------------


@given(streams)
def test_drain_returns_each_observation_once_in_time_order(stream):
    tr = ByteTracker(0.3, 2, 1.0)
    seen: dict[int, list[float]] = {}
    t = 0.0
    for dt, dets in stream:
        t += dt
        tr.step(t, dets)
        for rec in tr.drain():
            seen.setdefault(rec.key, []).extend(o.t for o in rec.observations)
    tr.end_all()
    for rec in tr.drain():
        seen.setdefault(rec.key, []).extend(o.t for o in rec.observations)
        assert rec.ended
    for times in seen.values():
        assert times == sorted(times) and len(times) == len(set(times))
    assert tr.drain() == []


# -- examples --------------------------------------------------------------------------------


def test_low_detection_continues_but_never_starts_a_track():
    tr = ByteTracker(0.3, 2, 2.0)
    assert tr.step(0.0, [det(0, 0, 40, 90, high=False)])[0].track_key is None
    k = tr.step(DT, [det(0, 0, 40, 90)])[0].track_key
    assert tr.step(2 * DT, [det(2, 0, 42, 90, conf=0.2, high=False)])[0].track_key == k


def test_tentative_track_dies_at_first_miss_confirmed_survives_gap():
    tr = ByteTracker(0.3, 3, 2.0)
    tr.step(0.0, [det(0, 0, 40, 90)])
    tr.step(DT, [])  # miss while tentative
    assert tr.tracks == []
    for s in range(3):
        k = tr.step(1 + s * DT, [det(0, 0, 40, 90)])[0].track_key
    tr.step(2.5, [])  # confirmed: survives a miss within buffer_s
    assert [x.key for x in tr.tracks] == [k]
    assert tr.step(3.0, [det(1, 0, 41, 90)])[0].track_key == k


def test_classes_never_mix():
    tr = ByteTracker(0.3, 1, 2.0)
    kp = tr.step(0.0, [det(0, 0, 40, 90, "person")])[0].track_key
    kc = tr.step(DT, [det(0, 0, 40, 90, "car")])[0].track_key
    assert kc != kp


def test_velocity_prediction_bridges_fast_motion():
    """30 px per step on a 40 px box: raw IoU between steps is 0.14 (< 0.3); the predicted box
    still matches once the velocity is learnt."""
    tr = ByteTracker(0.3, 1, 2.0)
    assert iou((0, 0, 40, 90), (30, 0, 70, 90)) < 0.3
    keys = {tr.step(s * DT, [det(x, 0, x + 40, 90)])[0].track_key
            for s, x in enumerate([0, 10, 25, 45, 70, 100, 130])}
    assert len(keys) == 1


def test_drain_backfills_hits_before_confirmation():
    tr = ByteTracker(0.3, 3, 2.0)
    for s in range(2):
        tr.step(s * DT, [det(0, 0, 40, 90)])
    assert tr.drain() == []  # tentative: nothing yet
    tr.step(2 * DT, [det(0, 0, 40, 90)])
    [rec] = tr.drain()
    assert len(rec.observations) == 3 and not rec.ended


def test_end_all_and_time_checks():
    tr = ByteTracker(0.3, 1, 2.0)
    k = tr.step(0.0, [det(0, 0, 40, 90)])[0].track_key
    assert tr.end_all() == [k] and not tr.active
    [rec] = tr.drain()
    assert rec.ended and rec.key == k
    with pytest.raises(ValueError):
        tr.step(-1.0, [])


def test_update_uses_video_start():
    tr = ByteTracker(0.3, 1, 2.0)
    img = np.zeros((4, 4, 3), np.uint8)
    tr.begin_video(1000.0)
    tr.update(DecodedFrame(0, 9.5, img), [det(0, 0, 40, 90)])
    tr.begin_video(1010.0)  # next 10 s Segment
    tr.update(DecodedFrame(0, 0.0, img), [det(1, 0, 41, 90)])
    assert len({x.key for x in tr.tracks}) == 1 and tr.tracks[0].last_t == 1010.0
