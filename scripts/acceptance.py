"""Acceptance runs on a generated scene (V2 Requirement 17, task 17.1).

Usage::

    venv\\Scripts\\python.exe scripts\\acceptance.py SCENE_DIR batch
    venv\\Scripts\\python.exe scripts\\acceptance.py SCENE_DIR realtime [--from 100] [--to 400]

``SCENE_DIR`` comes from ``scripts/make_scene.py`` (Segments, cameras.yaml, rules.yaml,
ground_truth.yaml). Both modes use the real detector, embedder, tracker and rules in a fresh
data folder under ``SCENE_DIR/run_<mode>/``.

* ``batch`` ingests every Segment as fast as possible and reports: rule precision and recall
  against the expected alerts (same rule, within +-``--tolerance`` s), tracking fragmentation
  (confirmed tracks per scripted object), gate recall (every object has a passed frame) and
  gate efficiency (passed / sampled frames while the scene is idle), ingest speed, CPU time
  and peak memory.
* ``realtime`` replays the Segments between ``--from`` and ``--to`` seconds at real-time pace -
  each file appears in the watched folder when its 10 s are over, renamed to the current wall
  clock as the Recorder would name it - while the Watcher and the Notifier run as in production
  (the Notifier with a transport that records instead of calling Telegram). It reports the
  alert latency: from the moment the event happened in the replay to the moment the alert
  photo would have been sent.

Results are printed and written to ``SCENE_DIR/run_<mode>/report.yaml``.
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nab_sentry.startup import enable_offline_mode  # noqa: E402

enable_offline_mode()

import psutil  # noqa: E402
import yaml  # noqa: E402

from nab_sentry.config import Config  # noqa: E402
from nab_sentry.ingest.metadata import FILENAME_RE  # noqa: E402
from nab_sentry.ingest.pipeline import IngestPipeline  # noqa: E402
from nab_sentry.live.camera_config import load_cameras  # noqa: E402
from nab_sentry.model_files import default_detector_factory, default_encoder_factory  # noqa: E402
from nab_sentry.rules.config import load_rules  # noqa: E402
from nab_sentry.startup import require_models  # noqa: E402
from nab_sentry.store.db import MetadataStore  # noqa: E402
from nab_sentry.store.vector_index import VectorIndex  # noqa: E402
from nab_sentry.track.live import LiveTracks  # noqa: E402

TOLERANCE_S = 5.0
IDLE_MARGIN_S = 3.0


# -- pure scoring --------------------------------------------------------------------------------


def match_alerts(expected: Sequence[dict], actual: Sequence[dict], tol: float) -> dict:
    """Greedy one-to-one matching per rule by time distance (<= ``tol``). ``expected`` and
    ``actual`` items have ``rule`` and ``t`` (seconds from the footage start)."""
    per_rule: dict[str, dict] = {}
    for rule in sorted({a["rule"] for a in expected} | {a["rule"] for a in actual}):
        exp = sorted(a["t"] for a in expected if a["rule"] == rule)
        act = sorted(a["t"] for a in actual if a["rule"] == rule)
        pairs = sorted((abs(e - a), i, j) for i, e in enumerate(exp) for j, a in enumerate(act) if abs(e - a) <= tol)
        used_e, used_a, delays = set(), set(), []
        for _d, i, j in pairs:
            if i in used_e or j in used_a:
                continue
            used_e.add(i)
            used_a.add(j)
            delays.append(act[j] - exp[i])
        per_rule[rule] = {"expected": len(exp), "actual": len(act), "tp": len(used_e),
                          "fn": len(exp) - len(used_e), "fp": len(act) - len(used_a),
                          "missed_at": [round(exp[i], 1) for i in range(len(exp)) if i not in used_e],
                          "extra_at": [round(act[j], 1) for j in range(len(act)) if j not in used_a],
                          "delay_s": [round(d, 2) for d in delays]}
    tp = sum(r["tp"] for r in per_rule.values())
    n_exp = sum(r["expected"] for r in per_rule.values())
    n_act = sum(r["actual"] for r in per_rule.values())
    return {"rules": per_rule, "recall": tp / n_exp if n_exp else 1.0,
            "precision": tp / n_act if n_act else 1.0}


def idle(t: float, objects: Sequence[dict], margin: float = IDLE_MARGIN_S) -> bool:
    return all(not o["visible"] or not (o["visible"][0] - margin <= t <= o["visible"][1] + margin)
               for o in objects)


# -- setup ---------------------------------------------------------------------------------------


def build(scene: Path, run: Path, rules: bool = True):
    shutil.rmtree(run, ignore_errors=True)
    cfg = Config(root=run, models_dir=_ROOT / "models")
    cfg.videos_dir.mkdir(parents=True)
    cameras = load_cameras(scene / "cameras.yaml")
    rule_list = load_rules(scene / "rules.yaml", cameras) if rules else []
    files = require_models(cfg.models_dir)
    db = MetadataStore(cfg.db_path)
    pipe = IngestPipeline(cfg, db, VectorIndex(), default_encoder_factory(cfg, files),
                          default_detector_factory(cfg, files), tracker=LiveTracks.from_config(cfg),
                          rules=rule_list, rois={c.camera_id: c.roi for c in cameras if c.roi})
    return cfg, db, pipe, cameras, rule_list


def offset_of(gt: dict, epoch_ms: int, start: datetime | None = None) -> float:
    base = start or datetime.fromisoformat(gt["start"])
    return epoch_ms / 1000 - base.timestamp()


# -- batch ---------------------------------------------------------------------------------------


def run_batch(scene: Path, tol: float) -> dict:
    gt = yaml.safe_load((scene / "ground_truth.yaml").read_text(encoding="utf-8"))
    cfg, db, pipe, _, _ = build(scene, scene / "run_batch")
    for f in sorted((scene / "videos").glob("*.mkv")):
        shutil.copy2(f, cfg.videos_dir / f.name)
    proc = psutil.Process()
    peak = {"rss": 0}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak["rss"] = max(peak["rss"], proc.memory_info().rss)
            stop.wait(0.5)
    threading.Thread(target=sample, daemon=True).start()
    cpu0, t0 = proc.cpu_times(), time.perf_counter()
    results = pipe.ingest_paths([cfg.videos_dir]).results
    wall = time.perf_counter() - t0
    cpu1 = proc.cpu_times()
    stop.set()
    try:
        c = db._conn
        alerts = [{"rule": r, "t": offset_of(gt, ms)} for r, ms in
                  c.execute("SELECT rule_id, trigger_epoch_ms FROM alerts WHERE kind = 'rule'")]
        scores = match_alerts(gt["expected_alerts"], alerts, tol)
        frames = [(offset_of(gt, ms), reason) for ms, reason in
                  c.execute("SELECT abs_epoch_ms, gate_reason FROM frames")]
        sampled = sum(r[0] for r in c.execute("SELECT sampled_count FROM videos"))
        idle_sampled = sum(1 for i in range(int(gt["duration_s"])) if idle(float(i), gt["objects"]))
        idle_passed = sum(1 for t, _ in frames if idle(t, gt["objects"]))
        idle_motion = sum(1 for t, r in frames if idle(t, gt["objects"]) and r == "motion")
        missed_objects = [o["name"] for o in gt["objects"] if o["visible"] and not any(
            o["visible"][0] <= t <= o["visible"][1] + 1 for t, _ in frames)]
        tracks = [(cls, offset_of(gt, f), offset_of(gt, l)) for cls, f, l in
                  c.execute("SELECT det_class, first_epoch_ms, last_epoch_ms FROM tracks")]
        frag = {}
        for o in gt["objects"]:
            if not o["visible"]:
                continue
            a, b = o["visible"]
            frag[o["name"]] = sum(1 for cls, f, l in tracks
                                  if cls == o["class"] and f <= b + 1 and l >= a - 1)
        report = {
            "segments": len(results), "status": sorted({r.status for r in results}),
            "footage_s": gt["duration_s"], "wall_s": round(wall, 1),
            "speed_x_realtime": round(gt["duration_s"] / wall, 2),
            "cpu_s": round((cpu1.user - cpu0.user) + (cpu1.system - cpu0.system), 1),
            "peak_rss_mb": round(peak["rss"] / 2**20),
            "alerts": scores,
            "gate": {"sampled": sampled, "passed": len(frames), "idle_sampled_s": idle_sampled,
                     "idle_passed": idle_passed, "idle_motion_passes": idle_motion,
                     "idle_pass_fraction": round(idle_passed / max(1, idle_sampled), 3),
                     "objects_without_a_passed_frame": missed_objects},
            "tracks": {"confirmed": len(tracks), "per_object": frag,
                       "unmatched": sum(1 for cls, f, l in tracks if not any(
                           o["visible"] and cls == o["class"] and f <= o["visible"][1] + 1 and l >= o["visible"][0] - 1
                           for o in gt["objects"]))},
        }
    finally:
        db.close()
    return report


# -- realtime ------------------------------------------------------------------------------------


class RecordingTransport:
    """Stands in for Telegram: accepts every call and records when photos would have gone out."""

    def __init__(self) -> None:
        self.sent: list[tuple[float, str, dict]] = []
        self.n = 0

    def call(self, method, data, files, timeout):
        if method == "getUpdates":
            time.sleep(min(1.0, float(data.get("timeout", 1))))
            return []
        self.n += 1
        self.sent.append((time.time(), method, dict(data)))
        return {"message_id": self.n} if method.startswith("send") else True


def run_realtime(scene: Path, t_from: float, t_to: float, tol: float) -> dict:
    from nab_sentry.ingest.watch import Watcher
    from nab_sentry.notify.commands import Commands
    from nab_sentry.notify.outbox import Outbox
    from nab_sentry.notify.service import Notifier
    from nab_sentry.notify.telegram import TelegramClient

    gt = yaml.safe_load((scene / "ground_truth.yaml").read_text(encoding="utf-8"))
    cfg, db, pipe, cameras, rules = build(scene, scene / "run_realtime")
    seg_s = cameras[0].segment_s
    segs = sorted((scene / "videos").glob("*.mkv"))
    first, last = int(t_from // seg_s), int(min(gt["duration_s"], t_to) // seg_s)
    segs = segs[first:last]
    replay_start = datetime.now().replace(microsecond=0) + timedelta(seconds=2)
    wall0 = replay_start.timestamp()

    stop = threading.Event()
    watcher = Watcher(pipe, cfg.videos_dir, lambda cam: float(seg_s), poll_s=0.5)
    wt = threading.Thread(target=watcher.run, args=(stop,), daemon=True)
    ndb = MetadataStore(cfg.db_path)
    tr = RecordingTransport()
    client = TelegramClient(tr)
    notifier = Notifier(ndb, client, [1], Commands(ndb, {}, rules, live_dir=cfg.live_dir,
                                                   offline_after_s=lambda c: 1e9),
                        Outbox(cfg, ndb, client, [1], segment_s=lambda c: float(seg_s)))
    nt = threading.Thread(target=notifier.run, args=(stop, 0.2), daemon=True)
    wt.start()
    nt.start()
    try:
        for k, src in enumerate(segs):
            closes_at = wall0 + (k + 1) * seg_s  # the Recorder closes Segment k at this time
            time.sleep(max(0.0, closes_at - time.time()))
            m = FILENAME_RE.match(src.name)
            name = f"{m.group('cam')}_{(replay_start + timedelta(seconds=k * seg_s)):%Y%m%dT%H%M%S}.mkv"
            shutil.copy2(src, cfg.videos_dir / name)
            print(f"  segment {k + 1}/{len(segs)} closed", flush=True)
        deadline = time.time() + 60  # let the last Segment be ingested and the alerts go out
        while time.time() < deadline:
            pending = [a for a in ndb.due_alerts(2**62, 1000)]
            done = watcher.done and len(watcher.done) >= len(segs)
            if done and not pending:
                break
            time.sleep(1)
    finally:
        stop.set()
        wt.join(30)
        nt.join(30)
    try:
        rows = ndb._conn.execute(
            "SELECT a.alert_id, a.rule_id, a.trigger_epoch_ms, m.message_id FROM alerts a "
            "JOIN alert_messages m USING (alert_id) WHERE m.step = 'photo'").fetchall()
        # The transport numbers messages in call order, so message_id - 1 indexes the send times.
        call_times = [ts for ts, _method, _data in tr.sent]
        latencies, alerts = [], []
        for aid, rule, ms, mid in rows:
            t_scene = offset_of(gt, ms, replay_start) + first * seg_s
            alerts.append({"rule": rule, "t": t_scene})
            event_wall = ms / 1000
            sent_wall = call_times[mid - 1]
            latencies.append(sent_wall - event_wall)
        expected = [a for a in gt["expected_alerts"] if first * seg_s <= a["t"] < last * seg_s]
        scores = match_alerts(expected, alerts, tol)
        report = {"window_s": [first * seg_s, last * seg_s], "segments": len(segs),
                  "alerts": scores,
                  "latency_s": {"n": len(latencies),
                                "median": round(statistics.median(latencies), 1) if latencies else None,
                                "max": round(max(latencies), 1) if latencies else None,
                                "all": [round(x, 1) for x in sorted(latencies)]}}
    finally:
        ndb.close()
        db.close()
    return report


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("scene", type=Path)
    p.add_argument("mode", choices=["batch", "realtime"])
    p.add_argument("--from", dest="t_from", type=float, default=0.0)
    p.add_argument("--to", dest="t_to", type=float, default=1e9)
    p.add_argument("--tolerance", type=float, default=TOLERANCE_S)
    args = p.parse_args(argv)
    if not (args.scene / "ground_truth.yaml").is_file():
        print(f"error: {args.scene} is not a make_scene.py output", file=sys.stderr)
        return 1
    report = (run_batch(args.scene, args.tolerance) if args.mode == "batch"
              else run_realtime(args.scene, args.t_from, args.t_to, args.tolerance))
    out = args.scene / f"run_{args.mode}" / "report.yaml"
    out.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(report, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
