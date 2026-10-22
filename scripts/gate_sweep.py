"""Motion gate settings sweep (V2 Requirement 3, tasks 3.3-3.6).

Usage::

    venv\\Scripts\\python.exe scripts\\gate_sweep.py VIDEO --labels eval\\ground_truth\\CAM-SEC01.yaml
    venv\\Scripts\\python.exe scripts\\gate_sweep.py SCENE_DIR          (make_scene.py output)
        [--methods diff mog2] [--thresholds 0.005 0.01 0.02 0.05] [--roi none 0,0.35,1,1]

Runs the Sampler and the MotionGate exactly as ingest does (Sample_Rate, keyframes, the gate
reset at the start of every video) for every combination and reports:

* ``missed``: incidents with no passed frame from their start to 1 s after their end (recall);
* ``idle pass``: passed / sampled frames while nothing is labelled (efficiency) - meaningful
  only for continuous footage; on the Compilation_Video every second is busy.

A scene directory supplies its own labels (``ground_truth.yaml``: each scripted object's
visible interval is an incident) and its camera ROI.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml  # noqa: E402

from nab_sentry.config import Config  # noqa: E402
from nab_sentry.ingest.motion import MotionGate  # noqa: E402
from nab_sentry.ingest.sampler import Sampler  # noqa: E402
from nab_sentry.ingest.sources import FileSource  # noqa: E402

IDLE_MARGIN_S = 2.0


# -- pure scoring ----------------------------------------------------------------------------


def missed_incidents(passed: Sequence[float], incidents: Sequence[tuple[float, float]],
                     slack: float = 1.0) -> list[tuple[float, float]]:
    """Incidents with no passed frame in ``[start, end + slack]``."""
    ps = sorted(passed)
    out = []
    for a, b in incidents:
        if not any(a <= t <= b + slack for t in ps):
            out.append((a, b))
    return out


def idle_pass_fraction(sampled: Sequence[float], passed: Sequence[float],
                       incidents: Sequence[tuple[float, float]], margin: float = IDLE_MARGIN_S) -> float | None:
    def is_idle(t: float) -> bool:
        return all(not (a - margin <= t <= b + margin) for a, b in incidents)
    idle_sampled = sum(1 for t in sampled if is_idle(t))
    if idle_sampled == 0:
        return None
    return sum(1 for t in passed if is_idle(t)) / idle_sampled


# -- running the gate ------------------------------------------------------------------------


def run_gate(videos: Sequence[tuple[Path, float]], cfg: Config, method: str, threshold: float,
             roi) -> tuple[list[float], list[float]]:
    """(sampled offsets, passed offsets) over all videos; ``videos`` = (path, start offset)."""
    sampled, passed = [], []
    for path, base in videos:
        src = FileSource.open(path)
        gate = MotionGate(threshold, cfg.keyframe_interval_s, cfg.gate_width, cfg.motion_pixel_delta,
                          method, roi=roi)  # type: ignore[arg-type]
        gate.reset()
        try:
            for f in Sampler(cfg.sample_rate).select(src.frames()):
                t = base + f.offset_s
                sampled.append(t)
                if gate.evaluate(f).passed:
                    passed.append(t)
        finally:
            src.close()
    return sampled, passed


def load_inputs(target: Path, labels: Path | None):
    if target.is_dir():
        gt = yaml.safe_load((target / "ground_truth.yaml").read_text(encoding="utf-8"))
        cams = yaml.safe_load((target / "cameras.yaml").read_text(encoding="utf-8"))["cameras"]
        segs = sorted((target / "videos").glob("*.mkv"))
        seg_s = cams[0].get("segment_s", 10)
        videos = [(p, i * seg_s) for i, p in enumerate(segs)]
        incidents = [tuple(o["visible"]) for o in gt["objects"] if o["visible"]]
        return videos, incidents, tuple(cams[0]["roi"]) if cams[0].get("roi") else None, True
    from nab_sentry.ground_truth import load_ground_truth

    if labels is None:
        raise SystemExit("error: a single video needs --labels (a Ground_Truth_File)")
    gt = load_ground_truth(labels)
    return [(target, 0.0)], [(i.start, i.end) for i in gt.incidents], None, False


def parse_roi(text: str):
    if text == "none":
        return None
    vals = tuple(float(v) for v in text.split(","))
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("ROI is x1,y1,x2,y2 or none")
    return vals


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("target", type=Path)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--methods", nargs="+", default=["diff", "mog2"])
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    p.add_argument("--roi", type=parse_roi, nargs="+", default=None,
                   help="ROIs to try (x1,y1,x2,y2 or none); default: none, plus the scene's ROI")
    args = p.parse_args(argv)

    videos, incidents, scene_roi, continuous = load_inputs(args.target, args.labels)
    rois = args.roi if args.roi is not None else ([None, scene_roi] if scene_roi else [None])
    cfg = Config()
    lines = [f"## Gate sweep: {args.target.name}, {len(videos)} video(s), {len(incidents)} incidents, "
             f"sample_rate={cfg.sample_rate}, keyframe_interval_s={cfg.keyframe_interval_s}", "",
             "| method | threshold | ROI | sampled | passed | pass fraction | missed incidents | idle pass fraction |",
             "|---|---|---|---|---|---|---|---|"]
    print("\n".join(lines), flush=True)
    rows = []
    for method in args.methods:
        for thr in args.thresholds:
            for roi in rois:
                sampled, passed = run_gate(videos, cfg, method, thr, roi)
                missed = missed_incidents(passed, incidents)
                idle = idle_pass_fraction(sampled, passed, incidents) if continuous else None
                row = (f"| {method} | {thr:g} | {'-' if roi is None else ','.join(f'{v:g}' for v in roi)} "
                       f"| {len(sampled)} | {len(passed)} | {len(passed) / max(1, len(sampled)):.3f} "
                       f"| {len(missed)}{' ' + str([round(a) for a, _ in missed]) if missed else ''} "
                       f"| {'n/a (busy footage)' if idle is None else f'{idle:.3f}'} |")
                rows.append(row)
                print(row, flush=True)
    out = (args.target if args.target.is_dir() else args.target.parent) / f"gate_sweep_{args.target.stem}.md"
    out.write_text("\n".join(lines + rows) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
