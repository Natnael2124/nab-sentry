"""Detector cost at tracking rates: YOLO11n ONNX at several input sizes and thread counts.

Usage::

    venv\\Scripts\\python.exe scripts\\det_bench.py VIDEO [--sizes 320 416 480 640 384x640]
        [--threads 2 4] [--frames 60] [--iters 3]

Each size is exported once from ``models/yolo11n.pt`` with Ultralytics (static shape, no NMS,
like ``fetch_models.py``) into ``data/bench/det/`` - the production ``models/yolo11n.onnx`` is
never touched. A size ``HxW`` exports a rectangular model (the detector letterboxes to ``H x W``),
which suits 16:9 cameras.

For every size x thread count the script reports median / p95 ms per frame over ``--frames``
frames spread across VIDEO (each timed ``--iters`` times after 5 warm-up runs; all
configurations are interleaved in a shuffled order per frame so CPU clock changes affect them
equally), the detector
busy fraction at Track_Rate 2, 3 and 5 fps (``ms x rate / 1000``: share of one camera's
wall-clock time the detector needs with that many threads), and agreement with the 640 model:
recall of the 640 boxes (same class, IoU >= 0.5) by box height (small < 80 px, medium < 200 px,
large) and the number of extra boxes. Results go to ``data/bench/det/summary.md``.
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nab_sentry.startup import enable_offline_mode  # noqa: E402  (stdlib-only)

# Must run before onnxruntime is imported (Requirement 13.3).
enable_offline_mode()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from nab_sentry.config import Config, load_config, parse_set_args  # noqa: E402

TRACK_RATES = (2, 3, 5)
SMALL_PX, LARGE_PX = 80, 200
MATCH_IOU = 0.5
BASELINE = (640, 640)


# ---------------------------------------------------------------- pure helpers


def parse_size(text: str) -> tuple[int, int]:
    """``"416"`` -> (416, 416); ``"384x640"`` -> (384, 640) as (height, width)."""
    parts = text.lower().split("x")
    if len(parts) == 1:
        h = w = int(parts[0])
    elif len(parts) == 2:
        h, w = int(parts[0]), int(parts[1])
    else:
        raise ValueError(f"bad size {text!r}")
    if h < 32 or w < 32 or h % 32 or w % 32:
        raise ValueError(f"size must be multiples of 32 and >= 32, got {text!r}")
    return h, w


def size_label(hw: tuple[int, int]) -> str:
    return str(hw[0]) if hw[0] == hw[1] else f"{hw[0]}x{hw[1]}"


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(base: list, other: list, thr: float = MATCH_IOU) -> tuple[list[bool], int]:
    """Greedy same-class matching by IoU. Returns per-base-box matched flags and the number of
    unmatched ``other`` boxes."""
    pairs = sorted(
        ((iou(b.box, o.box), i, j) for i, b in enumerate(base) for j, o in enumerate(other)
         if b.cls == o.cls),
        reverse=True,
    )
    used_b: set[int] = set()
    used_o: set[int] = set()
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_b or j in used_o:
            continue
        used_b.add(i)
        used_o.add(j)
    return [i in used_b for i in range(len(base))], len(other) - len(used_o)


def bucket(box: Sequence[int]) -> str:
    h = box[3] - box[1]
    return "small" if h < SMALL_PX else "medium" if h < LARGE_PX else "large"


def busy_fraction(ms: float, rate: float) -> float:
    return ms * rate / 1000.0


@dataclass
class Agreement:
    total: dict[str, int]
    found: dict[str, int]
    extra: int

    def recall(self, b: str | None = None) -> float:
        t = sum(self.total.values()) if b is None else self.total.get(b, 0)
        f = sum(self.found.values()) if b is None else self.found.get(b, 0)
        return f / t if t else float("nan")


def agreement(base_dets: list[list], other_dets: list[list]) -> Agreement:
    total = {"small": 0, "medium": 0, "large": 0}
    found = dict(total)
    extra = 0
    for base, other in zip(base_dets, other_dets):
        flags, ex = match(base, other)
        extra += ex
        for d, ok in zip(base, flags):
            b = bucket(d.box)
            total[b] += 1
            found[b] += int(ok)
    return Agreement(total, found, extra)


# ---------------------------------------------------------------- models


def export_onnx(pt: Path, out_dir: Path, hw: tuple[int, int]) -> Path:
    """Export ``pt`` at ``hw`` into ``out_dir`` (via a copy, so nothing lands in ``models/``)."""
    target = out_dir / f"yolo11n_{size_label(hw)}.onnx"
    if target.is_file():
        return target
    work = out_dir / f"export_{size_label(hw)}"
    work.mkdir(parents=True, exist_ok=True)
    local_pt = work / "yolo11n.pt"
    shutil.copy2(pt, local_pt)
    from ultralytics import YOLO

    imgsz = hw[0] if hw[0] == hw[1] else list(hw)
    produced = Path(YOLO(str(local_pt)).export(format="onnx", imgsz=imgsz, dynamic=False, nms=False))
    shutil.move(str(produced), target)
    shutil.rmtree(work, ignore_errors=True)
    return target


def make_detector(path: Path, hw: tuple[int, int], cfg: Config, threads: int) -> Any:
    from nab_sentry.ingest.detector import OnnxYoloDetector

    return OnnxYoloDetector(path, conf=cfg.det_conf, iou=cfg.det_iou,
                            max_det=cfg.det_max_per_frame, input_size=hw, threads=threads)


def sample_frames(video: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = []
        for i in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * count / n))
            ok, img = cap.read()
            if ok and img is not None:
                out.append(img)
        return out
    finally:
        cap.release()


def time_interleaved(dets: dict[Any, Any], frames: list[np.ndarray], iters: int,
                     seed: int = 0) -> dict[Any, tuple[list[float], list[list]]]:
    """Time every detector on every frame, rotating through the detectors in a shuffled order
    per (frame, iter), so CPU clock changes (turbo, thermal, power plan) hit all of them alike
    instead of whichever happened to run during a slow spell."""
    rng = np.random.default_rng(seed)
    keys = list(dets)
    for f in frames[:5]:
        for k in keys:
            dets[k].detect(f)
    out: dict[Any, tuple[list[float], list[list]]] = {k: ([], []) for k in keys}
    for f in frames:
        last: dict[Any, list] = {}
        for _ in range(iters):
            for i in rng.permutation(len(keys)):
                k = keys[i]
                t0 = time.perf_counter()
                last[k] = dets[k].detect(f)
                out[k][0].append((time.perf_counter() - t0) * 1000)
        for k in keys:
            out[k][1].append(last[k])
    return out


# ---------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("video", type=Path)
    p.add_argument("--sizes", nargs="+", default=["320", "416", "480", "640", "384x640"])
    p.add_argument("--threads", type=int, nargs="+", default=[2, 4])
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="NAME=VALUE")
    args = p.parse_args(argv)

    cfg = load_config(parse_set_args(args.overrides))
    sizes = [parse_size(s) for s in args.sizes]
    if BASELINE not in sizes:
        sizes.append(BASELINE)
    out_dir = cfg.data_dir / "bench" / "det"
    out_dir.mkdir(parents=True, exist_ok=True)
    pt = cfg.models_dir / "yolo11n.pt"
    if not pt.is_file():
        print(f"error: {pt} not found (run scripts/fetch_models.py)", file=sys.stderr)
        return 4

    frames = sample_frames(args.video, args.frames)
    print(f"{len(frames)} frames from {args.video.name} ({frames[0].shape[1]}x{frames[0].shape[0]})", flush=True)
    paths = {hw: export_onnx(pt, out_dir, hw) for hw in sizes}

    detectors = {(hw, t): make_detector(paths[hw], hw, cfg, t) for hw in sizes for t in args.threads}
    results = time_interleaved(detectors, frames, args.iters)
    for (hw, t), (times, _) in results.items():
        print(f"{size_label(hw):>8} threads={t}: median {statistics.median(times):6.1f} ms", flush=True)

    base = results[(BASELINE, max(args.threads))][1]
    lines = [f"## Detector cost: {args.video.name}, {len(frames)} frames x {args.iters} iters, "
             f"det_conf={cfg.det_conf}", "",
             "| input | threads | median ms | p95 ms | busy @2 fps | busy @3 fps | busy @5 fps "
             "| recall vs 640 (all / small / medium / large) | extra boxes |",
             "|---|---|---|---|---|---|---|---|---|"]
    for hw in sizes:
        agr = agreement(base, results[(hw, max(args.threads))][1])
        for t in args.threads:
            times = results[(hw, t)][0]
            med = statistics.median(times)
            p95 = float(np.percentile(times, 95))
            busy = " | ".join(f"{busy_fraction(med, r):.0%}" for r in TRACK_RATES)
            rec = " / ".join(f"{agr.recall(b):.2f}" for b in (None, "small", "medium", "large"))
            lines.append(f"| {size_label(hw)} | {t} | {med:.1f} | {p95:.1f} | {busy} | {rec} | {agr.extra} |")
    a = agreement(base, base)
    lines += ["", f"640 boxes: {sum(a.total.values())} (small {a.total['small']}, medium "
              f"{a.total['medium']}, large {a.total['large']}); small < {SMALL_PX} px tall, "
              f"large >= {LARGE_PX} px."]
    text = "\n".join(lines)
    (out_dir / "summary.md").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
