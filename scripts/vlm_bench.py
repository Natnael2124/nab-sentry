"""VLM latency benchmark: a local Ollama vision model (default Qwen3-VL 2B) on single frames and 2x2 storyboards.

Usage::

    venv\\Scripts\\python.exe scripts\\vlm_bench.py VIDEO [--label idle] [--windows 3] [--reps 3]
        [--num-ctx 4096 8192] [--variants frame grid crops] [--think]

Steps:

0. Vision sanity check (skip with ``--skip-sanity``): a solid red image must be called red and
   an image showing ``HELLO 42`` must be read back containing ``42``. A model that fails is not
   timed, because its image input is not working (as found with ``gemma4:e2b`` on 2026-10-18).
1. Scan VIDEO at 1 fps with the YOLO11n detector and find *windows*: runs of consecutive seconds
   with at least one person (gap <= 1 s, length >= 3 s). Up to ``--windows`` runs, spread over
   the video, are kept.
2. For each window build three images from 4 evenly spaced frames:
   ``frame`` (the middle frame alone), ``grid`` (2x2 storyboard of full frames) and ``crops``
   (2x2 storyboard of the largest person crop per frame, padded). Panels carry ``[i] t=..s``.
3. For each ``--num-ctx`` value: unload the model (cold start), then send every image
   ``--reps`` times to ``/api/chat`` with ``think`` off (unless ``--think``), ``temperature 0``
   and a JSON output schema. Ollama's own timings are recorded per call, plus wall time and the
   peak resident memory of all Ollama processes (sampled with ``psutil``).

Everything is written under ``data/bench/vlm/<label>_<timestamp>/`` (``results.jsonl``, the
images, ``summary.md``); the summary table is also printed. Ollama is reached on loopback only.
Run once with the machine idle and once while ``scripts/benchmark.py`` or an ingest is running
(``--label loaded``) to see contention.
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
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

from nab_sentry.config import Config, load_config  # noqa: E402

DEFAULT_MODEL = "qwen3-vl:2b-instruct"
DEFAULT_HOST = "http://127.0.0.1:11434"
PANEL_W, PANEL_H = 384, 216  # 16:9 panels -> 768x432 storyboard
CROP_PAD = 0.25
VARIANTS = ("frame", "grid", "crops")

PROMPT_GRID = (
    "This image is a storyboard of four CCTV frames in time order: [1] top-left, "
    "[2] top-right, [3] bottom-left, [4] bottom-right. First describe objectively what is "
    "visible in each panel (people, vehicles, clothing, what they hold, where they are). "
    "Then summarise in one sentence what happened across the sequence."
)
PROMPT_FRAME = (
    "This image is one CCTV frame. Describe objectively what is visible (people, vehicles, "
    "clothing, what they hold, where they are), then summarise the scene in one sentence."
)
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "observations": {"type": "string"},
        "summary": {"type": "string"},
        "people_count": {"type": "integer"},
        "vehicle_count": {"type": "integer"},
    },
    "required": ["observations", "summary", "people_count", "vehicle_count"],
}


# ---------------------------------------------------------------- pure helpers


def find_windows(person_seconds: Sequence[float], *, max_gap: float = 1.0,
                 min_len: float = 3.0, limit: int = 3) -> list[tuple[float, float]]:
    """Runs of person-bearing offsets (gap <= max_gap, length >= min_len), up to ``limit``
    spread evenly over the found runs, in time order."""
    xs = sorted(set(float(s) for s in person_seconds))
    runs: list[tuple[float, float]] = []
    for s in xs:
        if runs and s - runs[-1][1] <= max_gap:
            runs[-1] = (runs[-1][0], s)
        else:
            runs.append((s, s))
    runs = [r for r in runs if r[1] - r[0] >= min_len]
    if limit <= 0 or not runs:
        return []
    if len(runs) <= limit:
        return runs
    step = len(runs) / limit
    return [runs[int(i * step)] for i in range(limit)]


def keyframe_offsets(start: float, end: float, n: int = 4, max_span: float = 8.0) -> list[float]:
    """``n`` evenly spaced offsets from ``start`` to ``min(end, start + max_span)``."""
    stop = min(end, start + max_span)
    if n == 1:
        return [(start + stop) / 2]
    return [start + (stop - start) * i / (n - 1) for i in range(n)]


def fit_panel(image: np.ndarray, w: int = PANEL_W, h: int = PANEL_H) -> np.ndarray:
    """Letterbox ``image`` into a black ``h x w`` panel, preserving aspect ratio."""
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    ih, iw = image.shape[:2]
    if ih == 0 or iw == 0:
        return panel
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    panel[y0:y0 + nh, x0:x0 + nw] = resized
    return panel


def stamp(panel: np.ndarray, text: str) -> np.ndarray:
    """Draw ``text`` in the top-left corner, white with a black outline."""
    out = panel.copy()
    for color, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thick, cv2.LINE_AA)
    return out


def storyboard(panels: Sequence[np.ndarray]) -> np.ndarray:
    """2x2 grid of four equally sized panels in reading order."""
    if len(panels) != 4:
        raise ValueError(f"storyboard needs 4 panels, got {len(panels)}")
    return np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])


def bust_cache(image: np.ndarray, rep: int) -> np.ndarray:
    """Copy of ``image`` with one corner pixel changed per ``rep``, so Ollama's prompt cache never
    reuses an earlier call's image encoding (every real event is a new image)."""
    if rep == 0:
        return image
    out = image.copy()
    out[-1, -1] = (out[-1, -1].astype(np.int64) + rep * 37) % 256
    return out


def padded_crop(image: np.ndarray, box: tuple[int, int, int, int], pad: float = CROP_PAD) -> np.ndarray:
    """Crop ``box`` grown by ``pad`` of its size on every side, clamped to the image."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    a, b = max(0, int(x1 - dx)), max(0, int(y1 - dy))
    c, d = min(w, int(x2 + dx)), min(h, int(y2 + dy))
    return image[b:d, a:c]


@dataclass(frozen=True)
class CallTiming:
    wall_s: float
    load_s: float
    prompt_tokens: int
    prompt_s: float
    gen_tokens: int
    gen_s: float

    @property
    def gen_tok_per_s(self) -> float:
        return self.gen_tokens / self.gen_s if self.gen_s > 0 else 0.0


def parse_timing(resp: dict[str, Any], wall_s: float) -> CallTiming:
    """Ollama durations are nanoseconds; missing fields count as zero."""
    ns = 1e-9
    return CallTiming(
        wall_s=wall_s,
        load_s=resp.get("load_duration", 0) * ns,
        prompt_tokens=int(resp.get("prompt_eval_count", 0)),
        prompt_s=resp.get("prompt_eval_duration", 0) * ns,
        gen_tokens=int(resp.get("eval_count", 0)),
        gen_s=resp.get("eval_duration", 0) * ns,
    )


# ---------------------------------------------------------------- video + detector


def read_frame_at(cap: Any, offset_s: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, offset_s * 1000.0)
    ok, image = cap.read()
    return image if ok and image is not None and image.size else None


def scan_people(video: Path, detector: Any, duration_s: float) -> dict[float, list]:
    """Person detections per whole second of video."""
    cap = cv2.VideoCapture(str(video))
    out: dict[float, list] = {}
    try:
        for s in range(int(duration_s)):
            image = read_frame_at(cap, float(s))
            if image is None:
                continue
            people = [d for d in detector.detect(image) if d.cls == "person"]
            if people:
                out[float(s)] = people
    finally:
        cap.release()
    return out


def build_images(video: Path, detector: Any, window: tuple[float, float]) -> dict[str, np.ndarray]:
    offsets = keyframe_offsets(*window)
    cap = cv2.VideoCapture(str(video))
    try:
        frames = [(t, read_frame_at(cap, t)) for t in offsets]
    finally:
        cap.release()
    frames = [(t, f) for t, f in frames if f is not None]
    if len(frames) != 4:
        raise RuntimeError(f"could not decode 4 frames for window {window}")

    grid_panels, crop_panels = [], []
    for i, (t, image) in enumerate(frames, start=1):
        label = f"[{i}] t={t - offsets[0]:.1f}s"
        grid_panels.append(stamp(fit_panel(image), label))
        people = [d for d in detector.detect(image) if d.cls == "person"]
        if people:
            big = max(people, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
            crop = padded_crop(image, big.box)
        else:
            crop = image
        crop_panels.append(stamp(fit_panel(crop), label))
    middle = frames[len(frames) // 2][1]
    return {
        "frame": fit_panel(middle, 2 * PANEL_W, 2 * PANEL_H),
        "grid": storyboard(grid_panels),
        "crops": storyboard(crop_panels),
    }


# ---------------------------------------------------------------- Ollama


class RssSampler:
    """Peak summed RSS (MiB) of processes whose name contains 'ollama' or 'llama'."""

    def __init__(self, interval_s: float = 0.2) -> None:
        self.interval_s = interval_s
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> float:
        import psutil

        total = 0
        for p in psutil.process_iter(["name", "memory_info"]):
            name = (p.info.get("name") or "").lower()
            mem = p.info.get("memory_info")
            if mem is not None and ("ollama" in name or "llama" in name):
                total += mem.rss
        return total / (1024 * 1024)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self._sample())
            self._stop.wait(self.interval_s)

    def __enter__(self) -> RssSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


def unload(client: Any, host: str, model: str) -> None:
    client.post(f"{host}/api/generate", json={"model": model, "keep_alive": 0}).raise_for_status()


def chat(client: Any, host: str, model: str, image: np.ndarray, prompt: str, num_ctx: int,
         think: bool) -> tuple[dict[str, Any], float]:
    ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt,
                      "images": [base64.b64encode(jpg.tobytes()).decode("ascii")]}],
        "stream": False,
        "think": think,
        "format": OUTPUT_SCHEMA,
        "keep_alive": "10m",
        "options": {"num_ctx": num_ctx, "temperature": 0, "num_predict": 512},
    }
    t0 = time.perf_counter()
    r = client.post(f"{host}/api/chat", json=body)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    return r.json(), wall


def sanity_images() -> list[tuple[str, np.ndarray, str, str]]:
    """(name, image, question, expected substring, lower-case) for the vision sanity check."""
    red = np.zeros((448, 448, 3), np.uint8)
    red[:, :, 2] = 255
    text = np.full((448, 448, 3), 255, np.uint8)
    cv2.putText(text, "HELLO 42", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 6)
    return [("colour", red, "What colour is this image? Answer with one word.", "red"),
            ("text", text, "What text is written in this image? Answer with the text only.", "42")]


def sanity_passed(answer: str, expected: str) -> bool:
    return expected in answer.lower()


def run_sanity(client: Any, host: str, model: str) -> bool:
    ok = True
    for name, image, question, expected in sanity_images():
        ok_jpg, jpg = cv2.imencode(".jpg", image)
        body = {"model": model, "stream": False, "keep_alive": "10m", "think": False,
                "messages": [{"role": "user", "content": question,
                              "images": [base64.b64encode(jpg.tobytes()).decode("ascii")]}],
                "options": {"temperature": 0, "num_predict": 32}}
        answer = client.post(f"{host}/api/chat", json=body).json().get("message", {}).get("content", "")
        passed = sanity_passed(answer, expected)
        ok &= passed
        print(f"sanity {name}: {'PASS' if passed else 'FAIL'} ({answer.strip()[:80]!r})", flush=True)
    return ok


# ---------------------------------------------------------------- report


def summarize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One summary per (num_ctx, variant): cold load, warm medians, peak RSS."""
    out = []
    keys = sorted({(r["num_ctx"], r["variant"]) for r in rows}, key=lambda k: (k[0], VARIANTS.index(k[1])))
    for ctx, variant in keys:
        group = [r for r in rows if r["num_ctx"] == ctx and r["variant"] == variant]
        warm = [r for r in group if not r["cold"]] or group
        med = lambda f: statistics.median(r[f] for r in warm)  # noqa: E731
        out.append({
            "num_ctx": ctx, "variant": variant, "n": len(warm),
            "cold_load_s": max((r["load_s"] for r in group if r["cold"]), default=0.0),
            "wall_s": med("wall_s"), "wall_max_s": max(r["wall_s"] for r in warm),
            "prompt_tokens": med("prompt_tokens"), "prompt_s": med("prompt_s"),
            "gen_tokens": med("gen_tokens"), "gen_tok_s": med("gen_tok_per_s"),
            "peak_rss_mb": max(r["peak_rss_mb"] for r in group),
        })
    return out


def format_summary(summary: Sequence[dict[str, Any]], header: str) -> str:
    lines = [header, "",
             "| num_ctx | input | n | cold load (s) | wall median (s) | wall max (s) | image+prompt tokens "
             "| prompt eval (s) | gen tokens | gen tok/s | peak Ollama RSS (MiB) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(
            f"| {s['num_ctx']} | {s['variant']} | {s['n']} | {s['cold_load_s']:.1f} | {s['wall_s']:.1f} "
            f"| {s['wall_max_s']:.1f} | {s['prompt_tokens']:.0f} | {s['prompt_s']:.1f} "
            f"| {s['gen_tokens']:.0f} | {s['gen_tok_s']:.1f} | {s['peak_rss_mb']:,.0f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("video", type=Path)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--label", default="idle", help="tag for this run, e.g. idle / loaded")
    p.add_argument("--windows", type=int, default=3)
    p.add_argument("--reps", type=int, default=3, help="calls per image (first after load = cold)")
    p.add_argument("--num-ctx", type=int, nargs="+", default=[4096, 8192])
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--think", action="store_true", help="turn model thinking on (default off)")
    p.add_argument("--skip-sanity", action="store_true", help="time the model even if it fails the vision check")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="NAME=VALUE")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import httpx

    from nab_sentry.config import parse_set_args
    from nab_sentry.ingest.detector import OnnxYoloDetector
    from nab_sentry.model_files import detector_input, detector_path
    from nab_sentry.startup import require_models

    cfg: Config = load_config(parse_set_args(args.overrides))
    files = require_models(cfg.models_dir)
    if not args.host.startswith(("http://127.0.0.1", "http://localhost")):
        print("error: --host must be a loopback address", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(str(args.video))
    fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if not fps or not count:
        print(f"error: cannot read {args.video}", file=sys.stderr)
        return 1
    detector = OnnxYoloDetector(detector_path(cfg, files), conf=cfg.det_conf, iou=cfg.det_iou,
                                max_det=cfg.det_max_per_frame, input_size=detector_input(cfg),
                                threads=cfg.det_threads)

    print(f"scanning {args.video.name} for person windows ...", flush=True)
    people = scan_people(args.video, detector, count / fps)
    windows = find_windows(list(people), limit=args.windows)
    if not windows:
        print("error: no person windows of >= 3 s found", file=sys.stderr)
        return 1
    print("windows: " + ", ".join(f"{a:.0f}-{b:.0f}s" for a, b in windows), flush=True)

    out_dir = cfg.data_dir / "bench" / "vlm" / f"{args.label}_{datetime.now():%Y%m%dT%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    images: list[tuple[int, str, np.ndarray]] = []
    for w_i, window in enumerate(windows):
        built = build_images(args.video, detector, window)
        for variant in args.variants:
            cv2.imwrite(str(out_dir / f"w{w_i}_{variant}.jpg"), built[variant])
            images.append((w_i, variant, built[variant]))

    if not args.skip_sanity:
        with httpx.Client(timeout=900.0) as client:
            if not run_sanity(client, args.host, args.model):
                print("error: vision sanity check failed; not timing this model", file=sys.stderr)
                return 3

    rows: list[dict[str, Any]] = []
    results = out_dir / "results.jsonl"
    with httpx.Client(timeout=900.0) as client, results.open("w", encoding="utf-8") as fh:
        for ctx in args.num_ctx:
            unload(client, args.host, args.model)
            first = True
            for rep in range(args.reps):
                for w_i, variant, image in images:
                    prompt = PROMPT_FRAME if variant == "frame" else PROMPT_GRID
                    with RssSampler() as rss:
                        resp, wall = chat(client, args.host, args.model, bust_cache(image, rep),
                                          prompt, ctx, args.think)
                    t = parse_timing(resp, wall)
                    row = {"label": args.label, "num_ctx": ctx, "variant": variant, "window": w_i,
                           "rep": rep, "cold": first, **asdict(t), "gen_tok_per_s": t.gen_tok_per_s,
                           "peak_rss_mb": rss.peak_mb,
                           "output": resp.get("message", {}).get("content", "")}
                    first = False
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print(f"ctx={ctx} {variant:5s} w{w_i} rep{rep}: {wall:5.1f}s "
                          f"(prompt {t.prompt_tokens} tok {t.prompt_s:.1f}s, gen {t.gen_tokens} tok "
                          f"{t.gen_tok_per_s:.1f} tok/s, load {t.load_s:.1f}s, RSS {rss.peak_mb:,.0f} MiB)",
                          flush=True)

        # Ollama's own view of the loaded model size (RSS undercounts memory-mapped weights).
        ps = client.get(f"{args.host}/api/ps").json().get("models", [])
        loaded = next((m for m in ps if m.get("name") == args.model), {})
    header = (f"## VLM bench: {args.model}, label={args.label}, think={args.think}, "
              f"video={args.video.name}, windows={len(windows)}, reps={args.reps}, "
              f"Ollama loaded size={loaded.get('size', 0) / 2**30:.1f} GiB")
    text = format_summary(summarize_rows(rows), header)
    (out_dir / "summary.md").write_text(text + "\n", encoding="utf-8")
    print("\n" + text + f"\n\nwritten to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
