"""Generate continuous single-camera test footage with exact ground truth (V2 task 17).

Usage::

    venv\\Scripts\\python.exe scripts\\make_scene.py OUT_DIR [--minutes 30] [--source VIDEO]

A fixed backyard view (one frame of the Compilation_Video) with real person and car crops
from the same video moving along scripted paths: crossings in both directions, a person
stopping on the lawn, a car, two people together, someone walking outside the watched area.
Between events the scene is idle but not static: sensor noise, slow cloud-like brightness
drift, a swaying tree outside the motion ROI and one sudden "lights on" step - what makes
real motion gates and trackers misbehave.

Writes into OUT_DIR:

* ``videos/CAM-YARD_<YYYYMMDDTHHMMSS>.mkv`` - 10 s Segments (H.264, keyframe every 10 s), named
  like the Recorder names them, starting at ``--start`` (default today 20:00 local);
* ``cameras.yaml`` / ``rules.yaml`` - the camera (zone ``lawn``, line ``path``, ROI) and rules;
* ``ground_truth.yaml`` - every scripted object with its visible interval, zone entries, line
  crossings, plus ``expected_alerts``: the alerts of the rules computed by the real RuleEngine
  from the exact scripted positions at 3 fps (what a perfect detector + tracker would give).

The crops are taken from the local Compilation_Video at run time; no image is stored in the
repository.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from nab_sentry.ingest.transcode import ffmpeg_exe  # noqa: E402
from nab_sentry.live.camera_config import parse_cameras  # noqa: E402
from nab_sentry.rules.config import parse_rules  # noqa: E402
from nab_sentry.rules.engine import Obs, RuleEngine  # noqa: E402
from nab_sentry.rules.geometry import crossing, point_in_polygon  # noqa: E402

W, H, FPS, SEG_S = 1280, 720, 10, 10
CAMERA = "CAM-YARD"
SOURCE = _ROOT / "data" / "videos" / "CAM-SEC01_20261007T230010.mp4"
BACKGROUND_T = 127.0
# (time in the source video, box) of full-body people and side-view cars (picked by eye)
SPRITES = {
    "officer": (23.5, "person"), "red_jacket": (156.5, "person"), "blue_shirt": (368.5, "person"),
    "white_shirt": (380.5, "person"), "blue_sedan": (161.0, "car"), "white_sedan": (334.5, "car"),
}
LAWN = [[0.12, 0.50], [0.88, 0.50], [0.88, 0.97], [0.12, 0.97]]
PATH_LINE = [[0.5, 0.50], [0.5, 1.0]]  # arrow pointing down: walking left->right is "in"
ROI = [0.0, 0.35, 1.0, 1.0]
TREE = (0.80, 0.0, 0.97, 0.33)  # sways; outside the ROI

CAMERAS_YAML = {"cameras": [{"camera_id": CAMERA, "label": "Back yard (synthetic)", "source": "dshow:none",
                              "segment_s": SEG_S, "roi": ROI, "zones": {"lawn": LAWN},
                              "lines": {"path": PATH_LINE}}]}


def rules_yaml(start: datetime) -> dict:
    sched_end = (start + timedelta(minutes=10)).strftime("%H:%M")
    return {"rules": [
        {"id": "lawn-enter", "name": "Person on the lawn", "camera": CAMERA, "zone": "lawn",
         "classes": ["person"], "trigger": {"type": "enter"}, "cooldown_s": 60},
        {"id": "path-in", "name": "Came in along the path", "camera": CAMERA, "line": "path",
         "direction": "in", "classes": ["person", "car"], "trigger": {"type": "cross"}, "cooldown_s": 30},
        {"id": "path-out", "name": "Left along the path", "camera": CAMERA, "line": "path",
         "direction": "out", "classes": ["person", "car"], "trigger": {"type": "cross"}, "cooldown_s": 30},
        {"id": "loiter", "name": "Someone lingering on the lawn", "camera": CAMERA, "zone": "lawn",
         "classes": ["person"], "trigger": {"type": "dwell", "seconds": 25}, "cooldown_s": 120},
        {"id": "crowd", "name": "Two or more people on the lawn", "camera": CAMERA, "zone": "lawn",
         "classes": ["person"], "trigger": {"type": "count", "min": 2}, "cooldown_s": 120},
        {"id": "early-evening", "name": "Lawn visit in the first 10 minutes", "camera": CAMERA,
         "zone": "lawn", "classes": ["person"], "trigger": {"type": "enter"}, "cooldown_s": 0,
         "schedule": {"days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
                      "start": start.strftime("%H:%M"), "end": sched_end}},
    ]}


@dataclass(frozen=True)
class Walk:
    """One object moving through waypoints (normalized ground points) with dwell pauses."""

    name: str
    sprite: str
    t0: float  # seconds from the start of the footage
    points: tuple[tuple[float, float], ...]
    speed: float  # normalized units per second
    pauses: tuple[tuple[int, float], ...] = ()  # (waypoint index, seconds standing there)

    def schedule(self) -> list[tuple[float, float, float]]:
        """(t, x, y) knots of the piecewise-linear path."""
        knots, t = [], self.t0
        pause = dict(self.pauses)
        for i, (x, y) in enumerate(self.points):
            if i:
                px, py = self.points[i - 1]
                t += math.hypot(x - px, y - py) / self.speed
            knots.append((t, x, y))
            if i in pause:
                t += pause[i]
                knots.append((t, x, y))
        return knots

    def at(self, t: float) -> tuple[float, float] | None:
        k = self.schedule()
        if t < k[0][0] or t > k[-1][0]:
            return None
        for (ta, xa, ya), (tb, xb, yb) in zip(k, k[1:]):
            if ta <= t <= tb:
                f = 0.0 if tb == ta else (t - ta) / (tb - ta)
                return xa + f * (xb - xa), ya + f * (yb - ya)
        return k[-1][1], k[-1][2]

    @property
    def interval(self) -> tuple[float, float]:
        k = self.schedule()
        return k[0][0], k[-1][0]


def scenario(minutes: float) -> list[Walk]:
    """Events spread over the footage; the rest is idle."""
    s = minutes / 30  # scale the 30-minute plan's start times to the requested length
    off = lambda x: -0.08 if x < 0.5 else 1.08  # noqa: E731  (enter/leave just outside the frame)
    return [
        Walk("E1 crosses left to right", "red_jacket", 120 * s, ((off(0), 0.80), (off(1), 0.82)), 0.06),
        Walk("E2 crosses right to left", "blue_shirt", 300 * s, ((off(1), 0.70), (off(0), 0.74)), 0.07),
        Walk("E3 stops on the lawn", "white_shirt", 480 * s, ((off(0), 0.85), (0.35, 0.80), (off(0), 0.86)), 0.07,
             pauses=((1, 40.0),)),
        Walk("E4 car drives through", "blue_sedan", 700 * s, ((off(0), 0.95), (off(1), 0.95)), 0.18),
        Walk("E5a walks with E5b", "officer", 900 * s, ((off(0), 0.66), (off(1), 0.70)), 0.06),
        Walk("E5b walks with E5a", "red_jacket", 900 * s + 0.6, ((off(0), 0.88), (off(1), 0.90)), 0.06),
        Walk("E6 walks outside the lawn", "blue_shirt", 1100 * s, ((off(1), 0.42), (0.55, 0.40), (off(1), 0.41)), 0.06),
        Walk("E7 crosses in the dusk", "white_shirt", 1400 * s, ((off(0), 0.75), (off(1), 0.73)), 0.07),
        Walk("E8 car leaves", "white_sedan", 1600 * s, ((off(1), 0.93), (off(0), 0.93)), 0.2),
    ]


def brightness(t: float, total: float) -> float:
    """Cloud drift (+-10 % over minutes), dusk from 45 % to 60 % of the footage, lights-on step."""
    b = 1.0 + 0.10 * math.sin(2 * math.pi * t / 240.0) + 0.04 * math.sin(2 * math.pi * t / 37.0)
    dusk = min(1.0, max(0.0, (t - 0.45 * total) / (0.15 * total)))
    b *= 1.0 - 0.45 * dusk
    if t >= 0.70 * total:
        b *= 1.6  # floodlight switched on
    return b


# -- sprites -----------------------------------------------------------------------------------


def load_sprites(source: Path) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, str]]]:
    from nab_sentry.config import Config
    from nab_sentry.model_files import default_detector_factory
    from nab_sentry.startup import enable_offline_mode, require_models

    enable_offline_mode()
    cfg = Config()
    det = default_detector_factory(cfg, require_models(cfg.models_dir))
    cap = cv2.VideoCapture(str(source))

    def frame(t: float) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read {source} at {t} s")
        return img

    bg = cv2.resize(frame(BACKGROUND_T), (W, H), interpolation=cv2.INTER_AREA)
    sprites = {}
    for name, (t, cls) in SPRITES.items():
        img = frame(t)
        boxes = [d for d in det.detect(img) if d.cls == cls and d.high]
        if not boxes:
            raise RuntimeError(f"no {cls} found at {t} s for sprite {name}")
        d = max(boxes, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
        x1, y1, x2, y2 = d.box
        crop = img[y1:y2, x1:x2].copy()
        sprites[name] = (crop, cls, silhouette(crop))
    cap.release()
    return bg, sprites


def silhouette(crop: np.ndarray) -> np.ndarray:
    """Soft alpha mask of the object in its crop (GrabCut seeded with the crop rectangle), so
    only the person or car is blended into the scene, not their original background. Falls back
    to a soft ellipse when GrabCut finds too little."""
    h, w = crop.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (max(1, w // 20), max(1, h // 40), w - 2 * max(1, w // 20), h - 2 * max(1, h // 40))
    try:
        cv2.grabCut(crop, mask, rect, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                    5, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    except cv2.error:
        fg = np.zeros((h, w), np.float32)
    if fg.mean() < 0.2:
        return feather_mask(h, w)
    fg = cv2.GaussianBlur(fg, (0, 0), max(1.0, min(h, w) / 60))
    return np.clip(fg, 0, 1)[..., None]


def feather_mask(h: int, w: int) -> np.ndarray:
    """Soft elliptical alpha so pasted crops have no hard rectangle edge."""
    yy, xx = np.mgrid[0:h, 0:w]
    d = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    return np.clip((1.15 - d) / 0.35, 0, 1).astype(np.float32)[..., None]


def paste(canvas: np.ndarray, sprite: np.ndarray, cls: str, ground: tuple[float, float], t: float,
          alpha: np.ndarray) -> None:
    gx, gy = ground[0] * W, ground[1] * H
    scale_px = (130 + 300 * (ground[1] - 0.35)) if cls == "person" else (260 + 300 * (ground[1] - 0.35))
    sh, sw = sprite.shape[:2]
    if cls == "person":
        h = int(scale_px)
        w = max(4, int(sw * h / sh))
    else:
        w = int(scale_px * 1.6)
        h = max(4, int(sh * w / sw))
    bob = int(3 * math.sin(2 * math.pi * t * 1.8)) if cls == "person" else 0
    x1, y1 = int(gx - w / 2), int(gy - h + bob)
    img = cv2.resize(sprite, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    a = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W, x1 + w), min(H, y1 + h)
    if cx1 >= cx2 or cy1 >= cy2:
        return
    region = canvas[cy1:cy2, cx1:cx2].astype(np.float32)
    s_img = img[cy1 - y1:cy2 - y1, cx1 - x1:cx2 - x1]
    s_a = a[cy1 - y1:cy2 - y1, cx1 - x1:cx2 - x1]
    canvas[cy1:cy2, cx1:cx2] = (s_a * s_img + (1 - s_a) * region).astype(np.uint8)


def noise_bank(rng: np.random.Generator, n: int = 8) -> list[np.ndarray]:
    """A few frames of sensor noise (sigma 3 grey levels), reused at random: drawing fresh noise
    for every pixel of every frame made rendering several times slower than real time."""
    return [rng.normal(0, 3.0, (H, W, 3)).round().astype(np.int16) for _ in range(n)]


def render(bg: np.ndarray, sprites, walks: Sequence[Walk], t: float, total: float,
           rng: np.random.Generator, noise: Sequence[np.ndarray]) -> np.ndarray:
    frame = bg.copy()
    tx1, ty1, tx2, ty2 = (int(TREE[0] * W), int(TREE[1] * H), int(TREE[2] * W), int(TREE[3] * H))
    shift = int(round(4 * math.sin(2 * math.pi * t / 2.3)))  # tree swaying in the wind
    frame[ty1:ty2, tx1:tx2] = np.roll(bg[ty1:ty2, tx1:tx2], shift, axis=1)
    active = [(w.at(t), w) for w in walks]
    for pos, w in sorted([a for a in active if a[0] is not None], key=lambda a: a[0][1]):
        sprite, cls, alpha = sprites[w.sprite]
        paste(frame, sprite, cls, pos, t, alpha)
    f = cv2.convertScaleAbs(frame, alpha=brightness(t, total)).astype(np.int16)
    f += noise[int(rng.integers(len(noise)))]
    return np.clip(f, 0, 255).astype(np.uint8)


# -- ground truth ------------------------------------------------------------------------------


def ground_truth(walks: Sequence[Walk], sprites, start: datetime, total: float, rules) -> dict:
    step = 1 / 3  # the tracking rate
    objs = []
    for w in walks:
        a, b = w.interval
        cls = sprites[w.sprite][1]
        times = [a + i * step for i in range(int((b - a) / step) + 1)]
        pts = [(t, w.at(t)) for t in times]
        visible = [(t, p) for t, p in pts if p is not None and 0 <= p[0] <= 1]
        inside = [t for t, p in visible if point_in_polygon(p, [tuple(q) for q in LAWN])]
        crossings, prev = [], None
        for t, p in visible:
            if prev is not None:
                c = crossing(prev, p, tuple(PATH_LINE[0]), tuple(PATH_LINE[1]))
                if c:
                    crossings.append({"t": round(t, 2), "direction": "in" if c > 0 else "out"})
            prev = p
        objs.append({"name": w.name, "class": cls,
                     "visible": [round(visible[0][0], 2), round(visible[-1][0], 2)] if visible else None,
                     "in_lawn": [round(inside[0], 2), round(inside[-1], 2)] if inside else None,
                     "crossings": crossings})
    # Expected alerts: the real RuleEngine on perfect tracks (keys = object index, confirmed).
    eng = RuleEngine(rules)
    expected = []
    n = int(total / step)
    for i in range(n):
        t = i * step
        obs = []
        for k, w in enumerate(walks):
            p = w.at(t)
            if p is not None and 0 <= p[0] <= 1:
                obs.append(Obs(k, sprites[w.sprite][1], p, True))
        for f in eng.step(start.timestamp() + t, start + timedelta(seconds=t), obs):
            expected.append({"rule": f.rule_id, "t": round(t, 2),
                             "object": walks[f.track_key].name if f.track_key is not None else None})
    return {"camera": CAMERA, "start": start.isoformat(timespec="seconds"), "duration_s": total,
            "objects": objs, "expected_alerts": expected}


# -- main --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out", type=Path)
    p.add_argument("--minutes", type=float, default=30.0)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--start", default=None, help="footage start, local time YYYY-MM-DDTHH:MM (default today 20:00)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if not args.source.is_file():
        print(f"error: source video {args.source} not found", file=sys.stderr)
        return 1
    start = (datetime.fromisoformat(args.start) if args.start
             else datetime.now().replace(hour=20, minute=0, second=0, microsecond=0))
    total = args.minutes * 60
    out = args.out
    vids = out / "videos"
    vids.mkdir(parents=True, exist_ok=True)
    for old in vids.glob(f"{CAMERA}_*.mkv"):
        old.unlink()

    cameras = parse_cameras(CAMERAS_YAML)
    rules_doc = rules_yaml(start)
    rules = parse_rules(rules_doc, cameras)
    (out / "cameras.yaml").write_text(yaml.safe_dump(CAMERAS_YAML, sort_keys=False), encoding="utf-8")
    (out / "rules.yaml").write_text(yaml.safe_dump(rules_doc, sort_keys=False), encoding="utf-8")

    bg, sprites = load_sprites(args.source)
    walks = scenario(args.minutes)
    gt = ground_truth(walks, sprites, start, total, rules)
    (out / "ground_truth.yaml").write_text(yaml.safe_dump(gt, sort_keys=False), encoding="utf-8")

    tmp = vids / "seg%05d.mkv"
    enc = subprocess.Popen(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
         "-pix_fmt", "yuv420p", "-g", str(FPS * 2), "-force_key_frames", f"expr:gte(t,n_forced*{SEG_S})",
         "-f", "segment", "-segment_time", str(SEG_S), "-segment_format", "matroska",
         "-reset_timestamps", "1", str(tmp)], stdin=subprocess.PIPE)
    rng = np.random.default_rng(args.seed)
    noise = noise_bank(rng)
    n = int(total * FPS)
    for i in range(n):
        enc.stdin.write(render(bg, sprites, walks, i / FPS, total, rng, noise).tobytes())
        if i % (FPS * 60) == 0:
            print(f"  {i / FPS / 60:.0f} / {args.minutes:g} min", flush=True)
    enc.stdin.close()
    if enc.wait() != 0:
        print("error: ffmpeg failed", file=sys.stderr)
        return 1
    segs = sorted(vids.glob("seg*.mkv"))
    for k, f in enumerate(segs):
        f.rename(vids / f"{CAMERA}_{(start + timedelta(seconds=SEG_S * k)):%Y%m%dT%H%M%S}.mkv")
    print(f"{len(segs)} Segments, {len(gt['expected_alerts'])} expected alerts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
