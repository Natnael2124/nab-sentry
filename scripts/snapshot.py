"""Save one frame of a camera with a coordinate grid, for drawing ROI / zones / lines (V2 Req 6.3).

Usage::

    venv\\Scripts\\python.exe scripts\\snapshot.py CAMERA_ID [--cameras config\\cameras.yaml]

Writes ``data/live/<CAMERA_ID>_snapshot.jpg`` (the raw frame) and
``data/live/<CAMERA_ID>_grid.jpg`` (grid every 0.1 with labels, plus the camera's current ROI in
yellow, zones in green and lines in magenta). Coordinates in ``config/cameras.yaml`` are these
0..1 fractions.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from nab_sentry.config import Config  # noqa: E402
from nab_sentry.ingest.transcode import ffmpeg_exe  # noqa: E402
from nab_sentry.live.camera_config import CameraConfigError, CameraSpec, load_cameras, mask_url  # noqa: E402
from nab_sentry.live.recorder import snapshot_command  # noqa: E402

DEFAULT_CAMERAS = _ROOT / "config" / "cameras.yaml"


def draw_overlay(image: np.ndarray, spec: CameraSpec) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    px = lambda p: (int(round(p[0] * (w - 1))), int(round(p[1] * (h - 1))))  # noqa: E731
    for i in range(1, 10):
        x, y = int(w * i / 10), int(h * i / 10)
        cv2.line(out, (x, 0), (x, h - 1), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(out, (0, y), (w - 1, y), (255, 255, 255), 1, cv2.LINE_AA)
        for txt, org in ((f"{i / 10:.1f}", (x + 3, 14)), (f"{i / 10:.1f}", (3, y - 4))):
            cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if spec.roi:
        cv2.rectangle(out, px(spec.roi[:2]), px(spec.roi[2:]), (0, 255, 255), 2)
    for name, poly in spec.zones.items():
        pts = np.array([px(p) for p in poly], np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)
        cv2.putText(out, name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    for name, (a, b) in spec.lines.items():
        cv2.arrowedLine(out, px(a), px(b), (255, 0, 255), 2)
        cv2.putText(out, name, px(a), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("camera_id")
    p.add_argument("--cameras", type=Path, default=DEFAULT_CAMERAS)
    args = p.parse_args(argv)
    try:
        specs = {s.camera_id: s for s in load_cameras(args.cameras)}
    except CameraConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    spec = specs.get(args.camera_id)
    if spec is None:
        print(f"error: unknown camera id {args.camera_id}", file=sys.stderr)
        return 2

    live = Config().data_dir / "live"
    live.mkdir(parents=True, exist_ok=True)
    raw = live / f"{spec.camera_id}_snapshot.jpg"
    proc = subprocess.run(snapshot_command(spec, raw, ffmpeg_exe()), capture_output=True, timeout=60)
    image = cv2.imread(str(raw)) if proc.returncode == 0 else None
    if image is None:
        err = mask_url(proc.stderr.decode("utf-8", errors="replace").strip())
        print(f"error: could not grab a frame from {mask_url(spec.source)}: {err}", file=sys.stderr)
        return 1
    grid = live / f"{spec.camera_id}_grid.jpg"
    cv2.imwrite(str(grid), draw_overlay(image, spec))
    print(f"{image.shape[1]}x{image.shape[0]} frame saved: {raw}\ngrid: {grid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
