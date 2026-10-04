"""Synthetic_Generator CLI (Requirement 15).

Usage::

    venv\\Scripts\\python.exe scripts\\make_synthetic.py [--out data/videos] [--seed 0]
        [--duration 90] [--fps 10] [--width 640] [--height 480] [--camera-id CAM-SYN01]
        [--label "Synthetic Yard"] [--start-time 2025-01-01T08:00:00]
        [--red START END] [--blue START END]

Writes ``<CAMERA_ID>_<YYYYMMDDTHHMMSS>.mp4``, its Sidecar_File and ``ground_truth.json``.
Exits 1 with an error naming the problem (invalid setting, unwritable output dir, ffmpeg
failure) and no new output files.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nab_sentry.synthetic import (  # noqa: E402
    SyntheticError,
    SyntheticObject,
    SyntheticSpec,
    write_synthetic,
)

DEFAULT_OUT = _ROOT / "data" / "videos"


def build_parser() -> argparse.ArgumentParser:
    d = SyntheticSpec()
    p = argparse.ArgumentParser(description="Generate a synthetic test video with ground truth.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--duration", type=float, default=d.duration_s, help="seconds")
    p.add_argument("--fps", type=float, default=d.fps)
    p.add_argument("--width", type=int, default=d.width)
    p.add_argument("--height", type=int, default=d.height)
    p.add_argument("--camera-id", default=d.camera_id)
    p.add_argument("--label", default=d.label)
    p.add_argument("--start-time", default=d.start_time.isoformat(), help="ISO 8601")
    red, blue = d.objects
    p.add_argument("--red", nargs=2, type=float, metavar=("START", "END"),
                   default=[red.start_s, red.end_s], help="red square interval [START, END) s")
    p.add_argument("--blue", nargs=2, type=float, metavar=("START", "END"),
                   default=[blue.start_s, blue.end_s], help="blue circle interval [START, END) s")
    return p


def spec_from_args(args: argparse.Namespace) -> SyntheticSpec:
    try:
        start_time = datetime.fromisoformat(args.start_time)
    except ValueError as e:
        raise SyntheticError(f"invalid setting start_time: {args.start_time!r}") from e
    fps = int(args.fps) if float(args.fps).is_integer() else args.fps
    return SyntheticSpec(
        seed=args.seed,
        duration_s=args.duration,
        fps=fps,
        width=args.width,
        height=args.height,
        camera_id=args.camera_id,
        label=args.label,
        start_time=start_time,
        objects=(
            SyntheticObject("red square", "square", "red", args.red[0], args.red[1]),
            SyntheticObject("blue circle", "circle", "blue", args.blue[0], args.blue[1]),
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = spec_from_args(args)
        video, sidecar, gt = write_synthetic(spec, args.out)
    except SyntheticError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {video}")
    print(f"wrote {sidecar}")
    print(f"wrote {gt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
