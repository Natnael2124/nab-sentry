"""Replay a video file as if it were a live camera (V2 Requirement 17.1).

Usage::

    venv\\Scripts\\python.exe scripts\\replay_camera.py VIDEO CAMERA_ID [--segment-s 10]
        [--encode] [--loop]

Feeds VIDEO through ffmpeg in real time (``-re``) into ``<CAMERA_ID>_<YYYYMMDDTHHMMSS>.mkv``
Segments in ``data/videos/``, named from the current wall clock exactly like the Recorder.
Without ``--encode`` the video stream is copied, so Segments are cut at the source's keyframes
and may be longer than ``--segment-s``; ``--encode`` re-encodes with x264 and forces a keyframe
on every boundary. ``--loop`` repeats the file until Ctrl+C.
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

from nab_sentry.config import Config  # noqa: E402
from nab_sentry.ingest.metadata import CAMERA_ID_RE  # noqa: E402
from nab_sentry.ingest.transcode import ffmpeg_exe  # noqa: E402
from nab_sentry.live.recorder import replay_command  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("video", type=Path)
    p.add_argument("camera_id")
    p.add_argument("--segment-s", type=int, default=10)
    p.add_argument("--encode", action="store_true")
    p.add_argument("--loop", action="store_true")
    args = p.parse_args(argv)

    if not args.video.is_file():
        print(f"error: {args.video} not found", file=sys.stderr)
        return 1
    if not CAMERA_ID_RE.fullmatch(args.camera_id):
        print("error: camera id must be 1-64 letters, digits or hyphens", file=sys.stderr)
        return 2
    if not 5 <= args.segment_s <= 300:
        print("error: --segment-s must be 5 to 300", file=sys.stderr)
        return 2
    out = Config().videos_dir
    out.mkdir(parents=True, exist_ok=True)
    argv_ff = replay_command(args.video, args.camera_id, out, args.segment_s, ffmpeg_exe(),
                             encode=args.encode, loop=args.loop)
    print(f"replaying {args.video.name} as {args.camera_id} into {out}; Ctrl+C to stop", flush=True)
    try:
        return subprocess.run(argv_ff).returncode
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
