"""Record live cameras into Segments (V2 Requirement 7).

Usage::

    venv\\Scripts\\python.exe scripts\\record.py [CAMERA_ID ...] [--cameras config\\cameras.yaml]

Starts one supervised ffmpeg recorder per camera (all cameras in the Camera_Config when none
are named), writing ``<CAMERA_ID>_<YYYYMMDDTHHMMSS>.mkv`` Segments into ``data/videos/``,
heartbeats into ``data/live/`` and masked ffmpeg logs into ``data/logs/``. Ctrl+C stops every
recorder gracefully (ffmpeg finalises the current Segment).

Exit codes: 0 stopped normally; 2 invalid or missing Camera_Config, or unknown camera ID.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nab_sentry.config import Config  # noqa: E402
from nab_sentry.ingest.transcode import ffmpeg_exe  # noqa: E402
from nab_sentry.live.camera_config import CameraConfigError, load_cameras, mask_url  # noqa: E402
from nab_sentry.live.recorder import Recorder  # noqa: E402
from nab_sentry.logging_setup import get_logger, setup_logging  # noqa: E402

DEFAULT_CAMERAS = _ROOT / "config" / "cameras.yaml"
log = get_logger("record")


def select(specs, ids: Sequence[str]):
    if not ids:
        return specs, []
    by_id = {s.camera_id: s for s in specs}
    return [by_id[i] for i in ids if i in by_id], [i for i in ids if i not in by_id]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("camera_ids", nargs="*")
    p.add_argument("--cameras", type=Path, default=DEFAULT_CAMERAS)
    args = p.parse_args(argv)

    cfg = Config()
    setup_logging(cfg.logs_dir)
    try:
        specs = load_cameras(args.cameras)
    except CameraConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if not args.cameras.is_file():
            print("copy config\\cameras.example.yaml to config\\cameras.yaml and edit it", file=sys.stderr)
        return 2
    chosen, unknown = select(specs, args.camera_ids)
    if unknown:
        print(f"error: unknown camera id(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    ffmpeg = ffmpeg_exe()
    recorders = [Recorder(s, cfg.videos_dir, cfg.data_dir / "live", cfg.logs_dir, ffmpeg) for s in chosen]
    threads = [threading.Thread(target=r.run, name=f"rec-{r.spec.camera_id}", daemon=True) for r in recorders]
    for r, t in zip(recorders, threads):
        log.info("recording %s (%s) every %d s via %s", r.spec.camera_id, mask_url(r.spec.source),
                 r.spec.segment_s, r.spec.resolved_encoder)
        t.start()
    print(f"recording {len(recorders)} camera(s) into {cfg.videos_dir}; Ctrl+C to stop", flush=True)
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.5)
    except KeyboardInterrupt:
        print("stopping ...", flush=True)
        for r in recorders:
            r.stop()
        for t in threads:
            t.join(timeout=20)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
