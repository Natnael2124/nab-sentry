"""Prepare recorded footage of one camera for labelling and acceptance runs (V2 tasks 5.4, 17.3).

Usage::

    venv\\Scripts\\python.exe scripts\\prepare_footage.py CAMERA_ID [--out data\\real\\NAME]
        [--from 2026-10-26T18:00] [--to 2026-10-26T19:00] [--videos data\\videos]
        [--cameras config\\cameras.yaml] [--rules config\\rules.yaml] [--every 1]

Takes the camera's Segments (``CAMERA_ID_YYYYMMDDTHHMMSS.mkv``, as ``scripts/record.py`` writes
them) between ``--from`` and ``--to`` (default: all of them) and makes a folder that
``scripts/acceptance.py`` and ``scripts/gate_sweep.py`` accept like a generated scene::

    OUT/videos/*.mkv          copies of the Segments (the originals are not touched)
    OUT/cameras.yaml          this camera only, its source URL replaced (no password)
    OUT/rules.yaml            this camera's rules from config/rules.yaml
    OUT/contact/sheet_NN.jpg  one thumbnail every --every seconds, stamped with the clock time
    OUT/ground_truth.yaml     a template to fill in (format: nab_sentry/scene_truth.py)

Default OUT: ``data/real/<CAMERA_ID>_<start of the first Segment>``. Everything stays under
``data/`` (git-ignored).

Exit codes: 0 done; 1 no Segments found or none readable; 2 bad arguments or config.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from nab_sentry.config import Config  # noqa: E402
from nab_sentry.ingest.metadata import parse_filename  # noqa: E402
from nab_sentry.live.camera_config import CameraConfigError, load_cameras  # noqa: E402

THUMB_W, THUMB_H, LABEL_H = 192, 108, 18
PLACEHOLDER_SOURCE = "rtsp://recorded.invalid/footage"  # keeps the file valid, holds no password


# ---------------------------------------------------------------- pure helpers


def select_segments(names: Sequence[str], camera_id: str, t_from: datetime | None,
                    t_to: datetime | None) -> list[tuple[str, datetime]]:
    """This camera's Segment names with their start times, in time order, start in [from, to)."""
    out = []
    for n in names:
        parsed = parse_filename(n)
        if parsed is None or parsed[0] != camera_id or not n.lower().endswith(".mkv"):
            continue
        start = parsed[1]
        if (t_from is None or start >= t_from) and (t_to is None or start < t_to):
            out.append((n, start))
    return sorted(out, key=lambda x: x[1])


def camera_rules(rules_data: dict | None, camera_id: str) -> list[dict]:
    return [r for r in (rules_data or {}).get("rules") or [] if isinstance(r, dict) and r.get("camera") == camera_id]


def template_yaml(camera_id: str, start: datetime, duration_s: float, rules: Sequence[dict]) -> str:
    end = start + timedelta(seconds=duration_s)
    def trigger(r: dict) -> str:
        t = r.get("trigger") or {}
        extra = f" {t['seconds']} s" if "seconds" in t else f" min {t['min']}" if "min" in t else ""
        return f"{t.get('type')}{extra}"

    rule_lines = "".join(
        f"#   {r.get('id')}: {r.get('name', '')} ({trigger(r)}"
        f"{' zone ' + str(r['zone']) if r.get('zone') else ''}{' line ' + str(r['line']) if r.get('line') else ''}"
        f"{', ' + str(r['direction']) if r.get('direction') else ''}"
        f"{', schedule' if r.get('schedule') else ''}, classes {', '.join(r.get('classes') or [])})\n"
        for r in rules) or "#   (none: copy rules for this camera into rules.yaml first)\n"
    return (
        f"# Ground truth for {camera_id}, {start:%Y-%m-%d %H:%M:%S} to {end:%H:%M:%S}. Fill in with the\n"
        f"# contact sheets in contact/ (each thumbnail shows the clock time) and the videos in videos/.\n"
        f"# Times: 'HH:MM:SS' clock time, or seconds from start. Classes: person, bicycle, car,\n"
        f"# motorcycle, bus, truck.\n"
        f"#\n"
        f"# objects: EVERY person or vehicle that appears, also the ones no rule is about.\n"
        f"#   visible: first and last second it is on screen (inside the camera's ROI).\n"
        f"#   A parked car that is there the whole hour: leave it out (it never moves).\n"
        f"# expected_alerts: when each rule SHOULD fire (the moment the person enters the zone,\n"
        f"#   crosses the line, has stayed `seconds` in the zone, or the count is reached).\n"
        f"#   Mind cooldown_s: a second entry within the cooldown gives no second alert.\n"
        f"# Rules of this camera:\n"
        f"{rule_lines}"
        f"camera: {camera_id}\n"
        f"start: '{start.isoformat()}'\n"
        f"duration_s: {duration_s:.0f}\n"
        f"objects:\n"
        f"  # - name: man in a black jacket walks to the gate\n"
        f"  #   class: person\n"
        f"  #   visible: ['{(start + timedelta(minutes=4, seconds=10)):%H:%M:%S}', "
        f"'{(start + timedelta(minutes=4, seconds=31)):%H:%M:%S}']\n"
        f"expected_alerts:\n"
        f"  # - {{rule: {rules[0].get('id') if rules else 'RULE_ID'}, "
        f"t: '{(start + timedelta(minutes=4, seconds=22)):%H:%M:%S}'}}\n"
    )


def tile(image: np.ndarray, label: str) -> np.ndarray:
    cell = np.full((THUMB_H + LABEL_H, THUMB_W, 3), 255, np.uint8)
    h, w = image.shape[:2]
    s = min(THUMB_W / w, THUMB_H / h)
    thumb = cv2.resize(image, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    y0, x0 = (THUMB_H - thumb.shape[0]) // 2, (THUMB_W - thumb.shape[1]) // 2
    cell[y0:y0 + thumb.shape[0], x0:x0 + thumb.shape[1]] = thumb
    cv2.putText(cell, label, (3, THUMB_H + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return cell


def grid(cells: list[np.ndarray], cols: int) -> np.ndarray:
    blank = np.full_like(cells[0], 255)
    cells = cells + [blank] * (-len(cells) % cols)
    return np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])


# ---------------------------------------------------------------- main


def read_thumbs(path: Path, start: datetime, every: float) -> tuple[list[tuple[str, np.ndarray]], float]:
    """Thumbnails every ``every`` seconds of one Segment, labelled with the clock time, and the
    Segment's duration."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    out, index, next_t = [], 0, 0.0
    if fps <= 0:
        cap.release()
        return out, 0.0
    while True:
        if index < round(next_t * fps):
            if not cap.grab():
                break
            index += 1
            continue
        ok, img = cap.read()
        if not ok or img is None:
            break
        index += 1
        out.append(((start + timedelta(seconds=next_t)).strftime("%H:%M:%S"), img))
        next_t += every
    cap.release()
    return out, index / fps


def parse_time(text: str) -> datetime:
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a date-time like 2026-10-26T18:00: {text!r}") from None


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("camera_id")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--from", dest="t_from", type=parse_time, default=None)
    p.add_argument("--to", dest="t_to", type=parse_time, default=None)
    p.add_argument("--videos", type=Path, default=None)
    p.add_argument("--cameras", type=Path, default=_ROOT / "config" / "cameras.yaml")
    p.add_argument("--rules", type=Path, default=_ROOT / "config" / "rules.yaml")
    p.add_argument("--every", type=float, default=1.0, help="seconds between thumbnails")
    p.add_argument("--per-sheet", type=int, default=120)
    p.add_argument("--cols", type=int, default=12)
    args = p.parse_args(argv)
    if args.every <= 0 or args.per_sheet < 1 or args.cols < 1:
        print("error: --every, --per-sheet and --cols must be positive", file=sys.stderr)
        return 2

    try:
        cams = {c.camera_id: c for c in load_cameras(args.cameras)}
    except CameraConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    cam = cams.get(args.camera_id)
    if cam is None:
        print(f"error: {args.camera_id} is not in {args.cameras} (cameras: {', '.join(cams) or 'none'})",
              file=sys.stderr)
        return 2
    rules = []
    if args.rules.is_file():
        rules = camera_rules(yaml.safe_load(args.rules.read_text(encoding="utf-8")), args.camera_id)

    videos_dir = args.videos or Config().videos_dir
    names = [f.name for f in videos_dir.glob("*.mkv")] if videos_dir.is_dir() else []
    segs = select_segments(names, args.camera_id, args.t_from, args.t_to)
    if not segs:
        print(f"error: no {args.camera_id}_*.mkv Segments in {videos_dir} in that time range", file=sys.stderr)
        return 1
    start = segs[0][1]
    out = args.out or (Config().data_dir / "real" / f"{args.camera_id}_{start:%Y%m%dT%H%M%S}")
    (out / "videos").mkdir(parents=True, exist_ok=True)
    contact = out / "contact"
    shutil.rmtree(contact, ignore_errors=True)
    contact.mkdir()

    cells, end_s, unreadable, n_sheets = [], 0.0, [], 0

    def flush(final: bool = False) -> None:
        nonlocal cells, n_sheets
        while len(cells) >= args.per_sheet or (final and cells):
            n_sheets += 1
            cv2.imwrite(str(contact / f"sheet_{n_sheets:02d}.jpg"), grid(cells[:args.per_sheet], args.cols),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            cells = cells[args.per_sheet:]

    for name, seg_start in segs:
        src = videos_dir / name
        dst = out / "videos" / name
        if not dst.exists():
            shutil.copy2(src, dst)
        thumbs, dur = read_thumbs(dst, seg_start, args.every)
        if not thumbs:
            unreadable.append(name)
            continue
        cells += [tile(img, label) for label, img in thumbs]
        end_s = max(end_s, (seg_start - start).total_seconds() + dur)
        flush()
    flush(final=True)
    if end_s <= 0:
        print("error: none of the Segments could be read", file=sys.stderr)
        return 1

    spec = cam.to_dict()
    spec["source"] = PLACEHOLDER_SOURCE
    (out / "cameras.yaml").write_text(
        "# Copied from config/cameras.yaml by prepare_footage.py; source replaced (no password).\n"
        + yaml.safe_dump({"cameras": [spec]}, sort_keys=False), encoding="utf-8")
    (out / "rules.yaml").write_text(
        "# This camera's rules from config/rules.yaml. Edit here to try other rules on the same footage\n"
        "# (then update expected_alerts in ground_truth.yaml).\n"
        + yaml.safe_dump({"rules": rules}, sort_keys=False), encoding="utf-8")
    gt = out / "ground_truth.yaml"
    if gt.exists():
        print(f"kept your existing {gt}")
    else:
        gt.write_text(template_yaml(args.camera_id, start, end_s, rules), encoding="utf-8")

    print(f"{len(segs)} Segments, {start:%Y-%m-%d %H:%M:%S} + {end_s / 60:.1f} min, "
          f"{n_sheets} contact sheet(s), {len(rules)} rule(s)")
    if unreadable:
        print(f"warning: {len(unreadable)} Segment(s) could not be read: {', '.join(unreadable[:5])}")
    gaps = [(a[1], b[1]) for a, b in zip(segs, segs[1:]) if (b[1] - a[1]).total_seconds() > 3 * cam.segment_s]
    if gaps:
        print(f"warning: {len(gaps)} gap(s) in the recording, first at {gaps[0][0]:%H:%M:%S}")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
