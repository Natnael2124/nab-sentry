"""Contact sheets for labelling a video (V2 Requirement 2.1).

Usage::

    venv\\Scripts\\python.exe scripts\\contact_sheet.py VIDEO [--per-sheet 60] [--cols 10]
        [--cut-threshold 40]

Writes under ``data/bench/contact/<video stem>/``:

* ``sheet_NN.jpg`` - one thumbnail per second of video, each stamped with its offset
  (``mm:ss`` and seconds). Thumbnails that look like a scene cut (mean absolute grey-level
  difference to the previous second at or above ``--cut-threshold``) get a red border.
* ``<CAMERA_ID>.yaml`` - a Ground_Truth_File template with ``duration_s`` and the suggested
  ``scene_cuts`` filled in and one example incident to copy. Check the cuts, fill in the
  incidents, and save it as ``eval/ground_truth/<CAMERA_ID>.yaml``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from nab_sentry.config import Config  # noqa: E402
from nab_sentry.ingest.metadata import FILENAME_RE  # noqa: E402

THUMB_W, THUMB_H, LABEL_H, BORDER = 192, 108, 18, 3
CUT_SIG = (64, 36)  # grey signature size for the cut test


# ---------------------------------------------------------------- pure helpers


def fmt_offset(s: int) -> str:
    return f"{s // 60:02d}:{s % 60:02d} ({s}s)"


def signature(image: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, CUT_SIG, interpolation=cv2.INTER_AREA).astype(np.float32)


def cut_score(prev: np.ndarray | None, cur: np.ndarray) -> float:
    """Mean absolute grey-level difference (0..255) between two signatures; 0 for the first."""
    return 0.0 if prev is None else float(np.abs(cur - prev).mean())


def suggested_cuts(scores: Sequence[float], threshold: float) -> list[int]:
    """Seconds whose cut score is at or above ``threshold``; second 0 is always a cut."""
    return [0] + [s for s, v in enumerate(scores) if s > 0 and v >= threshold]


def tile(image: np.ndarray, second: int, is_cut: bool) -> np.ndarray:
    cell = np.full((THUMB_H + LABEL_H, THUMB_W, 3), 255, np.uint8)
    h, w = image.shape[:2]
    s = min(THUMB_W / w, THUMB_H / h)
    thumb = cv2.resize(image, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    y0, x0 = (THUMB_H - thumb.shape[0]) // 2, (THUMB_W - thumb.shape[1]) // 2
    cell[y0:y0 + thumb.shape[0], x0:x0 + thumb.shape[1]] = thumb
    if is_cut:
        cv2.rectangle(cell, (0, 0), (THUMB_W - 1, THUMB_H - 1), (0, 0, 255), BORDER)
    cv2.putText(cell, fmt_offset(second) + ("  CUT?" if is_cut and second else ""),
                (3, THUMB_H + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (0, 0, 200) if is_cut else (0, 0, 0), 1, cv2.LINE_AA)
    return cell


def grid(cells: list[np.ndarray], cols: int) -> np.ndarray:
    blank = np.full_like(cells[0], 255)
    cells = cells + [blank] * (-len(cells) % cols)
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    return np.vstack(rows)


def template_yaml(video: str, duration_s: float, cuts: Sequence[int]) -> str:
    cut_list = ", ".join(f"{c}.0" for c in cuts)
    return (
        f"# Ground_Truth_File for {video}. Check the suggested scene cuts against the contact\n"
        f"# sheets (red borders), then add one incident per labelled activity and save as\n"
        f"# eval/ground_truth/<CAMERA_ID>.yaml. Offsets are seconds from the start of the video.\n"
        f"# classes: person, bicycle, car, motorcycle, bus, truck. zone is optional.\n"
        f"video: {video}\n"
        f"duration_s: {duration_s:.1f}\n"
        f"scene_cuts: [{cut_list}]\n"
        f"incidents:\n"
        f"  # - {{start: 2.0, end: 14.0, classes: [person], description: \"man in black hoodie opens the gate\"}}\n"
    )


# ---------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("video", type=Path)
    p.add_argument("--per-sheet", type=int, default=60)
    p.add_argument("--cols", type=int, default=10)
    p.add_argument("--cut-threshold", type=float, default=40.0)
    args = p.parse_args(argv)

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not fps or fps <= 0:
        print(f"error: cannot read {args.video}", file=sys.stderr)
        return 1

    images: list[np.ndarray] = []
    scores: list[float] = []
    prev = None
    index, next_second = 0, 0
    while True:
        if index < round(next_second * fps):
            if not cap.grab():
                break
            index += 1
            continue
        ok, img = cap.read()
        if not ok or img is None:
            break
        index += 1
        sig = signature(img)
        scores.append(cut_score(prev, sig))
        prev = sig
        images.append(img)
        next_second += 1
    cap.release()
    if not images:
        print(f"error: no frames decoded from {args.video}", file=sys.stderr)
        return 1

    duration = index / fps
    cuts = suggested_cuts(scores, args.cut_threshold)
    cut_set = set(cuts)
    out_dir = Config().data_dir / "bench" / "contact" / args.video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("sheet_*.jpg"):
        old.unlink()
    cells = [tile(img, s, s in cut_set) for s, img in enumerate(images)]
    n_sheets = 0
    for i in range(0, len(cells), args.per_sheet):
        n_sheets += 1
        cv2.imwrite(str(out_dir / f"sheet_{n_sheets:02d}.jpg"), grid(cells[i:i + args.per_sheet], args.cols),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    m = FILENAME_RE.match(args.video.name)
    camera = m.group("cam") if m else args.video.stem
    (out_dir / f"{camera}.yaml").write_text(template_yaml(args.video.name, duration, cuts), encoding="utf-8")
    print(f"{len(images)} thumbnails on {n_sheets} sheet(s), {len(cuts) - 1} suggested scene cut(s), "
          f"duration {duration:.1f} s")
    print(f"written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
