"""Attribute tag accuracy per embedder on labelled crops (V2 tasks 4A.3, Step 4).

Usage::

    venv\\Scripts\\python.exe scripts\\attr_eval.py sheet [--db data\\nab_sentry.db]
    venv\\Scripts\\python.exe scripts\\attr_eval.py score MODEL [MODEL ...] [--labels eval\\attributes_compilation.yaml]

``sheet`` picks clear crops from the store - isolated (no other detection overlaps them), confident,
large enough, at least 3 s apart per class - and writes numbered contact sheets to
``data/bench/attrs/label_*.jpg`` plus a label template ``data/bench/attrs/labels_template.yaml``.
Fill in the colours you see and save the file as ``eval/attributes_compilation.yaml``.

``score`` tags each labelled crop on its own with :class:`~nab_sentry.describe.attributes.AttributeTagger`
(the product's groups, prompts and logit scale), using the crop vectors ``scripts/embed_eval.py
encode MODEL`` stored in ``data/bench/embed/MODEL.npz`` and MODEL's text encoder. It reports, per
group, right / wrong / no tag at several minimum probabilities (the product uses
:data:`~nab_sentry.describe.attributes.MIN_PROB`).

Labels: ``{vector_id: {class: car, colour: silver}}``; a colour label may list equivalents
(``silver|grey``); ``?`` = cannot tell (the crop is left out of that group).
"""

from __future__ import annotations

import argparse
import importlib.util
import sqlite3
import sys
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from nab_sentry.describe.attributes import DEFAULT_GROUPS, MIN_PROB, AttributeTagger  # noqa: E402

PERSON_MIN_H, VEHICLE_MIN_H, MIN_CONF, MIN_GAP_S = 90, 45, 0.6, 3.0
THRESHOLDS = (0.0, 0.5, 0.65, 0.8)
LABEL_KEYS = {"person": ("top colour",), "car": ("colour", "type"), "truck": ("colour", "type"),
              "bus": ("colour", "type")}


# ---------------------------------------------------------------- pure helpers


def isolated(box: tuple[int, int, int, int], others: Sequence[tuple[int, int, int, int]]) -> bool:
    x1, y1, x2, y2 = box
    return not any(o[0] < x2 and x1 < o[2] and o[1] < y2 and y1 < o[3] for o in others)


def spaced(items: Sequence[tuple[float, int]], gap: float) -> list[int]:
    """IDs of ``(time, id)`` items at least ``gap`` seconds apart, in time order."""
    out, last = [], None
    for t, i in sorted(items):
        if last is None or t - last >= gap:
            out.append(i)
            last = t
    return out


def matches(tag: str, label: str) -> bool:
    """A merged tag ("SUV/van") is right when any of its parts is one of the label's options."""
    return bool({t.strip().lower() for t in tag.split("/")} & {x.strip().lower() for x in str(label).split("|")})


def tally(results: Sequence[tuple[str | None, float, str]], min_prob: float) -> dict[str, int]:
    """(tag, probability, label) per crop -> right / wrong / none at ``min_prob``."""
    out = {"right": 0, "wrong": 0, "none": 0}
    for tag, prob, label in results:
        if tag is None or prob < min_prob:
            out["none"] += 1
        elif matches(tag, label):
            out["right"] += 1
        else:
            out["wrong"] += 1
    return out


# ---------------------------------------------------------------- sheet


def cmd_sheet(args) -> int:
    import cv2

    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = c.execute(
        "SELECT v.vector_id, v.frame_id, v.det_class, v.det_conf, v.x1, v.y1, v.x2, v.y2, f.offset_s, "
        "f.frame_idx, vi.src_path FROM vectors v JOIN frames f USING (frame_id) JOIN videos vi USING (video_id) "
        "WHERE v.kind = 'crop' ORDER BY v.vector_id").fetchall()
    c.close()
    by_frame: dict[int, list] = {}
    for r in rows:
        by_frame.setdefault(r[1], []).append(r)
    picks: dict[str, list[tuple[float, int]]] = {}
    info = {}
    for r in rows:
        vid, fid, cls, conf, x1, y1, x2, y2, off, idx, src = r
        if cls not in LABEL_KEYS or conf < MIN_CONF:
            continue
        if (y2 - y1) < (PERSON_MIN_H if cls == "person" else VEHICLE_MIN_H):
            continue
        if not isolated((x1, y1, x2, y2), [(o[4], o[5], o[6], o[7]) for o in by_frame[fid] if o[0] != vid]):
            continue
        group = "person" if cls == "person" else "vehicle"
        picks.setdefault(group, []).append((off, vid))
        info[vid] = (cls, off, idx, src, (x1, y1, x2, y2))
    chosen = [i for g in ("person", "vehicle") for i in spaced(picks.get(g, []), MIN_GAP_S)]
    out = _ROOT / "data" / "bench" / "attrs"
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    caps: dict[str, object] = {}
    for vid in chosen:
        cls, off, idx, src, (x1, y1, x2, y2) = info[vid]
        path = str(Path(src) if Path(src).is_absolute() else _ROOT / src)
        cap = caps.setdefault(path, cv2.VideoCapture(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            continue
        crop = img[y1:y2, x1:x2]
        s = 180 / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
        tile = np.full((206, 190, 3), 255, np.uint8)
        tile[:crop.shape[0], :crop.shape[1]] = crop
        cv2.putText(tile, f"{vid} {cls}", (3, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    for cap in caps.values():
        cap.release()
    for k in range(0, len(tiles), 30):
        part = tiles[k:k + 30] + [np.full_like(tiles[0], 255)] * (-len(tiles[k:k + 30]) % 6)
        sheet = np.vstack([np.hstack(part[i:i + 6]) for i in range(0, len(part), 6)])
        cv2.imwrite(str(out / f"label_{k // 30 + 1:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    lines = ["# vector_id: {class, colour labels}; use a|b for equivalents, ? for cannot tell", "crops:"]
    for vid in chosen:
        cls = info[vid][0]
        keys = ", ".join(f"{k.replace(' ', '_')}: ?" for k in LABEL_KEYS[cls])
        lines.append(f"  {vid}: {{class: {cls}, {keys}}}   # t={info[vid][1]:.0f}s")
    (out / "labels_template.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(chosen)} crops on {(len(tiles) + 29) // 30} sheet(s) in {out}")
    return 0


# ---------------------------------------------------------------- score


def _embed_eval():
    spec = importlib.util.spec_from_file_location("embed_eval_script", _ROOT / "scripts" / "embed_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def cmd_score(args) -> int:
    ee = _embed_eval()
    labels = yaml.safe_load(args.labels.read_text(encoding="utf-8"))["crops"]
    lines = []
    for model in args.models:
        data = np.load(_ROOT / "data" / "bench" / "embed" / f"{model}.npz")
        vecs = dict(zip(data["ids"].tolist(), data["images"]))
        cand = ee.make_candidate(model)
        tagger = AttributeTagger(lambda t: cand.encode_texts([t])[0], min_prob=0.0)
        per_group: dict[str, list[tuple[str | None, float, str]]] = {}
        for vid, lab in labels.items():
            vid = int(vid)
            if vid not in vecs:
                continue
            tags = tagger.tag(lab["class"], [vecs[vid]])
            for g in DEFAULT_GROUPS:
                if lab["class"] not in g.classes:
                    continue
                key = g.name.replace(" ", "_")
                label = str(lab.get(key, "?"))
                if label.strip() == "?":
                    continue
                tag, prob = tags.get(g.name, (None, 0.0))
                per_group.setdefault(g.name, []).append((tag, prob, label))
        for g, res in per_group.items():
            cells = []
            for th in THRESHOLDS:
                t = tally(res, th)
                cells.append(f"{t['right']}/{t['wrong']}/{t['none']}")
            lines.append(f"| {model} | {g} | {len(res)} | " + " | ".join(cells) + " |")
    head = ["Right / wrong / no tag per crop, by minimum probability (product: "
            f"{MIN_PROB}):", "",
            "| Model | Group | Crops | " + " | ".join(f"p >= {t}" for t in THRESHOLDS) + " |",
            "|---|---|---|" + "---|" * len(THRESHOLDS)]
    text = "\n".join(head + lines) + "\n"
    (_ROOT / "data" / "bench" / "attrs" / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("--db", type=Path, default=_ROOT / "data" / "nab_sentry.db")
    s = sub.add_parser("score")
    s.add_argument("models", nargs="+")
    s.add_argument("--labels", type=Path, default=_ROOT / "eval" / "attributes_compilation.yaml")
    args = p.parse_args(argv)
    return cmd_sheet(args) if args.cmd == "sheet" else cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
