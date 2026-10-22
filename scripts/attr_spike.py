"""Spike: zero-shot attribute tags from OpenCLIP crop embeddings (no VLM).

Samples person / vehicle crops from the ingested Compilation_Video, scores each against small
attribute groups, and writes contact sheets with the top tag per group to data/bench/attrs/.
"""
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nab_sentry.startup import enable_offline_mode, require_models  # noqa: E402

enable_offline_mode()
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from nab_sentry.config import Config  # noqa: E402
from nab_sentry.model_files import default_encoder_factory  # noqa: E402

PERSON_GROUPS = {
    "top colour": [f"a person wearing a {c} top" for c in
                   ("red", "blue", "black", "white", "grey", "green", "yellow", "brown")],
    "clothing": ["a person wearing a t-shirt", "a person wearing a hoodie", "a person wearing a jacket",
                 "a person wearing a police uniform", "a person wearing a dress", "a person wearing a suit"],
    "carrying": ["a person with empty hands", "a person carrying a bag", "a person carrying a box",
                 "a person holding a phone", "a person pushing a lawn mower"],
    "headwear": ["a person with no hat", "a person wearing a cap", "a person wearing a helmet"],
}
VEHICLE_GROUPS = {
    "colour": [f"a {c} vehicle" for c in ("white", "black", "silver", "red", "blue", "grey")],
    "type": ["a sedan car", "a pickup truck", "an SUV", "a van", "a police car", "a bus", "a motorcycle"],
}
LOGIT_SCALE = 100.0  # CLIP's learned temperature
MIN_PROB = 0.40      # below this the tag is reported as "?"


def short(prompt: str) -> str:
    for p in ("a person wearing a ", "a person with ", "a person carrying a ", "a person holding a ",
              "a person pushing a ", "a person ", "an ", "a "):
        if prompt.startswith(p):
            return prompt[len(p):]
    return prompt


def main() -> None:
    cfg = Config()
    files = require_models(cfg.models_dir)
    enc = default_encoder_factory(cfg, files)
    db = sqlite3.connect(cfg.db_path)
    rows = db.execute("""
        SELECT v.det_class, v.x1, v.y1, v.x2, v.y2, f.offset_s, vi.src_path
        FROM vectors v JOIN frames f ON f.frame_id = v.frame_id JOIN videos vi ON vi.video_id = f.video_id
        WHERE v.kind = 'crop' AND (v.y2 - v.y1) >= 90 ORDER BY f.offset_s""").fetchall()
    people = [r for r in rows if r[0] == "person"]
    vehicles = [r for r in rows if r[0] in ("car", "truck", "bus")]
    pick = lambda xs, n: [xs[int(i * len(xs) / n)] for i in range(min(n, len(xs)))]  # noqa: E731

    t0 = time.perf_counter()
    text = {}
    for groups in (PERSON_GROUPS, VEHICLE_GROUPS):
        for g, prompts in groups.items():
            text[g] = np.stack([enc.encode_text(p) for p in prompts])
    print(f"text prompts encoded once: {time.perf_counter() - t0:.1f} s (cached in production)")

    out = cfg.data_dir / "bench" / "attrs"
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(rows[0][6]))
    for name, sample, groups in (("people", pick(people, 24), PERSON_GROUPS),
                                 ("vehicles", pick(vehicles, 12), VEHICLE_GROUPS)):
        crops = []
        for cls, x1, y1, x2, y2, off, _ in sample:
            cap.set(cv2.CAP_PROP_POS_MSEC, off * 1000)
            ok, img = cap.read()
            crops.append((off, img[y1:y2, x1:x2]))
        vecs = enc.encode_images([c for _, c in crops])
        t1 = time.perf_counter()
        cells = []
        for (off, crop), v in zip(crops, vecs):
            lines = [f"t={off:.0f}s"]
            for g, prompts in groups.items():
                logits = LOGIT_SCALE * (text[g] @ v)
                p = np.exp(logits - logits.max()); p /= p.sum()
                i = int(p.argmax())
                lines.append(f"{short(prompts[i]) if p[i] >= MIN_PROB else '?'} {p[i]:.2f}")
            h, w = crop.shape[:2]
            s = min(160 / w, 200 / h)
            img = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))))
            cell = np.full((200 + 16 * len(lines) + 6, 170, 3), 255, np.uint8)
            cell[:img.shape[0], :img.shape[1]] = img
            for k, line in enumerate(lines):
                cv2.putText(cell, line, (2, 214 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
            cells.append(cell)
            print(name, " | ".join(lines))
        per_crop_ms = (time.perf_counter() - t1) * 1000 / len(crops)
        print(f"{name}: tagging cost {per_crop_ms:.3f} ms/crop (embeddings already exist at ingest)")
        cols = 6
        while len(cells) % cols:
            cells.append(np.full_like(cells[0], 255))
        sheet = np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])
        cv2.imwrite(str(out / f"{name}.jpg"), sheet)
    print("written to", out)


if __name__ == "__main__":
    main()
