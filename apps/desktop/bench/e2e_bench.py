"""End-to-end processing speed (desktop speed experiment, step 3).

Usage::

    venv-desktop\\Scripts\\python.exe apps\\desktop\\bench\\e2e_bench.py VIDEO_OR_FOLDER [...] [--device auto|CPU|GPU]

Creates a throw-away case under data/desktop/bench, adds the videos, runs both passes and prints
the wall time and x real time of each pass, what was stored, and the measured speeds.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "apps" / "desktop"))

from nab_find.library import add_videos  # noqa: E402
from nab_find.models import load_models  # noqa: E402
from nab_find.processor import Processor, total  # noqa: E402
from nab_find.store import CaseStore  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--device", default="auto")
    p.add_argument("--keep", action="store_true", help="keep the case folder")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    root = _ROOT / "data" / "desktop" / "bench"
    t0 = time.perf_counter()
    models = load_models(_ROOT / "models" / "desktop", _ROOT / "data" / "desktop" / "ovcache", prefer=args.device)
    load_s = time.perf_counter() - t0
    store = CaseStore.create(root, "bench")
    added = add_videos(store, args.paths)
    bad = [a for a in added if a.error]
    footage = sum(v.duration_s for v in store.videos())
    marks = {}
    proc = Processor(store, models, on_event=lambda e: marks.setdefault(e, time.perf_counter()))
    t1 = time.perf_counter()
    status = proc.run()
    t2 = time.perf_counter()
    quick_s = marks["quick_done"] - t1
    detail_s = t2 - marks["quick_done"]
    detail_footage = sum(total(store.spans(v.video_id, "detail")) for v in store.videos())
    c = store.counts()
    print(f"device {models.device}; model load {load_s:.1f} s; videos {len(added) - len(bad)} ok, {len(bad)} failed")
    print(f"footage {footage:.0f} s; status {status}")
    print(f"quick pass: {quick_s:.1f} s = {footage / quick_s:.1f}x real time")
    print(f"detail pass: {detail_footage:.0f} s of active footage ({100 * detail_footage / footage:.0f}%) in "
          f"{detail_s:.1f} s = {detail_footage / max(detail_s, 1e-6):.1f}x on active footage")
    print(f"whole case: {t2 - t1:.1f} s = {footage / (t2 - t1):.1f}x real time")
    print(f"stored: {c}")
    store.close()
    if not args.keep:
        shutil.rmtree(store.folder, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
