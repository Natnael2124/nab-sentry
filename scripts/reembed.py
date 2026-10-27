"""Convert the store to the configured embedder (V2 Step 4).

Usage::

    venv\\Scripts\\python.exe scripts\\reembed.py [--drop-missing] [--yes] [--set name=value ...]

Needed once after the embedder changes (search and ingest say so: "the stored vectors were made by
the embedder ..."). Stop ``run_live.ps1`` first: nothing else may write to the store meanwhile.

Re-encodes every stored frame and crop from its source video with the embedder in ``models/`` and
replaces the Vector_Index (see :mod:`nab_sentry.embed.reembed`); frames, tracks, alerts and the
activity log are kept. Takes about as long as embedding took during ingest (roughly 0.1 s per
stored vector on the Target_Hardware). ``--drop-missing`` deletes videos whose source file is gone
instead of refusing. Prints ``reembedded videos=N vectors=V dropped_videos=D``.

Exit codes: 0 done (or already on this embedder); 1 refused (missing sources without
``--drop-missing``, or not confirmed); 2 invalid Config; 4 manifest check failed; 6 model load
failure; 130 interrupted.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nab_sentry.startup import enable_offline_mode  # noqa: E402  (stdlib-only)

enable_offline_mode()

from nab_sentry.config import ConfigError, load_config, parse_set_args  # noqa: E402
from nab_sentry.embed.reembed import MissingSources, reembed  # noqa: E402
from nab_sentry.logging_setup import setup_logging  # noqa: E402
from nab_sentry.model_files import EMBEDDER_ID, default_encoder_factory  # noqa: E402
from nab_sentry.startup import require_models  # noqa: E402
from nab_sentry.store.db import MetadataStore  # noqa: E402


def main(argv: Sequence[str] | None = None, *, encoder=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--drop-missing", action="store_true",
                   help="delete stored videos whose source file no longer exists")
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="NAME=VALUE")
    args = p.parse_args(argv)
    try:
        cfg = load_config(parse_set_args(args.overrides))
        cfg.require_valid()
    except (ConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    setup_logging(cfg.logs_dir)
    if not cfg.db_path.is_file():
        print("nothing to do: there is no store yet")
        return 0
    if encoder is None:
        try:
            files = require_models(cfg.models_dir)
        except SystemExit as exc:
            return int(exc.code) if isinstance(exc.code, int) else 4
        try:
            encoder = default_encoder_factory(cfg, files)
        except Exception as exc:  # noqa: BLE001 - reported with the file
            print(f"error: model load failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 6
    wanted = getattr(encoder, "model_id", None) or EMBEDDER_ID
    db = MetadataStore(cfg.db_path)
    try:
        stored = db.embed_model()
        n = int(db.vector_ids().size)
        if stored == wanted and cfg.index_path.is_file():
            print(f"nothing to do: the store is already on {wanted} ({n} vectors)")
            return 0
        print(f"store: {n} vectors made by {stored or 'unknown'}; re-encoding them with {wanted}")
        print(f"estimated time: about {max(1, round(n * 0.12 / 60))} min. Stop run_live.ps1 first.")
        if not args.yes:
            try:
                answer = input("continue? [y/N] ").strip().lower()
            except EOFError:
                answer = ""
            if answer not in ("y", "yes"):
                print("not confirmed: nothing changed")
                return 1
        t0 = time.monotonic()
        last = [0.0]

        def progress(done: int, total: int) -> None:
            now = time.monotonic()
            if now - last[0] >= 10 or done == total:
                last[0] = now
                print(f"  {done}/{total} vectors, {now - t0:.0f} s", flush=True)
        try:
            rep = reembed(cfg, db, encoder, drop_missing=args.drop_missing, progress=progress)
        except MissingSources as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    except KeyboardInterrupt:
        print("interrupted: the old index is untouched; run again to convert", file=sys.stderr)
        return 130
    finally:
        db.close()
    print(f"reembedded videos={rep.videos} vectors={rep.vectors} dropped_videos={len(rep.dropped_videos)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
