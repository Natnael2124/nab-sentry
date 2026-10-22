"""Notifier: deliver alerts to Telegram, answer bot commands, raise health alerts (V2 Req 14-15).

Usage (PowerShell)::

    $env:NAB_TELEGRAM_TOKEN = "123456:ABC..."      # from @BotFather
    $env:NAB_TELEGRAM_CHATS = "123456789"          # comma-separated chat IDs allowed to use the bot
    venv\\Scripts\\python.exe scripts\\notify.py [--cameras config\\cameras.yaml]
        [--rules config\\rules.yaml] [--set name=value ...]

Don't know your chat ID? Start the Notifier with any number in ``NAB_TELEGRAM_CHATS``, send the
bot a message, and read the log line "ignoring Telegram chat <id>".

This is the only NAB Sentry process that uses the network, and only to ``api.telegram.org``
over HTTPS. The token is never printed. Ctrl+C stops it; alerts not yet delivered stay in the
store and go out on the next start.

Exit codes: 0 stopped; 2 missing or invalid settings.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nab_sentry.config import ConfigError, load_config, parse_set_args  # noqa: E402
from nab_sentry.live.camera_config import CameraConfigError, load_cameras  # noqa: E402
from nab_sentry.logging_setup import get_logger, setup_logging  # noqa: E402
from nab_sentry.rules.config import RulesConfigError, load_rules  # noqa: E402

log = get_logger("notify.cli")
EXIT_OK, EXIT_CONFIG = 0, 2


def parse_chats(raw: str | None) -> list[int]:
    """``"123, -456"`` -> [123, -456]; raises ValueError on anything else."""
    if not raw or not raw.strip():
        raise ValueError("NAB_TELEGRAM_CHATS is not set")
    chats = []
    for part in raw.split(","):
        part = part.strip()
        if not part.lstrip("-").isdigit():
            raise ValueError(f"NAB_TELEGRAM_CHATS: {part!r} is not a chat ID")
        chats.append(int(part))
    return chats


def vlm_describer(cfg):
    """The on-demand VLM's describe function, or None until Step 3's VLM module exists."""
    try:
        from nab_sentry.describe.vlm import describer_from_config
    except ImportError:
        return None
    return describer_from_config(cfg)


def main(argv: Sequence[str] | None = None, *, transport=None, stop: threading.Event | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cameras", type=Path, default=_ROOT / "config" / "cameras.yaml")
    p.add_argument("--rules", type=Path, default=_ROOT / "config" / "rules.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--vlm", action="store_true",
                   help="enable /describe and verified rules with the local Ollama VLM (Step 3)")
    args = p.parse_args(argv)

    token = os.environ.get("NAB_TELEGRAM_TOKEN", "").strip()
    try:
        chats = parse_chats(os.environ.get("NAB_TELEGRAM_CHATS"))
        if not token and transport is None:
            raise ValueError("NAB_TELEGRAM_TOKEN is not set (create a bot with @BotFather)")
        cfg = load_config(parse_set_args(args.overrides))
        cfg.require_valid()
        cameras = load_cameras(args.cameras) if args.cameras.is_file() else []
        rules = load_rules(args.rules, cameras) if args.rules.is_file() else []
    except (ValueError, ConfigError, CameraConfigError, RulesConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    from nab_sentry.notify.commands import Commands
    from nab_sentry.notify.health import HealthMonitor
    from nab_sentry.notify.outbox import Outbox
    from nab_sentry.notify.service import Notifier
    from nab_sentry.notify.telegram import HttpxTransport, TelegramClient
    from nab_sentry.store.db import MetadataStore

    setup_logging(cfg.logs_dir)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    db = MetadataStore(cfg.db_path)
    try:
        cam_info = {c.camera_id: (c.label, float(c.segment_s)) for c in cameras}
        labels = {cam: label for cam, label in db.cameras()} | {k: v[0] for k, v in cam_info.items()}
        health = HealthMonitor(cfg, db, cam_info, started_s=time.time())
        client = TelegramClient(transport or HttpxTransport(token))
        outbox = Outbox(cfg, db, client, chats, camera_labels=labels,
                        segment_s=lambda cam: cam_info.get(cam, ("", 10.0))[1])
        from nab_sentry.notify.search import TodaySearch

        def load_encoder():
            from nab_sentry.model_files import default_encoder_factory
            from nab_sentry.startup import verify_manifest

            manifest = verify_manifest(cfg.models_dir)
            if manifest.failures:
                raise RuntimeError("models are not provisioned (run scripts\fetch_models.py)")
            return default_encoder_factory(cfg, manifest.files)

        commands = Commands(db, labels, rules, live_dir=cfg.live_dir, offline_after_s=health.offline_after_s,
                            thumbs_dir=cfg.thumbs_dir, searcher=TodaySearch(cfg, db, load_encoder),
                            describer=vlm_describer(cfg) if args.vlm else None)
        notifier = Notifier(db, client, chats, commands, outbox, camera_labels=labels,
                            health=health.check)
        log.info("notifier started for %d chat(s), %d camera(s), %d rule(s)", len(chats), len(cam_info), len(rules))
        print(f"notifier running for {len(chats)} chat(s); Ctrl+C to stop", flush=True)
        stop = stop or threading.Event()
        try:
            notifier.run(stop)
        except KeyboardInterrupt:
            pass
        return EXIT_OK
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
