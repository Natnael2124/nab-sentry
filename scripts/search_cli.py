"""Search CLI, the Phase 1 demo tool (Requirements 8.15, 9.12; design "Scripts" table).

Usage::

    venv\\Scripts\\python.exe scripts\\search_cli.py "query" [--camera ID] [--start ISO]
        [--end ISO] [--cls CLASS] [--limit N] [--set name=value ...]

Steps: offline mode -> Config (``--set`` overrides, validated) -> logging -> model manifest
check -> load Embedder -> open Metadata_Store and Vector_Index -> ``SearchEngine.check_ready``
(refuses when the index IDs differ from the ``vectors`` IDs, 8.15) -> ``search``. Prints one
table row per Event in ranked order (9.12): camera, label, absolute ISO start/end, offsets
within the video, score, relative score, thumbnail file name.

``--start``/``--end`` are ISO 8601; a value without a UTC offset is local time (1.12).

Exit codes (design "Exit codes"): 0 success (including no results); 1 invalid search input
(bad ISO time, start later than end, unknown class, empty query, limit out of range);
2 invalid Config; 4 manifest check failed; 5 vector index unavailable or inconsistent;
6 model load failure; 130 interrupted.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nab_sentry.startup import enable_offline_mode  # noqa: E402  (stdlib-only)

# Must run before torch / open_clip are imported (Requirement 13.3).
enable_offline_mode()

from nab_sentry.config import Config, ConfigError, load_config, parse_set_args  # noqa: E402
from nab_sentry.embed.clip_encoder import EmptyQueryError  # noqa: E402
from nab_sentry.errors import ModelMissingError  # noqa: E402
from nab_sentry.logging_setup import get_logger, setup_logging  # noqa: E402
from nab_sentry.search.clustering import Event  # noqa: E402
from nab_sentry.search.engine import (  # noqa: E402
    FilterError,
    SearchEngine,
    SearchFilter,
    SearchUnavailable,
)
from nab_sentry.startup import require_models  # noqa: E402
from nab_sentry.store.db import MetadataStore, as_aware, iso_ms  # noqa: E402
from nab_sentry.store.vector_index import DEFAULT_DIM, IndexUnavailable, VectorIndex  # noqa: E402

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG = 2
EXIT_MODELS = 4
EXIT_INDEX = 5
EXIT_MODEL_LOAD = 6
EXIT_INTERRUPTED = 130

EMBEDDER_REL = "open_clip/ViT-B-32-laion2b_s34b_b79k/open_clip_pytorch_model.bin"

# (cfg, manifest files rel->abs) -> Encoder
EncoderFactory = Callable[[Config, dict[str, Path]], Any]

HEADERS = ("camera", "label", "start", "end", "offsets", "score", "rel", "thumbnail")

log = get_logger("search.cli")


class InputError(ValueError):
    """Invalid command-line search input (exit 1)."""


def default_encoder_factory(cfg: Config, files: dict[str, Path]) -> Any:
    from nab_sentry.embed.clip_encoder import OpenClipEncoder  # torch/open_clip load lazily

    weights = files.get(EMBEDDER_REL, cfg.models_dir / EMBEDDER_REL)
    return OpenClipEncoder(weights, batch_size=cfg.batch_size, threads=cfg.num_threads)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="search_cli.py",
                                description="Search ingested footage with a text query.")
    p.add_argument("query", help='text query, e.g. "red square"')
    p.add_argument("--camera", help="restrict to one camera ID")
    p.add_argument("--start", help="ISO 8601 start time (no offset = local time)")
    p.add_argument("--end", help="ISO 8601 end time (no offset = local time)")
    p.add_argument("--cls", help="restrict to crops of this detection class")
    p.add_argument("--limit", type=int, default=None,
                   help="maximum number of Events (default: Config.default_limit)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="NAME=VALUE",
                   help="override a Config value (repeatable)")
    return p


def parse_time(value: str | None, name: str) -> datetime | None:
    """ISO 8601 -> aware datetime; a naive value is local time (1.12)."""
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(value.strip())
    except ValueError:
        raise InputError(f"--{name} {value!r} is not an ISO 8601 date-time") from None
    return as_aware(dt)


def build_filter(args: argparse.Namespace) -> SearchFilter:
    return SearchFilter(
        camera_id=args.camera or None,
        start=parse_time(args.start, "start"),
        end=parse_time(args.end, "end"),
        cls=args.cls or None,
    )


def format_rows(events: Sequence[Event], db: MetadataStore) -> list[tuple[str, ...]]:
    """One row per Event; absolute times are the video start plus the Event offsets."""
    rows_by_id = db.hit_rows([e.hits[0].vector_id for e in events if e.hits])
    out: list[tuple[str, ...]] = []
    for e in events:
        row = rows_by_id.get(e.hits[0].vector_id) if e.hits else None
        if row is not None:
            label = row.camera_label
            start = iso_ms(row.start_ts + timedelta(seconds=e.start_s))
            end = iso_ms(row.start_ts + timedelta(seconds=e.end_s))
        else:  # row vanished between search and display
            label, start, end = "?", "?", "?"
        out.append((
            e.camera_id, label, start, end, f"{e.start_s:.1f}-{e.end_s:.1f}s",
            f"{e.score:.4f}", f"{e.relative_score:+.4f}", e.thumb_path,
        ))
    return out


def render_table(rows: Sequence[tuple[str, ...]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(HEADERS)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(HEADERS, widths)).rstrip(),
             "  ".join("-" * w for w in widths)]
    lines += ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]
    return "\n".join(lines)


def _err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


def main(argv: Sequence[str] | None = None, *,
         encoder_factory: EncoderFactory | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args, encoder_factory or default_encoder_factory)
    except KeyboardInterrupt:
        _err("interrupted")
        return EXIT_INTERRUPTED


def _run(args: argparse.Namespace, encoder_factory: EncoderFactory) -> int:
    enable_offline_mode()  # idempotent

    try:
        cfg = load_config(parse_set_args(args.overrides))
        cfg.require_valid()
    except ConfigError as exc:
        _err(str(exc))
        return EXIT_CONFIG

    # Cheap input checks before any model loads.
    try:
        flt = build_filter(args)
        limit = cfg.default_limit if args.limit is None else args.limit
        if not 1 <= limit <= cfg.max_limit:
            raise InputError(f"--limit must be between 1 and {cfg.max_limit}, got {limit}")
        if not args.query.strip():
            raise InputError("the query is empty")
        if len(args.query) > cfg.max_query_len:
            raise InputError(f"the query is longer than {cfg.max_query_len} characters")
    except InputError as exc:
        _err(str(exc))
        return EXIT_FAILURE

    setup_logging(cfg.logs_dir)
    try:
        files = require_models(cfg.models_dir, logger=log)
    except SystemExit as exc:  # require_models prints the failing files + fetch command
        return int(exc.code) if isinstance(exc.code, int) else EXIT_MODELS

    try:
        encoder = encoder_factory(cfg, files)
    except ModelMissingError as exc:
        _err(str(exc))
        return EXIT_MODEL_LOAD
    except Exception as exc:  # noqa: BLE001
        _err(f"model load failed: {type(exc).__name__}: {exc}")
        return EXIT_MODEL_LOAD

    if not cfg.db_path.exists() or not cfg.index_path.exists():
        _err(f"no ingested data ({cfg.db_path}, {cfg.index_path}); run scripts\\ingest.py first")
        return EXIT_INDEX

    db = MetadataStore(cfg.db_path)
    try:
        try:
            index = VectorIndex.load(cfg.index_path, int(getattr(encoder, "dim", DEFAULT_DIM)),
                                     force_postfilter=cfg.force_postfilter)
        except IndexUnavailable as exc:
            _err(f"{exc}; run scripts\\ingest.py --repair")
            return EXIT_INDEX

        engine = SearchEngine(cfg, db, index, encoder)
        try:
            engine.check_ready()
            events = engine.search(args.query, flt, limit)
        except FilterError as exc:
            _err(f"invalid {exc.part} filter: {exc.message}")
            return EXIT_FAILURE
        except EmptyQueryError:
            _err("the query is empty")
            return EXIT_FAILURE
        except SearchUnavailable as exc:
            _err(str(exc))
            return EXIT_INDEX

        if not events:
            print("no results")
            return EXIT_OK
        print(render_table(format_rows(events, db)))
        return EXIT_OK
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
