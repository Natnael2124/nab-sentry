"""Evaluator: retrieval quality metrics over ``eval/queries.yaml`` (Requirement 16).

Usage::

    venv\\Scripts\\python.exe scripts\\evaluate.py [--queries eval/queries.yaml] [--set name=value ...]

Steps: offline mode -> Config (``--set`` overrides, validated) -> load the queries file
(missing / unparseable / count outside 15..20 -> exit 1, 16.8) -> logging -> model manifest
check -> load Embedder -> open Metadata_Store and Vector_Index -> ``SearchEngine.check_ready``
-> run every valid query in-process with no Search_Filter (the engine reads ``top_k`` from the
Config, 16.2). Queries naming a camera ID that is not in the Metadata_Store are error rows
(16.7). Each Event's padded window (``start_s``/``end_s`` already include Event_Padding) is
converted to absolute time as video start + offset and scored with
:func:`nab_sentry.evaluation.is_relevant`.

The report JSON (run time, every 16.5 Config parameter, aggregates, per-query rows in file
order) is written to ``<data_dir>/eval/report-YYYYmmdd-HHMMSS.json`` (16.6).

Exit codes (design "Exit codes"): 0 success; 1 queries file error or every query in error;
2 invalid Config; 4 manifest check failed; 5 vector index unavailable or inconsistent;
6 model load failure; 130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from nab_sentry.evaluation import (  # noqa: E402
    Aggregate,
    EvalFileError,
    EvalQuery,
    QueryError,
    QueryResult,
    aggregate,
    hit_at_1,
    is_relevant,
    load_queries_file,
    precision_at_5,
)
from nab_sentry.logging_setup import get_logger, setup_logging  # noqa: E402
from nab_sentry.search.clustering import Event  # noqa: E402
from nab_sentry.search.engine import SearchEngine, SearchFilter, SearchUnavailable  # noqa: E402
from nab_sentry.startup import require_models  # noqa: E402
from nab_sentry.store.db import MetadataStore  # noqa: E402
from nab_sentry.store.vector_index import DEFAULT_DIM, IndexUnavailable, VectorIndex  # noqa: E402

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG = 2
EXIT_MODELS = 4
EXIT_INDEX = 5
EXIT_MODEL_LOAD = 6
EXIT_INTERRUPTED = 130

EMBEDDER_REL = "open_clip/ViT-B-32-laion2b_s34b_b79k/open_clip_pytorch_model.bin"
DEFAULT_QUERIES = _ROOT / "eval" / "queries.yaml"

# Every tunable listed in Requirement 16.5, by Config field name.
REPORT_PARAMS = (
    "sample_rate", "motion_threshold", "keyframe_interval_s", "gate_width", "det_conf",
    "det_max_per_frame", "batch_size", "top_k", "merge_gap_s", "event_padding_s", "label_boost",
)

# (cfg, manifest files rel->abs) -> Encoder
EncoderFactory = Callable[[Config, dict[str, Path]], Any]

log = get_logger("evaluate")


def default_encoder_factory(cfg: Config, files: dict[str, Path]) -> Any:
    from nab_sentry.embed.clip_encoder import OpenClipEncoder  # torch/open_clip load lazily

    weights = files.get(EMBEDDER_REL, cfg.models_dir / EMBEDDER_REL)
    return OpenClipEncoder(weights, batch_size=cfg.batch_size, threads=cfg.num_threads)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evaluate.py",
                                description="Measure precision@5 / hit@1 over eval/queries.yaml.")
    p.add_argument("--queries", type=Path, default=DEFAULT_QUERIES,
                   help="queries file (default: eval/queries.yaml)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="NAME=VALUE",
                   help="override a Config value (repeatable)")
    return p


def _err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


# --------------------------------------------------------------------------------------------
# Scoring


def relevance_flags(events: Sequence[Event], db: MetadataStore, gt: EvalQuery) -> list[bool]:
    """Relevance per ranked Event; the window is the Event's padded offsets on the video start."""
    rows = db.hit_rows([e.hits[0].vector_id for e in events if e.hits])
    flags: list[bool] = []
    for e in events:
        row = rows.get(e.hits[0].vector_id) if e.hits else None
        if row is None:  # row vanished between search and scoring
            flags.append(False)
            continue
        # start_s / end_s already include Event_Padding (cluster_hits); do not pad again.
        start = row.start_ts + timedelta(seconds=e.start_s)
        end = row.start_ts + timedelta(seconds=e.end_s)
        flags.append(is_relevant(e.camera_id, start, end, gt))
    return flags


def run_queries(engine: SearchEngine, db: MetadataStore, cfg: Config,
                queries: Sequence[EvalQuery],
                load_errors: Sequence[QueryError]) -> list[QueryResult]:
    """One QueryResult per YAML entry, in file order."""
    known = {cam for cam, _label in db.cameras()}
    by_index: dict[int, QueryResult] = {
        e.index: QueryResult(text=e.text or "", error=e.reason) for e in load_errors
    }
    for q in queries:
        if q.camera_id not in known:
            by_index[q.index] = QueryResult(
                text=q.text, error=f"camera ID {q.camera_id!r} is not in the Metadata_Store")
            continue
        t0 = time.perf_counter()
        try:
            events = engine.search(q.text, SearchFilter(), cfg.top_k)
        except EmptyQueryError:
            by_index[q.index] = QueryResult(text=q.text, error="missing query text")
            continue
        latency_ms = (time.perf_counter() - t0) * 1000.0
        flags = relevance_flags(events, db, q)
        by_index[q.index] = QueryResult(
            text=q.text, precision_at_5=precision_at_5(flags), hit_at_1=hit_at_1(flags),
            latency_ms=latency_ms)
    return [by_index[i] for i in sorted(by_index)]


# --------------------------------------------------------------------------------------------
# Report


def config_params(cfg: Config) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in REPORT_PARAMS:
        value = getattr(cfg, name)
        out[name] = str(value) if isinstance(value, Path) else value
    return out


def _row(r: QueryResult) -> dict[str, Any]:
    if r.is_error:
        return {"text": r.text, "error": r.error}
    return {"text": r.text, "precision_at_5": r.precision_at_5, "hit_at_1": r.hit_at_1,
            "latency_ms": r.latency_ms}


def build_report(run_at: datetime, cfg: Config, results: Sequence[QueryResult],
                 agg: Aggregate | None) -> dict[str, Any]:
    return {
        "run_at": run_at.isoformat(timespec="seconds"),
        "config": config_params(cfg),
        "aggregate": None if agg is None else {
            "precision_at_5": agg.precision_at_5,
            "hit_at_1": agg.hit_at_1,
            "latency_ms_mean": agg.latency_ms_mean,
            "queries_run": agg.queries_run,
            "queries_errored": agg.queries_errored,
        },
        "queries": [_row(r) for r in results],
    }


def write_report(report: dict[str, Any], out_dir: Path, run_at: datetime) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"report-{run_at:%Y%m%d-%H%M%S}"
    path = out_dir / f"{stem}.json"
    n = 1
    while path.exists():  # two runs within the same second keep both reports
        path = out_dir / f"{stem}-{n}.json"
        n += 1
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def render_summary(results: Sequence[QueryResult], agg: Aggregate | None) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        if r.is_error:
            lines.append(f"{i:>2}. ERROR  {r.text!r}: {r.error}")
        else:
            lines.append(f"{i:>2}. p@5={r.precision_at_5:.1f} hit@1={r.hit_at_1} "
                         f"{r.latency_ms:7.1f} ms  {r.text!r}")
    if agg is not None and agg.precision_at_5 is not None:
        lines.append(f"mean p@5={agg.precision_at_5:.3f} hit@1={agg.hit_at_1:.3f} "
                     f"latency={agg.latency_ms_mean:.1f} ms  "
                     f"({agg.queries_run} run, {agg.queries_errored} errored)")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Entry point


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

    # File-level problems before any model loads (16.8).
    try:
        queries, load_errors = load_queries_file(args.queries)
    except EvalFileError as exc:
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
        run_at = datetime.now().astimezone()
        try:
            engine.check_ready()
            results = run_queries(engine, db, cfg, queries, load_errors)
        except SearchUnavailable as exc:
            _err(str(exc))
            return EXIT_INDEX
    finally:
        db.close()

    all_errored = all(r.is_error for r in results)
    agg = None if all_errored else aggregate(results)
    report_path = write_report(build_report(run_at, cfg, results, agg), cfg.data_dir / "eval",
                               run_at)
    print(render_summary(results, agg))
    print(f"report: {report_path}")
    if all_errored:
        _err(f"every query in {args.queries} is in error; no aggregate metrics")
        return EXIT_FAILURE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
