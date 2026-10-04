"""Ingest CLI (Requirements 2.8, 3.8, 7.6, 17.2, 17.7; design "Scripts" table).

Usage::

    venv\\Scripts\\python.exe scripts\\ingest.py [paths...] [--repair] [--set name=value ...]

Steps: offline mode -> Config (``--set`` overrides, validated) -> logging -> model manifest
check -> discover video files -> load Detector + Embedder -> open Metadata_Store and
Vector_Index (an absent index starts empty) -> ``reconcile()`` -> ``ingest_paths``. Prints::

    ingested=N failed=M already_indexed=K vectors=V

where ``vectors`` is the Vector_Index size after the run. ``paths`` default to
``data/videos/`` (``Config.videos_dir``); directories expand non-recursively to video files.

``--repair`` runs only ``reconcile()`` (no models loaded beyond the manifest check, no
ingest). It also replaces an index file that cannot be loaded with an empty index, so
reconciliation deletes the videos whose vectors were lost and the next ingest re-ingests them.
Without ``--repair`` an index file that exists but cannot be loaded exits 5.

Detector: when ``Config.enable_detector`` is true (default), :func:`default_detector_factory`
builds ``OnnxYoloDetector`` from the manifest path of ``yolo11n.onnx`` (``det_conf``,
``det_iou``, ``det_max_per_frame``, ``det_input_size``, ``num_threads``) before any video is
touched, so each passed frame also gets ``crop`` vectors with class, confidence, and box
(Requirements 4.5, 5.2). A missing or unreadable ONNX file exits 6 with its path and the
Model_Fetcher command (Requirement 4.6). ``--set enable_detector=false`` gives frame-only ingest.

Exit codes (design "Exit codes"): 0 success; 1 no video files found or zero vectors in the
index afterwards; 2 invalid Config; 4 manifest check failed; 5 index unavailable; 6 model
load failure; 130 interrupted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nab_sentry.startup import enable_offline_mode  # noqa: E402  (stdlib-only)

# Must run before torch / open_clip / onnxruntime are imported (Requirement 13.3).
enable_offline_mode()

from nab_sentry.config import Config, ConfigError, load_config, parse_set_args  # noqa: E402
from nab_sentry.errors import ModelMissingError  # noqa: E402
from nab_sentry.ingest.pipeline import VIDEO_EXTENSIONS, IngestPipeline  # noqa: E402
from nab_sentry.logging_setup import get_logger, setup_logging  # noqa: E402
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
DETECTOR_REL = "yolo11n.onnx"

# (cfg, manifest files rel->abs) -> Encoder / Detector
EncoderFactory = Callable[[Config, dict[str, Path]], Any]
DetectorFactory = Callable[[Config, dict[str, Path]], Any]

log = get_logger("ingest.cli")


def default_encoder_factory(cfg: Config, files: dict[str, Path]) -> Any:
    from nab_sentry.embed.clip_encoder import OpenClipEncoder  # torch/open_clip load lazily

    weights = files.get(EMBEDDER_REL, cfg.models_dir / EMBEDDER_REL)
    return OpenClipEncoder(weights, batch_size=cfg.batch_size, threads=cfg.num_threads)


def default_detector_factory(cfg: Config, files: dict[str, Path]) -> Any:
    """Build ``OnnxYoloDetector`` from the manifest path (``ModelMissingError`` if absent)."""
    from nab_sentry.ingest.detector import OnnxYoloDetector  # onnxruntime loads lazily

    path = files.get(DETECTOR_REL, cfg.models_dir / DETECTOR_REL)
    return OnnxYoloDetector(
        path,
        conf=cfg.det_conf,
        iou=cfg.det_iou,
        max_det=cfg.det_max_per_frame,
        input_size=cfg.det_input_size,
        threads=cfg.num_threads,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ingest.py",
        description="Ingest video files into the NAB Sentry Metadata_Store and Vector_Index.",
    )
    p.add_argument("paths", nargs="*", type=Path,
                   help="video files or directories (default: data/videos/)")
    p.add_argument("--repair", action="store_true",
                   help="only reconcile the Metadata_Store with the Vector_Index")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="NAME=VALUE",
                   help="override a Config value (repeatable)")
    return p


def discover_videos(paths: Sequence[Path]) -> tuple[list[Path], list[Path]]:
    """Return (video files, missing paths). Directories expand non-recursively by extension;
    explicitly named files are kept whatever their extension."""
    found: set[Path] = set()
    missing: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            found.update(c for c in p.iterdir()
                         if c.is_file() and c.suffix.lower() in VIDEO_EXTENSIONS)
        elif p.is_file():
            found.add(p)
        else:
            missing.append(p)
    return sorted(found, key=str), missing


def open_index(cfg: Config, dim: int, *, repair: bool) -> VectorIndex:
    """Load the index; an absent file gives an empty index. Unloadable: raise unless repairing."""
    if not cfg.index_path.exists():
        log.info("vector index %s absent; starting with an empty index", cfg.index_path)
        return VectorIndex(dim, force_postfilter=cfg.force_postfilter)
    try:
        return VectorIndex.load(cfg.index_path, dim, force_postfilter=cfg.force_postfilter)
    except IndexUnavailable:
        if not repair:
            raise
        log.warning("vector index %s cannot be loaded; --repair replaces it with an empty index",
                    cfg.index_path)
        return VectorIndex(dim, force_postfilter=cfg.force_postfilter)


def _err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


def main(
    argv: Sequence[str] | None = None,
    *,
    encoder_factory: EncoderFactory | None = None,
    detector_factory: DetectorFactory | None = None,
    open_source: Callable[[Path], Any] | None = None,
    start_playback: Callable[[Path, Path], Any] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args, encoder_factory or default_encoder_factory,
                    detector_factory or default_detector_factory, open_source, start_playback)
    except KeyboardInterrupt:
        _err("interrupted")
        return EXIT_INTERRUPTED


def _run(
    args: argparse.Namespace,
    encoder_factory: EncoderFactory,
    detector_factory: DetectorFactory,
    open_source: Callable[[Path], Any] | None,
    start_playback: Callable[[Path, Path], Any] | None,
) -> int:
    enable_offline_mode()  # idempotent; keeps the env forced even if main() is imported

    try:
        cfg = load_config(parse_set_args(args.overrides))
        cfg.require_valid()
    except ConfigError as exc:
        _err(str(exc))
        return EXIT_CONFIG

    setup_logging(cfg.logs_dir)
    try:
        files = require_models(cfg.models_dir, logger=log)
    except SystemExit as exc:  # require_models prints the failing files + fetch command
        return int(exc.code) if isinstance(exc.code, int) else EXIT_MODELS

    if args.repair:
        return _repair(cfg)

    paths = list(args.paths) or [cfg.videos_dir]
    videos, missing = discover_videos(paths)
    for p in missing:
        log.error("path not found: %s", p)
    if not videos:
        _err("no video files found in " + ", ".join(str(p) for p in paths))
        return EXIT_FAILURE

    # Models load before the store is opened, so a load failure creates no data files.
    try:
        detector = detector_factory(cfg, files) if cfg.enable_detector else None
        encoder = encoder_factory(cfg, files)
    except ConfigError as exc:  # detector parameters out of range (already validated above)
        _err(str(exc))
        return EXIT_CONFIG
    except ModelMissingError as exc:
        _err(str(exc))
        return EXIT_MODEL_LOAD
    except Exception as exc:  # noqa: BLE001  any other load error names the failure
        _err(f"model load failed: {type(exc).__name__}: {exc}")
        return EXIT_MODEL_LOAD

    db = MetadataStore(cfg.db_path)
    try:
        try:
            index = open_index(cfg, int(getattr(encoder, "dim", DEFAULT_DIM)), repair=False)
        except IndexUnavailable as exc:
            _err(f"{exc}; run scripts\\ingest.py --repair")
            return EXIT_INDEX

        kw: dict[str, Any] = {}
        if open_source is not None:
            kw["open_source"] = open_source
        if start_playback is not None:
            kw["start_playback"] = start_playback
        pipeline = IngestPipeline(cfg, db, index, encoder, detector, **kw)
        pipeline.reconcile()
        report = pipeline.ingest_paths(videos)
        total = index.ntotal
        print(f"ingested={report.ingested} failed={report.failed} "
              f"already_indexed={report.already_indexed} vectors={total}")
        if total == 0:
            _err("ingest finished with zero vectors in the Vector_Index; check the videos in "
                 + ", ".join(str(p) for p in paths))
            return EXIT_FAILURE
        return EXIT_OK
    finally:
        db.close()


def _repair(cfg: Config) -> int:
    db = MetadataStore(cfg.db_path)
    try:
        index = open_index(cfg, DEFAULT_DIM, repair=True)
        # reconcile() never touches the encoder or detector.
        pipeline = IngestPipeline(cfg, db, index, encoder=None, detector=None)  # type: ignore[arg-type]
        rep = pipeline.reconcile()
        index.save(cfg.index_path)  # also replaces an unloadable file with the reconciled index
        print(f"repaired removed_index_ids={len(rep.removed_index_ids)} "
              f"deleted_videos={len(rep.deleted_video_ids)} "
              f"deleted_files={len(rep.deleted_files)} vectors={index.ntotal}")
        return EXIT_OK
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
