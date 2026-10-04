"""Benchmark_Tool: per-stage CPU throughput, ingest projection and search latency.

Usage::

    venv\\Scripts\\python.exe scripts\\benchmark.py VIDEO [--iters 50] [--set name=value ...]
    venv\\Scripts\\python.exe scripts\\benchmark.py --search 200000 [--search-iters 100]
    venv\\Scripts\\python.exe scripts\\benchmark.py VIDEO --search 200000

Video stages (each: 5 untimed warm-up iterations, then ``--iters`` >= 50 timed
iterations; median and p95 reported, Requirement 14.1), all using the real components:

* ``decode``       - ms per frame from ``FileSource``
* ``motion_gate``  - ``MotionGate.evaluate`` frames per second on sampled frames
* ``thumbnail``    - ``write_thumbnail`` ms per JPEG (written under ``data/bench/``, removed after)
* ``embedder_b1``  - ``OpenClipEncoder.encode_images`` ms per image at batch size 1
* ``embedder_b8``  - ``OpenClipEncoder.encode_images`` ms per image at batch size 8
* ``detector``     - ``OnnxYoloDetector.detect`` ms per frame

An untimed ``scan`` pass runs ``Sampler`` + ``MotionGate`` over the whole input video to
measure the pass fraction ``p``; the detector's mean kept detections per passed frame
``c`` is measured on a sample of the passed frames. The projected ingest minutes per
footage hour then use the design formula (Requirement 14.2)::

    minutes = (3600*F*t_decode + 3600*r*t_gate + 3600*r*p*(t_det + t_thumb + (1+c)*t_embed@8)) / 60

``--search N`` builds an in-memory store (``:memory:``) with N random unit vectors and
matching camera/video/frame/vector rows, then times >= 100 sequential unfiltered and
filtered ``SearchEngine.search_vector`` calls with random unit query vectors, after one
excluded first request (Requirement 14.3). Text encoding is not part of this timing.

Peak resident memory of this process is sampled with ``psutil`` in a background thread
and printed at the end (Requirement 14.5).

Failures (Requirement 14.8): a stage that cannot load its model or read the video prints
the stage and file to stderr and is skipped; after the completed stages are reported the
script exits with code 1. When VIDEO is given the model manifest is checked first and a
missing or corrupt manifest exits 4 (``require_models``). ``--search`` without VIDEO
uses no model files, so the manifest check is skipped in that mode.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nab_sentry.startup import enable_offline_mode  # noqa: E402  (stdlib-only)

# Must run before torch / open_clip / onnxruntime are imported (Requirement 13.3).
enable_offline_mode()

from nab_sentry.config import Config, ConfigError, load_config, parse_set_args  # noqa: E402
from nab_sentry.model_files import DETECTOR_REL, EMBEDDER_REL, detector_path, embedder_path  # noqa: E402,F401
from nab_sentry.startup import require_models  # noqa: E402

WARMUP_ITERS = 5
MIN_ITERS = 50
MIN_SEARCH_ITERS = 100
SEARCH_TARGET_P95_MS = 1500.0
KEEP_FRAMES = 32  # sampled / passed frames kept from the scan for the per-stage timings
EXIT_STAGE_FAILED = 1


# ---------------------------------------------------------------- pure helpers


def percentile(samples: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (same as numpy's default), ``q`` in [0, 100]."""
    if not samples:
        raise ValueError("percentile of empty sample")
    if not 0.0 <= q <= 100.0 or math.isnan(q):
        raise ValueError(f"q must be in [0, 100], got {q!r}")
    xs = sorted(samples)
    pos = (len(xs) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


@dataclass(frozen=True)
class Stats:
    n: int
    median: float
    p95: float


def summarize(samples: Sequence[float]) -> Stats:
    """Median and p95 of timed samples."""
    return Stats(n=len(samples), median=percentile(samples, 50.0), p95=percentile(samples, 95.0))


def time_iterations(
    fn: Callable[[], float | None],
    iters: int,
    warmup: int = WARMUP_ITERS,
    clock: Callable[[], float] = time.perf_counter,
) -> list[float]:
    """Run ``fn`` ``warmup`` times untimed, then ``iters`` timed times.

    Returns one sample per timed iteration in milliseconds. If ``fn`` returns a
    positive number ``k`` the sample is divided by it (per-item time, e.g. per
    image in a batch).
    """
    for _ in range(warmup):
        fn()
    out: list[float] = []
    for _ in range(iters):
        t0 = clock()
        per = fn()
        ms = (clock() - t0) * 1000.0
        out.append(ms / per if per else ms)
    return out


def project_ingest_minutes(
    *,
    source_fps: float,
    sample_rate: float,
    pass_fraction: float,
    dets_per_passed: float,
    t_decode_ms: float,
    t_gate_ms: float,
    t_det_ms: float,
    t_thumb_ms: float,
    t_embed8_ms: float,
) -> float:
    """Projected ingest minutes per hour of footage (design formula, Requirement 14.2).

    ``minutes = (3600·F·t_decode + 3600·r·t_gate + 3600·r·p·(t_det + t_thumb + (1 + c)·t_embed@8)) / 60``
    with ``t_*`` the measured medians (given here in milliseconds).
    """
    for name, v in (("source_fps", source_fps), ("sample_rate", sample_rate)):
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be positive, got {v!r}")
    if not 0.0 <= pass_fraction <= 1.0:
        raise ValueError(f"pass_fraction must be in [0, 1], got {pass_fraction!r}")
    if dets_per_passed < 0:
        raise ValueError(f"dets_per_passed must be >= 0, got {dets_per_passed!r}")
    sec = 1.0 / 1000.0
    per_passed = (t_det_ms + t_thumb_ms + (1.0 + dets_per_passed) * t_embed8_ms) * sec
    total_s = (
        3600.0 * source_fps * t_decode_ms * sec
        + 3600.0 * sample_rate * t_gate_ms * sec
        + 3600.0 * sample_rate * pass_fraction * per_passed
    )
    return total_s / 60.0


@dataclass(frozen=True)
class StageResult:
    stage: str
    unit: str
    stats: Stats | None = None
    file: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.stats is not None


def format_report(results: Sequence[StageResult]) -> str:
    lines = ["", f"{'stage':<16} {'unit':<10} {'n':>4} {'median':>10} {'p95':>10}"]
    for r in results:
        if r.ok:
            s = r.stats
            lines.append(f"{r.stage:<16} {r.unit:<10} {s.n:>4} {s.median:>10.2f} {s.p95:>10.2f}")
        else:
            lines.append(f"{r.stage:<16} {r.unit:<10} {'-':>4} {'FAILED':>10} {'-':>10}")
    return "\n".join(lines)


class StageFailed(Exception):
    def __init__(self, file: Path | str, message: str) -> None:
        super().__init__(message)
        self.file = str(file)


def _failed_file(exc: BaseException, default: Path | str) -> str:
    """The file named by a stage failure: ``StageFailed.file``, ``ModelMissingError.path``, or default."""
    for attr in ("file", "path"):
        val = getattr(exc, attr, None)
        if val:
            return str(val)
    return str(default)


# ---------------------------------------------------------------- peak RSS


class RssSampler:
    """Background thread recording the peak RSS of this process (Requirement 14.5)."""

    def __init__(self, interval_s: float = 0.05) -> None:
        import psutil

        self._proc = psutil.Process()
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rss-sampler", daemon=True)
        self.peak = 0

    def _sample(self) -> None:
        try:
            self.peak = max(self.peak, int(self._proc.memory_info().rss))
        except Exception:  # noqa: BLE001 - sampling must never break the benchmark
            pass

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def __enter__(self) -> RssSampler:
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._sample()


# ---------------------------------------------------------------- video stages


def _open_source(video: Path):
    from nab_sentry.ingest.sources import FileSource

    if not video.is_file():
        raise StageFailed(video, "video file not found")
    try:
        return FileSource.open(video)
    except Exception as exc:  # UnreadableVideo / UnknownFrameRate
        raise StageFailed(video, str(exc)) from exc


def bench_decode(video: Path, iters: int) -> Stats:
    """ms per decoded frame from ``FileSource``; short clips are reopened and looped."""
    state = {"src": _open_source(video), "it": None, "frames": 0}
    state["it"] = state["src"].frames()

    def read_one() -> None:
        try:
            next(state["it"])
        except StopIteration:
            state["src"].close()
            if state["frames"] == 0:
                raise StageFailed(video, "no decodable frames") from None
            state["src"] = _open_source(video)
            state["it"] = state["src"].frames()
            try:
                next(state["it"])
            except StopIteration:
                raise StageFailed(video, "cannot re-read video") from None
        state["frames"] += 1

    try:
        return summarize(time_iterations(read_one, iters))
    finally:
        state["src"].close()


@dataclass
class ScanResult:
    fps: float
    sampled: int
    passed: int
    sampled_frames: list  # first KEEP_FRAMES sampled DecodedFrames, in order
    passed_images: list  # reservoir of up to KEEP_FRAMES passed images

    @property
    def pass_fraction(self) -> float:
        return self.passed / self.sampled if self.sampled else 0.0


def _new_gate(cfg: Config):
    from nab_sentry.ingest.motion import MotionGate

    return MotionGate(cfg.motion_threshold, cfg.keyframe_interval_s, cfg.gate_width,
                      cfg.motion_pixel_delta, cfg.motion_method)  # type: ignore[arg-type]


def scan_video(video: Path, cfg: Config, keep: int = KEEP_FRAMES, seed: int = 0) -> ScanResult:
    """Untimed Sampler + MotionGate pass over the whole video: pass fraction and frame samples."""
    import random

    from nab_sentry.ingest.sampler import Sampler

    rng = random.Random(seed)
    gate = _new_gate(cfg)
    sampled_frames: list = []
    passed_images: list = []
    sampled = passed = 0
    with _open_source(video) as src:
        fps = src.fps
        for frame in Sampler(cfg.sample_rate).select(src.frames()):
            decision = gate.evaluate(frame)
            if decision.reason == "empty":
                continue
            sampled += 1
            if len(sampled_frames) < keep:
                sampled_frames.append(frame)
            if decision.passed:
                passed += 1
                if len(passed_images) < keep:  # reservoir sample over all passed frames
                    passed_images.append(frame.image)
                else:
                    j = rng.randrange(passed)
                    if j < keep:
                        passed_images[j] = frame.image
    if sampled == 0:
        raise StageFailed(video, "no decodable sampled frames")
    return ScanResult(fps, sampled, passed, sampled_frames, passed_images)


def bench_gate(cfg: Config, frames: list, iters: int) -> Stats:
    """MotionGate frames per second; each sample is one pass over ``frames``."""
    gate = _new_gate(cfg)

    def run() -> int:
        for f in frames:
            gate.evaluate(f)
        return len(frames)

    ms_per_frame = time_iterations(run, iters)
    return summarize([1000.0 / ms if ms > 0 else float("inf") for ms in ms_per_frame])


def bench_thumbnail(images: list, out_dir: Path, width: int, iters: int) -> Stats:
    from nab_sentry.ingest.transcode import write_thumbnail

    i = {"n": 0}

    def run() -> None:
        n = i["n"]
        i["n"] += 1
        path = out_dir / f"t{n % len(images):03d}.jpg"
        try:
            write_thumbnail(images[n % len(images)], path, width)
        except Exception as exc:  # ThumbnailError
            raise StageFailed(path, str(exc)) from exc

    return summarize(time_iterations(run, iters))


def bench_embedder(encoder, images: list, batch: int, iters: int) -> Stats:
    chosen = [images[i % len(images)] for i in range(batch)]

    def run() -> int:
        encoder.encode_images(chosen)
        return batch

    return summarize(time_iterations(run, iters))


def bench_detector(detector, images: list, iters: int) -> tuple[Stats, float]:
    """ms per frame, plus mean kept detections per passed frame over ``images``."""
    counts = [len(detector.detect(img)) for img in images]
    i = {"n": 0}

    def run() -> None:
        img = images[i["n"] % len(images)]
        i["n"] += 1
        detector.detect(img)

    return summarize(time_iterations(run, iters)), sum(counts) / len(counts)


def _synthetic_frames(n: int = 8) -> list:
    """1080p noise frames so later stages still run when the video stage fails."""
    import numpy as np

    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8) for _ in range(n)]


def _physical_threads(cfg: Config) -> int:
    if isinstance(cfg.num_threads, int) and cfg.num_threads > 0:
        return cfg.num_threads
    import psutil

    return psutil.cpu_count(logical=False) or 1


def _bench_dir(cfg: Config) -> Path:
    """Fresh scratch dir under ``data/bench/`` (never the OS temp dir, Requirement 13.7)."""
    d = Path(cfg.data_dir) / "bench" / f"run-{os.getpid()}-{time.time_ns()}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def _remove_bench_dir(d: Path) -> None:
    shutil.rmtree(d, ignore_errors=True)
    try:
        d.parent.rmdir()  # data/bench, only if now empty
    except OSError:
        pass


def format_projection(*, minutes: float, source_fps: float, sample_rate: float, sampled: int,
                      passed: int, dets: float, det_frames: int, detector_on: bool) -> str:
    p = passed / sampled if sampled else 0.0
    det_note = (f"mean detections per passed frame={dets:.2f} (over {det_frames} passed frames)"
                if detector_on else "detector disabled (t_det=0, c=0)")
    return (
        f"\nprojected ingest: {minutes:.1f} min per footage hour "
        f"(source fps={source_fps:g}, Sample_Rate={sample_rate:g}, "
        f"pass fraction={p:.3f} ({passed}/{sampled} sampled frames), {det_note}; "
        f"transcode runs concurrently and is not included)"
    )


def run_video_stages(video: Path, cfg: Config, iters: int,
                     results: list[StageResult], fail: Callable[..., None]) -> list[str]:
    """Run all video/model stages; append to ``results``; return extra report lines."""
    files = require_models(cfg.models_dir)  # exits 4 on manifest failure
    threads = _physical_threads(cfg)
    medians: dict[str, float] = {}
    print(f"warm-up={WARMUP_ITERS} timed={iters} threads={threads} video={video}")

    def ok(stage: str, unit: str, stats: Stats) -> None:
        results.append(StageResult(stage, unit, stats))
        medians[stage] = stats.median

    # decode
    try:
        ok("decode", "ms/frame", bench_decode(video, iters))
    except Exception as exc:  # noqa: BLE001
        fail("decode", "ms/frame", exc, _failed_file(exc, video))

    # untimed scan: pass fraction + frame samples
    scan: ScanResult | None = None
    if "decode" in medians:
        try:
            scan = scan_video(video, cfg)
            print(f"scan: fps={scan.fps:g} sampled={scan.sampled} passed={scan.passed} "
                  f"pass fraction={scan.pass_fraction:.3f} at Sample_Rate={cfg.sample_rate:g}")
        except Exception as exc:  # noqa: BLE001
            fail("scan", "-", exc, _failed_file(exc, video))

    from nab_sentry.ingest.sources import DecodedFrame

    if scan is not None:
        gate_frames = scan.sampled_frames
        images = scan.passed_images or [f.image for f in scan.sampled_frames]
    else:
        print("note: using synthetic 1920x1080 frames for the remaining stages", file=sys.stderr)
        images = _synthetic_frames()
        gate_frames = [DecodedFrame(i, i / cfg.sample_rate, im) for i, im in enumerate(images)]

    # motion gate
    try:
        ok("motion_gate", "frames/s", bench_gate(cfg, gate_frames, iters))
    except Exception as exc:  # noqa: BLE001
        fail("motion_gate", "frames/s", exc, _failed_file(exc, video))

    # thumbnails, in a scratch dir under data/bench/
    bench_dir: Path | None = None
    try:
        bench_dir = _bench_dir(cfg)
        ok("thumbnail", "ms/image", bench_thumbnail(images, bench_dir, cfg.thumb_width, iters))
    except Exception as exc:  # noqa: BLE001
        fail("thumbnail", "ms/image", exc, _failed_file(exc, bench_dir or cfg.data_dir))
    finally:
        if bench_dir is not None:
            _remove_bench_dir(bench_dir)

    # embedder
    emb_path = embedder_path(cfg, files)
    try:
        from nab_sentry.embed.clip_encoder import OpenClipEncoder

        encoder = OpenClipEncoder(emb_path, batch_size=8, threads=threads)
    except Exception as exc:  # noqa: BLE001
        for stage in ("embedder_b1", "embedder_b8"):
            fail(stage, "ms/image", exc, _failed_file(exc, emb_path))
    else:
        for batch in (1, 8):
            stage = f"embedder_b{batch}"
            try:
                ok(stage, "ms/image", bench_embedder(encoder, images, batch, iters))
            except Exception as exc:  # noqa: BLE001
                fail(stage, "ms/image", exc, str(emb_path))
        del encoder

    # detector
    det_path = detector_path(cfg, files)
    dets = 0.0
    try:
        from nab_sentry.ingest.detector import OnnxYoloDetector

        detector = OnnxYoloDetector(det_path, conf=cfg.det_conf, iou=cfg.det_iou,
                                    max_det=cfg.det_max_per_frame,
                                    input_size=cfg.det_input_size, threads=threads)
        stats, dets = bench_detector(detector, images, iters)
        ok("detector", "ms/frame", stats)
    except Exception as exc:  # noqa: BLE001
        fail("detector", "ms/frame", exc, _failed_file(exc, det_path))

    # projection (14.2)
    needed = ["decode", "motion_gate", "thumbnail", "embedder_b8"]
    if cfg.enable_detector:
        needed.append("detector")
    missing = [s for s in needed if s not in medians] + ([] if scan else ["scan"])
    if missing:
        return [f"\nprojected ingest: not available (missing stages: {', '.join(missing)})"]
    assert scan is not None
    detector_on = bool(cfg.enable_detector)
    minutes = project_ingest_minutes(
        source_fps=scan.fps,
        sample_rate=float(cfg.sample_rate),
        pass_fraction=scan.pass_fraction,
        dets_per_passed=dets if detector_on else 0.0,
        t_decode_ms=medians["decode"],
        t_gate_ms=1000.0 / medians["motion_gate"],
        t_det_ms=medians["detector"] if detector_on else 0.0,
        t_thumb_ms=medians["thumbnail"],
        t_embed8_ms=medians["embedder_b8"],
    )
    return [format_projection(minutes=minutes, source_fps=scan.fps, sample_rate=float(cfg.sample_rate),
                              sampled=scan.sampled, passed=scan.passed, dets=dets,
                              det_frames=len(images), detector_on=detector_on)]


# ---------------------------------------------------------------- search benchmark

SEARCH_CAMERAS = ("cam1", "cam2", "cam3", "cam4")
SEARCH_BASE_TS = datetime(2025, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
FRAMES_PER_VIDEO = 3600  # one sampled frame per second of a one-hour video


@dataclass
class SearchFixture:
    db: Any
    index: Any
    videos: int
    filter: Any  # SearchFilter used for the filtered timings


def _random_unit(rng, n: int, dim: int):
    import numpy as np

    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def build_search_fixture(n: int, cfg: Config, seed: int = 0, dim: int = 512) -> SearchFixture:
    """In-memory store + index with ``n`` random unit vectors and matching rows.

    Each sampled frame (one per second, 3600 per video) gets a ``frame`` vector and a
    ``crop`` vector whose class cycles through the Target_Classes; videos rotate over
    four cameras, each video starting one hour after the previous one.
    """
    import numpy as np

    from nab_sentry.search.engine import VALID_CLASSES, SearchFilter
    from nab_sentry.store.db import MetadataStore, NewFrame, NewVector, NewVideo
    from nab_sentry.store.vector_index import VectorIndex

    if n < 1:
        raise ValueError("--search N must be >= 1")
    classes = sorted(VALID_CLASSES)
    db = MetadataStore(":memory:")
    ids: list[int] = []
    frames_in_video: dict[int, int] = {}
    with db.transaction():
        for cam in SEARCH_CAMERAS:
            db.upsert_camera(cam, f"Bench {cam}")
        video_id = -1
        vidx = -1
        frame_no = FRAMES_PER_VIDEO
        start = SEARCH_BASE_TS
        while len(ids) < n:
            if frame_no >= FRAMES_PER_VIDEO:
                vidx += 1
                frame_no = 0
                start = SEARCH_BASE_TS + timedelta(hours=vidx)
                video_id = db.insert_video(NewVideo(
                    camera_id=SEARCH_CAMERAS[vidx % len(SEARCH_CAMERAS)],
                    src_path=f"bench/v{vidx:05d}.mp4", src_hash=f"bench-{vidx:08d}",
                    start_ts=start, fps=30.0, width=1920, height=1080,
                    est_duration_s=float(FRAMES_PER_VIDEO)))
                frames_in_video[video_id] = 0
            frame_id = db.insert_frame(NewFrame(
                video_id=video_id, frame_idx=frame_no * 30, offset_s=float(frame_no),
                abs_time=start + timedelta(seconds=frame_no),
                gate_reason="motion", motion_frac=0.5, thumb_path=f"bench_{video_id}_{frame_no}.jpg"))
            frames_in_video[video_id] += 1
            ids.append(db.insert_vector(NewVector(frame_id=frame_id, kind="frame")))
            if len(ids) < n:
                cls = classes[frame_no % len(classes)]
                ids.append(db.insert_vector(NewVector(
                    frame_id=frame_id, kind="crop", det_class=cls, det_conf=0.6,
                    box=(10, 10, 110, 210))))
            frame_no += 1
        for vid, count in frames_in_video.items():
            db.finalize_video(vid, sampled=count, passed=count, duration_s=float(FRAMES_PER_VIDEO),
                              playback_path=f"v{vid}.mp4")

    index = VectorIndex(dim, force_postfilter=bool(cfg.force_postfilter))
    rng = np.random.default_rng(seed)
    id_arr = np.asarray(ids, dtype=np.int64)
    chunk = 50_000
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        index.add(id_arr[lo:hi], _random_unit(rng, hi - lo, dim))

    videos = vidx + 1
    half = max(1, videos // 2)
    filt = SearchFilter(camera_id=SEARCH_CAMERAS[0], start=SEARCH_BASE_TS,
                        end=SEARCH_BASE_TS + timedelta(hours=half), cls="person")
    return SearchFixture(db=db, index=index, videos=videos, filter=filt)


def bench_search(fx: SearchFixture, cfg: Config, iters: int, seed: int = 1) -> tuple[Stats, Stats]:
    """(unfiltered, filtered) ms per ``search_vector`` call; the first call of each is excluded."""
    import numpy as np

    from nab_sentry.search.engine import SearchEngine, SearchFilter

    engine = SearchEngine(cfg, fx.db, fx.index, encoder=None)
    rng = np.random.default_rng(seed)
    out = []
    for f in (SearchFilter(), fx.filter):
        queries = _random_unit(rng, iters + 1, fx.index.dim)
        i = {"n": 0}

        def run(f=f, queries=queries) -> None:
            q = queries[i["n"]]
            i["n"] += 1
            engine.search_vector(q, f, "benchmark")

        out.append(summarize(time_iterations(run, iters, warmup=1)))
    return out[0], out[1]


def run_search_stage(n: int, cfg: Config, iters: int,
                     results: list[StageResult], fail: Callable[..., None]) -> list[str]:
    t0 = time.perf_counter()
    try:
        fx = build_search_fixture(n, cfg)
    except Exception as exc:  # noqa: BLE001
        for stage in ("search", "search_filtered"):
            fail(stage, "ms/query", exc, ":memory:")
        return []
    build_s = time.perf_counter() - t0
    try:
        allowed = int(fx.db.allowed_vector_ids(fx.filter).size)
        print(f"search fixture: {fx.index.ntotal} vectors, {fx.videos} videos, top_k={cfg.top_k}, "
              f"filter=camera {fx.filter.camera_id} + {fx.filter.start.isoformat()}.."
              f"{fx.filter.end.isoformat()} + cls {fx.filter.cls} ({allowed} allowed IDs), "
              f"built in {build_s:.1f} s; timed={iters} sequential queries each, first excluded")
        plain, filtered = bench_search(fx, cfg, iters)
        results.append(StageResult("search", "ms/query", plain))
        results.append(StageResult("search_filtered", "ms/query", filtered))
    except Exception as exc:  # noqa: BLE001
        fail("search", "ms/query", exc, ":memory:")
        return []
    finally:
        fx.db.close()
    lines = []
    for name, s in (("unfiltered", plain), ("filtered", filtered)):
        verdict = "within" if s.p95 <= SEARCH_TARGET_P95_MS else "ABOVE"
        lines.append(f"search p95 {name}: {s.p95:.2f} ms over {s.n} queries "
                     f"({verdict} the {SEARCH_TARGET_P95_MS:.0f} ms target)")
    return ["", *lines]


# ---------------------------------------------------------------- main


def _min_int(minimum: int, flag: str) -> Callable[[str], int]:
    def parse(raw: str) -> int:
        try:
            v = int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"not an integer: {raw!r}") from exc
        if v < minimum:
            raise argparse.ArgumentTypeError(f"{flag} must be >= {minimum}")
        return v

    return parse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchmark.py",
        description="Per-stage CPU benchmark, ingest projection and search latency.",
    )
    p.add_argument("video", type=Path, nargs="?", help="input video file (video/model stages)")
    p.add_argument(
        "--iters", type=_min_int(MIN_ITERS, "--iters"), default=MIN_ITERS,
        help=f"timed iterations per stage (>= {MIN_ITERS}; {WARMUP_ITERS} warm-up iterations run first)",
    )
    p.add_argument("--search", type=_min_int(1, "--search"), metavar="N",
                   help="build a random N-vector index plus store rows and time searches")
    p.add_argument("--search-iters", type=_min_int(MIN_SEARCH_ITERS, "--search-iters"),
                   default=MIN_SEARCH_ITERS,
                   help=f"timed sequential searches per mode (>= {MIN_SEARCH_ITERS}; first request excluded)")
    p.add_argument("--set", dest="overrides", action="append", metavar="NAME=VALUE",
                   help="Config override (repeatable), e.g. --set num_threads=4")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.video is None and args.search is None:
        parser.error("give a VIDEO, --search N, or both")
    try:
        cfg = load_config(parse_set_args(args.overrides))
        cfg.require_valid()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    results: list[StageResult] = []
    extra: list[str] = []

    def fail(stage: str, unit: str, exc: BaseException, file: str) -> None:
        print(f"[FAILED] stage={stage} file={file}: {exc}", file=sys.stderr)
        results.append(StageResult(stage, unit, file=file, error=str(exc)))

    with RssSampler() as rss:
        if args.video is not None:
            extra += run_video_stages(args.video, cfg, args.iters, results, fail)
        if args.search is not None:
            extra += run_search_stage(args.search, cfg, args.search_iters, results, fail)

    print(format_report(results))
    for line in extra:
        print(line)
    print(f"\npeak RSS (this process): {rss.peak / (1024 * 1024):.0f} MiB")
    failed = [r for r in results if not r.ok]
    if failed:
        print(f"\n{len(failed)} stage(s) failed: " + ", ".join(f"{r.stage} ({r.file})" for r in failed),
              file=sys.stderr)
        return EXIT_STAGE_FAILED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
