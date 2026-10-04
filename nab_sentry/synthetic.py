"""Synthetic_Generator: test video with exact ground truth (Requirement 15).

:func:`render_frame` is a pure function of ``(spec, i)``: a static seed-derived background
(NumPy ``default_rng(seed)`` noise, Gaussian-blurred, grey levels 150-210) shared by every
frame, plus whichever object is active at Frame_Offset ``i / fps``, drawn 100 px wide and
moving 120 px/s horizontally with bounce.

:func:`write_synthetic` encodes the frames losslessly with the bundled ffmpeg
(``libx264 -qp 0 -preset ultrafast -pix_fmt yuv420p -threads 1``) into a temp dir inside the
output dir, writes the MP4 (Filename_Convention name), its Sidecar_File and
``ground_truth.json`` there, then renames all three into the output dir. Any failure removes
the temp dir (and anything already renamed) and raises :class:`SyntheticError`.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from nab_sentry.ingest.metadata import (
    CAMERA_ID_RE,
    LABEL_MAX_LEN,
    Sidecar,
    format_filename,
    serialize_sidecar,
)

OBJECT_SIZE_PX = 100  # square side / circle diameter
SPEED_PX_S = 120.0
BG_LOW, BG_HIGH = 150, 210
BG_BLUR_SIGMA = 3.0
GROUND_TRUTH_NAME = "ground_truth.json"
FFMPEG_TIMEOUT_S = 600.0
STDERR_TAIL_CHARS = 2000

# BGR colours (frames are BGR like OpenCV-decoded frames).
COLOURS_BGR: dict[str, tuple[int, int, int]] = {
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
}


class SyntheticError(RuntimeError):
    """Invalid settings or a failure writing the synthetic output."""


@dataclass(frozen=True)
class SyntheticObject:
    label: str
    shape: Literal["square", "circle"]
    colour: str  # key of COLOURS_BGR
    start_s: float  # inclusive Frame_Offset
    end_s: float  # exclusive Frame_Offset


def default_objects() -> tuple[SyntheticObject, ...]:
    return (
        SyntheticObject("red square", "square", "red", 10.0, 25.0),
        SyntheticObject("blue circle", "circle", "blue", 50.0, 65.0),
    )


@dataclass(frozen=True)
class SyntheticSpec:
    seed: int = 0
    duration_s: float = 90.0
    fps: float = 10
    width: int = 640
    height: int = 480
    camera_id: str = "CAM-SYN01"
    label: str = "Synthetic Yard"
    start_time: datetime = datetime(2025, 1, 1, 8, 0, 0)
    objects: tuple[SyntheticObject, ...] = field(default_factory=default_objects)

    @property
    def frame_count(self) -> int:
        return int(round(self.duration_s * self.fps))

    @property
    def video_name(self) -> str:
        return format_filename(self.camera_id, self.start_time, "mp4")

    @property
    def sidecar_name(self) -> str:
        return Path(self.video_name).with_suffix(".json").name


# --------------------------------------------------------------------------- validation


def _positive_finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def validate_spec(spec: SyntheticSpec) -> None:
    """Raise :class:`SyntheticError` naming the first invalid setting (Requirement 15.8)."""
    if not _positive_finite(spec.duration_s):
        raise SyntheticError(f"invalid setting duration: {spec.duration_s!r} (must be > 0)")
    if not _positive_finite(spec.fps):
        raise SyntheticError(f"invalid setting fps: {spec.fps!r} (must be > 0)")
    if spec.frame_count < 1:
        raise SyntheticError(
            f"invalid setting duration: {spec.duration_s!r} at fps {spec.fps!r} gives no frames"
        )
    for name in ("width", "height"):
        v = getattr(spec, name)
        # yuv420p needs even dimensions; objects must fit inside the frame.
        if not isinstance(v, int) or isinstance(v, bool) or v < OBJECT_SIZE_PX or v % 2:
            raise SyntheticError(
                f"invalid setting {name}: {v!r} (must be an even integer >= {OBJECT_SIZE_PX})"
            )
    if not isinstance(spec.seed, int) or isinstance(spec.seed, bool) or spec.seed < 0:
        raise SyntheticError(f"invalid setting seed: {spec.seed!r} (must be an integer >= 0)")
    if not isinstance(spec.camera_id, str) or CAMERA_ID_RE.fullmatch(spec.camera_id) is None:
        raise SyntheticError(f"invalid setting camera_id: {spec.camera_id!r}")
    if (
        not isinstance(spec.label, str)
        or not 1 <= len(spec.label) <= LABEL_MAX_LEN
        or not spec.label.strip()
    ):
        raise SyntheticError(f"invalid setting label: {spec.label!r}")
    if not isinstance(spec.start_time, datetime):
        raise SyntheticError(f"invalid setting start_time: {spec.start_time!r}")

    for obj in spec.objects:
        key = obj.label
        start, end = obj.start_s, obj.end_s
        ok_num = all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
            for x in (start, end)
        )
        if not ok_num:
            raise SyntheticError(f"invalid setting {key} interval: [{start!r}, {end!r})")
        if start < 0:
            raise SyntheticError(f"invalid setting {key} interval: start {start!r} is before 0 s")
        if end > spec.duration_s:
            raise SyntheticError(
                f"invalid setting {key} interval: end {end!r} is after the duration "
                f"{spec.duration_s!r} s"
            )
        if end <= start:
            raise SyntheticError(
                f"invalid setting {key} interval: end {end!r} is at or before start {start!r}"
            )
        if obj.shape not in ("square", "circle"):
            raise SyntheticError(f"invalid setting {key} shape: {obj.shape!r}")
        if obj.colour not in COLOURS_BGR:
            raise SyntheticError(f"invalid setting {key} colour: {obj.colour!r}")


# --------------------------------------------------------------------------- rendering


@lru_cache(maxsize=8)
def _background(seed: int, width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.random((height, width), dtype=np.float64).astype(np.float32)
    blurred = cv2.GaussianBlur(noise, (0, 0), BG_BLUR_SIGMA)
    lo, hi = float(blurred.min()), float(blurred.max())
    scaled = (blurred - lo) / (hi - lo) if hi > lo else np.zeros_like(blurred)
    grey = np.round(BG_LOW + scaled * (BG_HIGH - BG_LOW)).astype(np.uint8)
    bg = np.repeat(grey[:, :, None], 3, axis=2)
    bg.setflags(write=False)
    return bg


def background(spec: SyntheticSpec) -> np.ndarray:
    """The static background (writable copy)."""
    return _background(spec.seed, spec.width, spec.height).copy()


def active_objects(i: int, spec: SyntheticSpec) -> list[SyntheticObject]:
    t = i / spec.fps
    return [o for o in spec.objects if o.start_s <= t < o.end_s]


def object_position(i: int, obj: SyntheticObject, spec: SyntheticSpec) -> tuple[int, int]:
    """Top-left ``(x, y)`` of the object's bounding box at frame ``i`` (triangle-wave bounce)."""
    travel = SPEED_PX_S * (i / spec.fps - obj.start_s)
    span = spec.width - OBJECT_SIZE_PX
    if span <= 0:
        x = 0.0
    else:
        p = math.fmod(max(travel, 0.0), 2 * span)
        x = p if p <= span else 2 * span - p
    y = (spec.height - OBJECT_SIZE_PX) // 2
    return int(round(x)), y


def _make_disk_mask(size: int) -> np.ndarray:
    """Boolean ``size``x``size`` disk inscribed in the box (pixel centres within radius)."""
    r = size / 2
    c = np.arange(size, dtype=np.float64) + 0.5 - r
    mask = (c[None, :] ** 2 + c[:, None] ** 2) <= r * r
    mask.setflags(write=False)
    return mask


_DISK_MASK = _make_disk_mask(OBJECT_SIZE_PX)


def render_frame(i: int, spec: SyntheticSpec) -> np.ndarray:
    """BGR uint8 frame ``i`` (HxWx3). Pure: depends only on ``spec`` and ``i``.

    Every object stays strictly inside its ``OBJECT_SIZE_PX`` bounding box at
    :func:`object_position`; the circle is the disk inscribed in that box.
    """
    frame = background(spec)
    for obj in active_objects(i, spec):
        x, y = object_position(i, obj, spec)
        colour = COLOURS_BGR[obj.colour]
        box = frame[y : y + OBJECT_SIZE_PX, x : x + OBJECT_SIZE_PX]
        if obj.shape == "square":
            box[...] = colour
        else:
            box[_DISK_MASK] = colour
    return frame


# --------------------------------------------------------------------------- documents


def sidecar_for(spec: SyntheticSpec) -> Sidecar:
    return Sidecar(spec.camera_id, spec.label, spec.start_time)


def ground_truth_document(spec: SyntheticSpec) -> dict[str, Any]:
    return {
        "seed": spec.seed,
        "videos": [
            {
                "file": spec.video_name,
                "camera_id": spec.camera_id,
                "start_time": spec.start_time.isoformat(),
                "fps": spec.fps,
                "frames": spec.frame_count,
                "width": spec.width,
                "height": spec.height,
                "objects": [
                    {"label": o.label, "start_s": float(o.start_s), "end_s": float(o.end_s)}
                    for o in spec.objects
                ],
            }
        ],
    }


def ground_truth_json(spec: SyntheticSpec) -> str:
    return json.dumps(ground_truth_document(spec), indent=2) + "\n"


# --------------------------------------------------------------------------- writer


def _ffmpeg_argv(ffmpeg: str, spec: SyntheticSpec, out_path: Path) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner", "-nostats", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{spec.width}x{spec.height}", "-r", repr(spec.fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", "-threads", "1",
        "-fflags", "+bitexact", "-flags:v", "+bitexact", "-map_metadata", "-1",
        str(out_path),
    ]


def _encode(spec: SyntheticSpec, out_path: Path, ffmpeg: str) -> None:
    # stderr goes to a file so a chatty ffmpeg can never block on a full pipe.
    err_path = out_path.with_name(out_path.name + ".stderr.txt")
    with open(err_path, "wb") as err:
        try:
            proc = subprocess.Popen(
                _ffmpeg_argv(ffmpeg, spec, out_path),
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err,
            )
        except OSError as e:
            raise SyntheticError(f"cannot run ffmpeg {ffmpeg!r}: {e}") from e
        try:
            assert proc.stdin is not None
            try:
                for i in range(spec.frame_count):
                    proc.stdin.write(render_frame(i, spec).tobytes())
            except (BrokenPipeError, OSError):
                pass  # reported via the exit code below
            finally:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            rc = proc.wait(timeout=FFMPEG_TIMEOUT_S)
        except BaseException:
            proc.kill()
            proc.wait()
            raise
    if rc != 0:
        tail = err_path.read_text(encoding="utf-8", errors="replace")[-STDERR_TAIL_CHARS:]
        raise SyntheticError(f"ffmpeg failed with exit code {rc}: {tail.strip()}")
    err_path.unlink(missing_ok=True)


def write_synthetic(
    spec: SyntheticSpec, out_dir: Path, ffmpeg: str | None = None
) -> tuple[Path, Path, Path]:
    """Validate, render, and write ``(video, sidecar, ground_truth.json)`` into ``out_dir``.

    Raises :class:`SyntheticError`; on any failure no new output file is left behind.
    """
    validate_spec(spec)
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".synthetic-", dir=out_dir))
    except OSError as e:
        raise SyntheticError(f"output directory is not writable: {out_dir}: {e}") from e

    names = (spec.video_name, spec.sidecar_name, GROUND_TRUTH_NAME)
    moved: list[Path] = []
    try:
        if ffmpeg is None:
            from nab_sentry.ingest.transcode import ffmpeg_exe

            try:
                ffmpeg = ffmpeg_exe()
            except Exception as e:  # imageio-ffmpeg raises RuntimeError when missing
                raise SyntheticError(f"bundled ffmpeg not found: {e}") from e
        try:
            _encode(spec, tmp / names[0], ffmpeg)
            (tmp / names[1]).write_text(serialize_sidecar(sidecar_for(spec)), encoding="utf-8")
            (tmp / names[2]).write_text(ground_truth_json(spec), encoding="utf-8")
            for name in names:
                os.replace(tmp / name, out_dir / name)
                moved.append(out_dir / name)
        except OSError as e:
            raise SyntheticError(f"output directory is not writable: {out_dir}: {e}") from e
        except subprocess.TimeoutExpired as e:
            raise SyntheticError(f"ffmpeg timed out after {FFMPEG_TIMEOUT_S:.0f} s") from e
    except BaseException:
        for p in moved:
            try:
                p.unlink()
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_dir / names[0], out_dir / names[1], out_dir / names[2]
