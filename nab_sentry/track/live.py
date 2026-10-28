"""Per-camera live tracking across Segments (V2 Requirement 10.4-10.6, task 10.3).

:class:`LiveTracks` implements :class:`nab_sentry.track.TrackSink` for the Ingest_Pipeline and
keeps one :class:`ByteTracker` per camera between videos:

* **Continuity.** A video continues its camera's tracks when it starts at most
  ``gap_reset_s`` after the previous video of that camera ended; a longer gap, or a start more
  than ``overlap_tolerance_s`` *before* that end (out of order), ends them first. Segment names
  have one-second resolution, so a following Segment can appear to start up to about a second
  early; frame times are therefore kept non-decreasing instead of failing.
* **Scene cuts.** Each frame is compared with the previous tracked frame of the same camera
  (also across a Segment boundary - a camera moved between Segments is a cut) on the motion
  gate's grey signature; a changed fraction at or above ``scene_cut_frac`` ends all tracks.
* **Rollback.** ``begin_video`` snapshots the camera's state; ``rollback_video`` restores it, so
  a rolled-back Segment leaves no trace in the tracks.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from nab_sentry.config import Config
from nab_sentry.ingest.detector import Detection
from nab_sentry.ingest.motion import changed_fraction, downscale_gray
from nab_sentry.ingest.sources import DecodedFrame
from nab_sentry.logging_setup import get_logger
from nab_sentry.track.bytetrack import Assignment, ByteTracker, TrackRecord

log = get_logger("track")


@dataclass
class _CameraState:
    tracker: ByteTracker
    last_t: float | None = None  # time of the last tracked frame (tracker clock)
    end_t: float | None = None  # end of the last committed video
    sig: np.ndarray | None = None  # grey signature of the last tracked frame
    stats: dict[str, int] = field(default_factory=lambda: {"scene_cuts": 0, "gap_resets": 0})


class LiveTracks:
    def __init__(self, new_tracker: Callable[[], ByteTracker], *, scene_cut_frac: float = 0.6,
                 gap_reset_s: float = 3.0, gate_width: int = 320, pixel_delta: int = 25,
                 overlap_tolerance_s: float = 2.0) -> None:
        self._new_tracker = new_tracker
        self.scene_cut_frac = float(scene_cut_frac)
        self.gap_reset_s = float(gap_reset_s)
        self.gate_width = int(gate_width)
        self.pixel_delta = int(pixel_delta)
        self.overlap_tolerance_s = float(overlap_tolerance_s)
        self._cams: dict[str, _CameraState] = {}
        self._cur: _CameraState | None = None
        self._cam_id: str | None = None
        self._start_t = 0.0
        self._video_id: int | None = None
        self._snap: tuple[str, _CameraState | None] | None = None

    @classmethod
    def from_config(cls, cfg: Config) -> LiveTracks:
        return cls(lambda: ByteTracker(cfg.track_match_iou, cfg.track_min_hits, cfg.track_buffer_s),
                   scene_cut_frac=cfg.scene_cut_frac, gap_reset_s=cfg.segment_gap_reset_s,
                   gate_width=cfg.gate_width, pixel_delta=cfg.motion_pixel_delta)

    def stats(self, camera_id: str) -> dict[str, int]:
        st = self._cams.get(camera_id)
        return dict(st.stats) if st else {"scene_cuts": 0, "gap_resets": 0}

    def tracker(self, camera_id: str) -> ByteTracker | None:
        st = self._cams.get(camera_id)
        return st.tracker if st else None

    # -- TrackSink -----------------------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self._cur is not None and self._cur.tracker.active

    def begin_video(self, camera_id: str, start_t: float, video_id: int | None = None) -> None:
        st = self._cams.get(camera_id)
        self._snap = (camera_id, copy.deepcopy(st))
        if st is None:
            st = self._cams[camera_id] = _CameraState(self._new_tracker())
        elif st.end_t is not None:
            gap = start_t - st.end_t
            if gap > self.gap_reset_s or gap < -self.overlap_tolerance_s:
                if st.tracker.active:
                    log.info("track %s: %.1f s %s between videos; ending tracks", camera_id,
                             abs(gap), "gap" if gap > 0 else "overlap")
                st.tracker.reset()
                st.sig = None
                st.last_t = None
                st.stats["gap_resets"] += 1
        st.tracker.begin_video(start_t)
        self._cur, self._cam_id, self._start_t = st, camera_id, float(start_t)
        self._video_id = video_id

    def update(self, frame: DecodedFrame, detections: Sequence[Detection]) -> list[Assignment]:
        st = self._require()
        t = self._start_t + float(frame.offset_s)
        if st.last_t is not None and t < st.last_t:
            t = st.last_t  # name-resolution overlap at a Segment boundary: keep time monotone
        if frame.image is not None and frame.image.size:
            sig = downscale_gray(frame.image, self.gate_width)
            if st.sig is not None and changed_fraction(st.sig, sig, self.pixel_delta) >= self.scene_cut_frac:
                if st.tracker.active:
                    log.info("track %s: scene cut at %.2f s; ending tracks", self._cam_id, frame.offset_s)
                st.tracker.end_all()
                st.stats["scene_cuts"] += 1
            st.sig = sig
        res = st.tracker.step(t, detections, tag=(self._video_id, float(frame.offset_s)))
        st.last_t = t
        return res

    def end_video(self, end_t: float) -> list[TrackRecord]:
        st = self._require()
        st.end_t = float(end_t)
        st.tracker.expire(max(float(end_t), st.last_t or float(end_t)))
        return st.tracker.drain()

    def rollback_video(self) -> None:
        if self._snap is None:
            return
        cam, saved = self._snap
        if saved is None:
            self._cams.pop(cam, None)
        else:
            self._cams[cam] = saved
        self._snap = None
        self._cur = None

    def _require(self) -> _CameraState:
        if self._cur is None:
            raise RuntimeError("LiveTracks: update/end_video without begin_video")
        return self._cur
