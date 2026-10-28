"""Multi-object tracking (V2 Requirement 10).

:class:`TrackSink` is what the Ingest_Pipeline feeds: every frame on which the detector ran
at the tracking rate, with all its detections (high and low), bracketed per video by
``begin_video`` / ``end_video`` (or ``rollback_video`` when the video's ingest is rolled
back). :class:`nab_sentry.track.live.LiveTracks` implements it on top of one
:class:`nab_sentry.track.bytetrack.ByteTracker` per camera.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from nab_sentry.ingest.detector import Detection
from nab_sentry.ingest.sources import DecodedFrame


@runtime_checkable
class TrackSink(Protocol):
    @property
    def active(self) -> bool:
        """True while any track is alive; the pipeline then keeps detecting even when the
        motion gate rejects the frame (a person standing still must not be dropped)."""
        ...

    def begin_video(self, camera_id: str, start_t: float, video_id: int | None = None) -> None:
        """A video of ``camera_id`` starting at ``start_t`` (epoch seconds) begins; observations
        made during it are tagged ``(video_id, frame offset_s)``."""
        ...

    def update(self, frame: DecodedFrame, detections: Sequence[Detection]) -> Sequence[Any]:
        """Consume one detector run, frames in increasing ``offset_s`` order; returns one
        assignment per detection, aligned with ``detections``, with ``.detection``,
        ``.track_key`` (``None``: no track) and ``.confirmed``
        (:class:`nab_sentry.track.bytetrack.Assignment`)."""
        ...

    def end_video(self, end_t: float) -> list[Any]:
        """The video's frames are done (called inside its transaction); returns the drained
        track records of this video."""
        ...

    def rollback_video(self) -> None:
        """Undo everything since the matching ``begin_video`` (the video was rolled back)."""
        ...
