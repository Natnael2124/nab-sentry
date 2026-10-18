"""Multi-object tracking (V2 Requirement 10).

:class:`TrackSink` is what the Ingest_Pipeline feeds: every frame on which the detector ran
at the tracking rate, with all its detections (high and low). The ByteTrack-style tracker
(task 10.1) implements it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from nab_sentry.ingest.detector import Detection
from nab_sentry.ingest.sources import DecodedFrame


@runtime_checkable
class TrackSink(Protocol):
    @property
    def active(self) -> bool:
        """True while any track is alive; the pipeline then keeps detecting even when the
        motion gate rejects the frame (a person standing still must not be dropped)."""
        ...

    def update(self, frame: DecodedFrame, detections: Sequence[Detection]) -> None:
        """Consume one detector run, frames in increasing ``offset_s`` order."""
        ...
