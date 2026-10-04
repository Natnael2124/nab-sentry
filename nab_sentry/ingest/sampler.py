"""Frame sampler: pick the first frame at or after each target time ``k / rate``.

Requirements 2.1, 2.2, 2.4, 2.5, 2.6.

Both the pure reference :func:`sample_indices` and the streaming
:meth:`Sampler.select` apply the same rule: keep ``k = 0``; for each yielded
frame in order, if ``offset_s >= k / rate - EPS`` select it and ``k += 1``.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Iterator

from nab_sentry.ingest.sources import DecodedFrame

EPS = 1e-9


def _check_rate(rate: float) -> float:
    r = float(rate)
    if not math.isfinite(r) or r <= 0:
        raise ValueError(f"sample rate must be a positive finite number, got {rate!r}")
    return r


def _always(_: int) -> bool:
    return True


def sample_indices(n_frames: int, fps: float, rate: float,
                   decodable: Callable[[int], bool] = _always) -> list[int]:
    """Pure reference: indices selected from ``n_frames`` frames at ``fps``.

    Frames for which ``decodable(i)`` is false are never yielded by the source,
    so they are skipped without consuming a target.
    """
    r = _check_rate(rate)
    f = float(fps)
    if not math.isfinite(f) or f <= 0:
        raise ValueError(f"fps must be a positive finite number, got {fps!r}")
    selected: list[int] = []
    k = 0
    for i in range(max(0, int(n_frames))):
        if not decodable(i):
            continue
        if i / f >= k / r - EPS:
            selected.append(i)
            k += 1
    return selected


class Sampler:
    """Streaming sampler over a ``VideoSource.frames()`` iterator."""

    def __init__(self, rate: float) -> None:
        self.rate = _check_rate(rate)
        self.selected_count = 0

    def select(self, frames: Iterable[DecodedFrame]) -> Iterator[DecodedFrame]:
        """Yield frames chosen by the target-time rule; each call starts at ``k = 0``."""
        k = 0
        self.selected_count = 0
        for frame in frames:
            if frame.offset_s >= k / self.rate - EPS:
                k += 1
                self.selected_count = k
                yield frame
