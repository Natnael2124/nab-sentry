"""Quick example-based sanity checks for the sampler (properties live in test_sampler.py)."""

import numpy as np
import pytest

from nab_sentry.ingest.sampler import Sampler, sample_indices
from nab_sentry.ingest.sources import DecodedFrame


def _frames(n, fps, skip=()):
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    return [DecodedFrame(i, i / fps, img) for i in range(n) if i not in skip]


def test_one_per_second_at_30fps():
    # 10 s at 30 fps, rate 1 -> frames 0, 30, ..., 270
    assert sample_indices(300, 30.0, 1.0) == list(range(0, 300, 30))
    assert [f.index for f in Sampler(1.0).select(_frames(300, 30.0))] == list(range(0, 300, 30))


def test_fps_below_rate_selects_every_frame():
    assert sample_indices(20, 2.0, 5.0) == list(range(20))


def test_undecodable_target_frame_falls_to_next():
    skip = {30, 31}
    expected = sample_indices(90, 30.0, 1.0, decodable=lambda i: i not in skip)
    assert expected == [0, 32, 60]
    got = [f.index for f in Sampler(1.0).select(_frames(90, 30.0, skip))]
    assert got == expected


def test_empty_and_selected_count():
    s = Sampler(1.0)
    assert list(s.select([])) == []
    assert s.selected_count == 0
    list(s.select(_frames(60, 30.0)))
    assert s.selected_count == 2


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_invalid_rate_rejected(rate):
    with pytest.raises(ValueError):
        Sampler(rate)
