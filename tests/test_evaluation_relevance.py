"""Property 49: Relevance is camera match plus interval overlap.

Requirement 16.2 counts an Event as relevant iff its camera ID equals the ground-truth camera ID
and ``event_start <= gt_end AND event_end >= gt_start``. Both comparisons are INCLUSIVE, so an
Event that merely touches the ground-truth window at a single instant (its end equals gt start,
or its start equals gt end) is relevant; this file asserts the inclusive reading.

The oracle works on integer epoch microseconds so comparisons are exact. Every instant is
rendered in an independently chosen fixed UTC offset, so wall-clock order and instant order
differ; a wall-clock comparison would disagree with the oracle.

**Validates: Requirements 16.2**
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from nab_sentry.evaluation import EvalQuery, is_relevant
from tests.strategies import utc_offsets

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
US = timedelta(microseconds=1)

# Instants between 1971 and ~2100 (in microseconds since the epoch); leaves room for shifts and
# for +/-23:59 offsets without leaving datetime's range.
MIN_US = 365 * 86_400 * 1_000_000
MAX_US = 130 * 365 * 86_400 * 1_000_000
HOUR_US = 3_600 * 1_000_000

CAMERA_POOL = ["cam-a", "cam-b", "cam-c"]
cameras = st.sampled_from(CAMERA_POOL)


def to_us(dt: datetime) -> int:
    return (dt - EPOCH) // US


def render(us: int, tz: timezone) -> datetime:
    return (EPOCH + timedelta(microseconds=us)).astimezone(tz)


def oracle(ev_cam: str, ev_s: int, ev_e: int, gt_cam: str, gt_s: int, gt_e: int) -> bool:
    return ev_cam == gt_cam and ev_s <= gt_e and ev_e >= gt_s


@st.composite
def scenarios(draw):
    """(gt_start_us, gt_end_us, ev_start_us, ev_end_us) with gt_s < gt_e and ev_s <= ev_e.

    Event endpoints are biased toward the ground-truth boundaries (exact and +/-1 us) so that
    touching endpoints and zero-length Events are exercised often.
    """
    gt_s = draw(st.integers(MIN_US, MAX_US))
    gt_e = gt_s + draw(st.one_of(st.just(1), st.integers(1, 2 * HOUR_US)))

    def endpoint():
        anchor = draw(st.sampled_from([gt_s, gt_e]))
        return draw(
            st.one_of(
                st.sampled_from([anchor - 1, anchor, anchor + 1]),
                st.integers(anchor - 3 * HOUR_US, anchor + 3 * HOUR_US),
            )
        )

    a = endpoint()
    if draw(st.booleans()):
        b = a  # zero-length Event
    else:
        b = endpoint()
    ev_s, ev_e = min(a, b), max(a, b)
    return gt_s, gt_e, ev_s, ev_e


def make_gt(camera: str, gt_s: int, gt_e: int, tz_s: timezone, tz_e: timezone) -> EvalQuery:
    return EvalQuery(
        index=0, text="q", camera_id=camera, start=render(gt_s, tz_s), end=render(gt_e, tz_e)
    )


@settings(max_examples=500)
@given(
    scen=scenarios(),
    ev_cam=cameras,
    gt_cam=cameras,
    tzs=st.tuples(utc_offsets, utc_offsets, utc_offsets, utc_offsets),
)
def test_relevance_matches_oracle(scen, ev_cam, gt_cam, tzs):
    """is_relevant == camera equality AND inclusive overlap, on exact instants (Property 49)."""
    gt_s, gt_e, ev_s, ev_e = scen
    gt = make_gt(gt_cam, gt_s, gt_e, tzs[0], tzs[1])
    ev_start, ev_end = render(ev_s, tzs[2]), render(ev_e, tzs[3])
    # Sanity: rendering preserves the instant.
    assert to_us(ev_start) == ev_s and to_us(gt.end) == gt_e

    assert is_relevant(ev_cam, ev_start, ev_end, gt) == oracle(
        ev_cam, ev_s, ev_e, gt_cam, gt_s, gt_e
    )


@settings(max_examples=200)
@given(
    scen=scenarios(),
    cam=cameras,
    shift=st.integers(-10 * 365 * 86_400 * 1_000_000, 10 * 365 * 86_400 * 1_000_000),
    tzs=st.tuples(utc_offsets, utc_offsets, utc_offsets, utc_offsets),
    tzs2=st.tuples(utc_offsets, utc_offsets, utc_offsets, utc_offsets),
)
def test_relevance_invariant_under_common_shift_and_rezoning(scen, cam, shift, tzs, tzs2):
    """Shifting both intervals by the same delta (and re-rendering in other zones) keeps the result."""
    gt_s, gt_e, ev_s, ev_e = scen
    before = is_relevant(
        cam, render(ev_s, tzs[2]), render(ev_e, tzs[3]), make_gt(cam, gt_s, gt_e, tzs[0], tzs[1])
    )
    after = is_relevant(
        cam,
        render(ev_s + shift, tzs2[2]),
        render(ev_e + shift, tzs2[3]),
        make_gt(cam, gt_s + shift, gt_e + shift, tzs2[0], tzs2[1]),
    )
    assert before == after


@settings(max_examples=200)
@given(
    scen=scenarios(),
    cams=st.lists(cameras, min_size=2, max_size=2, unique=True),
    tzs=st.tuples(utc_offsets, utc_offsets, utc_offsets, utc_offsets),
)
def test_different_camera_is_never_relevant(scen, cams, tzs):
    """Even a fully overlapping window is irrelevant once the camera differs."""
    gt_s, gt_e, ev_s, ev_e = scen
    gt = make_gt(cams[0], gt_s, gt_e, tzs[0], tzs[1])
    ev_start, ev_end = render(ev_s, tzs[2]), render(ev_e, tzs[3])
    # Same camera result follows the oracle; switching the camera forces False.
    assert is_relevant(cams[0], ev_start, ev_end, gt) == oracle(
        cams[0], ev_s, ev_e, cams[0], gt_s, gt_e
    )
    assert is_relevant(cams[1], ev_start, ev_end, gt) is False


def test_touching_endpoints_are_relevant_examples():
    """Inclusive boundaries: touching at one instant counts; 1 us apart does not."""
    plus2 = timezone(timedelta(hours=2))
    minus5 = timezone(timedelta(hours=-5))
    gt = EvalQuery(
        0, "q", "cam-a",
        datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc),
    )
    # Event end == gt start (same instant, expressed in +02:00).
    assert is_relevant("cam-a", datetime(2024, 1, 1, 13, 0, tzinfo=plus2) - timedelta(hours=1),
                       datetime(2024, 1, 1, 14, 0, tzinfo=plus2), gt)
    # Zero-length Event exactly at gt end, expressed in -05:00.
    at_end = datetime(2024, 1, 1, 8, 0, tzinfo=minus5)
    assert is_relevant("cam-a", at_end, at_end, gt)
    # One microsecond after gt end.
    assert not is_relevant("cam-a", at_end + US, at_end + US, gt)
    # Starts 1 us after gt end (13:00:00.000001Z), so no overlap regardless of length.
    assert not is_relevant("cam-a", at_end + US, at_end + timedelta(minutes=30), gt)
