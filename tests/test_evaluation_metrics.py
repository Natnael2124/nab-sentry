"""Property 50: Metric bounds for precision@5, hit@1 and their aggregation.

**Validates: Requirements 16.3, 16.4**
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.evaluation import QueryResult, aggregate, hit_at_1, precision_at_5

TOL = 1e-12
ALLOWED_PRECISIONS = {k / 5 for k in range(6)}  # {0, 0.2, 0.4, 0.6, 0.8, 1.0}

flag_lists = st.lists(st.booleans(), min_size=0, max_size=50)
latencies = st.floats(min_value=0.0, max_value=60_000.0, allow_nan=False, allow_infinity=False)


@st.composite
def ok_results(draw: st.DrawFn) -> QueryResult:
    flags = draw(flag_lists)
    return QueryResult(
        text=draw(st.text(min_size=1, max_size=20)),
        precision_at_5=precision_at_5(flags),
        hit_at_1=hit_at_1(flags),
        latency_ms=draw(latencies),
    )


@st.composite
def error_results(draw: st.DrawFn) -> QueryResult:
    # Error rows may carry stray metric values; aggregate must ignore them.
    return QueryResult(
        text=draw(st.text(min_size=1, max_size=20)),
        precision_at_5=draw(st.none() | st.floats(-5, 5, allow_nan=False)),
        hit_at_1=draw(st.none() | st.integers(-3, 3)),
        latency_ms=draw(st.none() | latencies),
        error=draw(st.text(min_size=0, max_size=20)),
    )


result_lists = st.lists(st.one_of(ok_results(), error_results()), min_size=0, max_size=25)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


# ---------------------------------------------------------------- per-query metrics


# Feature: nab-sentry, Property 50: Metric bounds
@given(flags=flag_lists)
def test_per_query_metric_bounds(flags: list[bool]) -> None:
    """**Validates: Requirements 16.3, 16.4**"""
    p = precision_at_5(flags)
    h = hit_at_1(flags)

    # 16.4: bounds, multiple of 0.2, hit@1 is 0 or 1.
    assert 0.0 <= p <= 1.0
    assert p in ALLOWED_PRECISIONS
    assert h in (0, 1)
    # 16.4: hit@1 = 1 implies precision@5 >= 0.2.
    if h == 1:
        assert p >= 0.2

    # 16.3: definitions (missing ranks count as not relevant; empty list -> hit@1 = 0).
    assert p == sum(flags[:5]) / 5
    assert h == (1 if flags and flags[0] else 0)


# Feature: nab-sentry, Property 50: Metric bounds
@given(head=st.lists(st.booleans(), max_size=5), tail_a=flag_lists, tail_b=flag_lists)
def test_precision_depends_only_on_first_five(
    head: list[bool], tail_a: list[bool], tail_b: list[bool]
) -> None:
    """**Validates: Requirements 16.3, 16.4**"""
    # Only meaningful tails once the head fills all 5 ranks; otherwise pad with False
    # (missing ranks count as not relevant).
    padded = head + [False] * (5 - len(head))
    assert precision_at_5(head) == precision_at_5(padded)
    assert precision_at_5(padded + tail_a) == precision_at_5(padded + tail_b)
    assert hit_at_1(padded + tail_a) == hit_at_1(padded + tail_b)


# ---------------------------------------------------------------- aggregation


# Feature: nab-sentry, Property 50: Metric bounds
@given(results=result_lists)
def test_aggregate_bounds_and_counts(results: list[QueryResult]) -> None:
    """**Validates: Requirements 16.3, 16.4**"""
    agg = aggregate(results)
    ok = [r for r in results if r.error is None]

    assert agg.queries_run == len(results)
    assert agg.queries_errored == len(results) - len(ok)

    if not ok:
        assert agg.precision_at_5 is None
        assert agg.hit_at_1 is None
        assert agg.latency_ms_mean is None
        return

    assert agg.precision_at_5 is not None and agg.hit_at_1 is not None
    assert 0.0 <= agg.precision_at_5 <= 1.0
    assert 0.0 <= agg.hit_at_1 <= 1.0
    assert abs(agg.precision_at_5 - _mean([r.precision_at_5 for r in ok])) <= TOL
    assert abs(agg.hit_at_1 - _mean([float(r.hit_at_1) for r in ok])) <= TOL
    assert agg.latency_ms_mean is not None
    assert abs(agg.latency_ms_mean - _mean([r.latency_ms for r in ok])) <= 1e-9 * max(
        1.0, agg.latency_ms_mean
    )


# Feature: nab-sentry, Property 50: Metric bounds
@given(results=result_lists, extra_errors=st.lists(error_results(), max_size=10), data=st.data())
def test_aggregate_ignores_error_rows_and_order(
    results: list[QueryResult], extra_errors: list[QueryResult], data: st.DataObject
) -> None:
    """**Validates: Requirements 16.3, 16.4**"""
    base = aggregate(results)

    # Adding error rows at random positions changes only the counts.
    mixed = list(results)
    for err in extra_errors:
        mixed.insert(data.draw(st.integers(0, len(mixed)), label="insert_at"), err)
    with_errors = aggregate(mixed)
    assert with_errors.queries_run == base.queries_run + len(extra_errors)
    assert with_errors.queries_errored == base.queries_errored + len(extra_errors)

    # Removing every error row likewise changes only the counts.
    only_ok = [r for r in results if r.error is None]
    stripped = aggregate(only_ok)
    assert stripped.queries_run == len(only_ok)
    assert stripped.queries_errored == 0

    # Order independence.
    shuffled = data.draw(st.permutations(mixed), label="shuffled")
    reordered = aggregate(shuffled)
    assert reordered.queries_run == with_errors.queries_run
    assert reordered.queries_errored == with_errors.queries_errored

    for other in (with_errors, stripped, reordered):
        for field in ("precision_at_5", "hit_at_1", "latency_ms_mean"):
            a, b = getattr(base, field), getattr(other, field)
            if a is None or b is None:
                assert a is None and b is None, field
            else:
                tol = TOL if field != "latency_ms_mean" else 1e-9 * max(1.0, abs(a))
                assert abs(a - b) <= tol, field
