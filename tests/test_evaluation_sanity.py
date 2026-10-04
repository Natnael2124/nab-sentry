"""Sanity tests for nab_sentry.evaluation (Requirement 16)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from nab_sentry.evaluation import (
    EvalFileError,
    EvalQuery,
    QueryResult,
    aggregate,
    hit_at_1,
    is_relevant,
    load_queries,
    load_queries_file,
    precision_at_5,
)

UTC = timezone.utc


def _yaml(entries: list[str]) -> str:
    return "queries:\n" + "".join(entries)


def _entry(text="red square", cam="SYNTH01", start="2025-01-01T08:00:10+00:00",
           end="2025-01-01T08:00:20+00:00") -> str:
    return f'  - text: "{text}"\n    camera_id: {cam}\n    start: "{start}"\n    end: "{end}"\n'


def test_load_valid_file():
    queries, errors = load_queries(_yaml([_entry(text=f"q{i}") for i in range(15)]))
    assert errors == [] and len(queries) == 15
    q = queries[0]
    assert q.camera_id == "SYNTH01" and q.start.tzinfo is not None and q.end > q.start


@pytest.mark.parametrize("n", [14, 21])
def test_count_out_of_range_raises(n):
    with pytest.raises(EvalFileError, match="expected 15 to 20"):
        load_queries(_yaml([_entry() for _ in range(n)]))


@pytest.mark.parametrize("text", ["queries: [unclosed", "- just a list\n", "queries: 5\n"])
def test_bad_document_raises(text):
    with pytest.raises(EvalFileError):
        load_queries(text)


def test_missing_file_raises(tmp_path):
    with pytest.raises(EvalFileError, match="not found"):
        load_queries_file(tmp_path / "nope.yaml")


def test_per_query_errors_do_not_abort():
    entries = [_entry(text=f"q{i}") for i in range(13)]
    entries.append('  - camera_id: X\n    start: "2025-01-01T08:00:00"\n    end: "2025-01-01T08:00:01"\n')
    entries.append(_entry(text="backwards", start="2025-01-01T08:00:20", end="2025-01-01T08:00:20"))
    entries.append('  - text: "no cam"\n    start: "2025-01-01T08:00:00"\n    end: "2025-01-01T08:00:01"\n')
    queries, errors = load_queries(_yaml(entries))
    assert len(queries) == 13
    reasons = {e.index: e.reason for e in errors}
    assert "missing query text" in reasons[13]
    assert "not later" in reasons[14]
    assert "camera_id" in reasons[15]


def test_naive_times_become_local_aware():
    queries, _ = load_queries(_yaml([_entry(start="2025-01-01T08:00:10", end="2025-01-01T08:00:20")] * 15))
    assert queries[0].start.utcoffset() is not None


def test_is_relevant_inclusive_overlap():
    gt = EvalQuery(0, "q", "CAM01", datetime(2025, 1, 1, 8, 0, 10, tzinfo=UTC),
                   datetime(2025, 1, 1, 8, 0, 20, tzinfo=UTC))
    t = lambda s: datetime(2025, 1, 1, 8, 0, s, tzinfo=UTC)  # noqa: E731
    assert is_relevant("CAM01", t(20), t(25), gt)          # touches end
    assert is_relevant("CAM01", t(5), t(10), gt)           # touches start
    assert not is_relevant("CAM01", t(21), t(25), gt)
    assert not is_relevant("CAM01", t(0), t(9), gt)
    assert not is_relevant("CAM02", t(12), t(15), gt)
    # Same instant expressed in a different offset still overlaps.
    plus1 = timezone(timedelta(hours=1))
    assert is_relevant("CAM01", datetime(2025, 1, 1, 9, 0, 15, tzinfo=plus1),
                       datetime(2025, 1, 1, 9, 0, 16, tzinfo=plus1), gt)


def test_metrics():
    assert precision_at_5([]) == 0.0 and hit_at_1([]) == 0
    assert precision_at_5([True, False, True]) == pytest.approx(0.4)
    assert precision_at_5([True] * 8) == 1.0
    assert hit_at_1([True, False]) == 1 and hit_at_1([False, True]) == 0


def test_aggregate_excludes_errors():
    agg = aggregate([
        QueryResult("a", 0.4, 1, 10.0),
        QueryResult("b", 0.0, 0, 30.0),
        QueryResult("c", error="unknown camera ID"),
    ])
    assert agg.precision_at_5 == pytest.approx(0.2)
    assert agg.hit_at_1 == pytest.approx(0.5)
    assert agg.latency_ms_mean == pytest.approx(20.0)
    assert (agg.queries_run, agg.queries_errored) == (3, 1)


def test_aggregate_all_errors():
    agg = aggregate([QueryResult("a", error="x")])
    assert agg.precision_at_5 is None and agg.hit_at_1 is None and agg.latency_ms_mean is None
    assert agg.queries_errored == 1
