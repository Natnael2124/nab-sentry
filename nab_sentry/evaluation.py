"""Pure relevance and metric functions for the Evaluator (Requirement 16).

``scripts/evaluate.py`` loads ``eval/queries.yaml`` with :func:`load_queries_file`, runs each
valid query through the Search_Engine, turns the ranked Events into relevance flags with
:func:`is_relevant`, scores them with :func:`precision_at_5` / :func:`hit_at_1`, and summarises
the run with :func:`aggregate`.

File-level problems (missing, unparseable, wrong shape, query count outside 15..20) raise
:class:`EvalFileError` (16.8). Problems with a single entry become :class:`QueryError` values so
the run continues with the remaining queries (16.7).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

import yaml

from nab_sentry.ingest.metadata import to_aware_local

MIN_QUERIES = 15
MAX_QUERIES = 20
TEXT_MAX_LEN = 200
GT_FIELDS = ("camera_id", "start", "end")


class EvalFileError(Exception):
    """``eval/queries.yaml`` is missing, unparseable, malformed, or has the wrong query count."""


@dataclass(frozen=True)
class EvalQuery:
    index: int              # 0-based position in the YAML ``queries`` list
    text: str
    camera_id: str
    start: datetime         # timezone-aware
    end: datetime           # timezone-aware, strictly later than ``start``


@dataclass(frozen=True)
class QueryError:
    index: int
    text: str | None        # query text if present, so the report row can name the query
    reason: str


@dataclass(frozen=True)
class QueryResult:
    text: str
    precision_at_5: float | None = None
    hit_at_1: int | None = None
    latency_ms: float | None = None
    error: str | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None


@dataclass(frozen=True)
class Aggregate:
    """Means over non-error queries; the metric fields are ``None`` when every query errored."""

    precision_at_5: float | None
    hit_at_1: float | None
    latency_ms_mean: float | None
    queries_run: int        # total queries processed (scored + errored)
    queries_errored: int


# --------------------------------------------------------------------------------------------
# Loading


def _parse_time(value: Any) -> datetime | None:
    """ISO 8601 string or YAML-native timestamp -> aware datetime; ``None`` if invalid."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        # A bare date is not an Absolute_Timestamp.
        return None
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.strip())
        except ValueError:
            return None
    else:
        return None
    return to_aware_local(dt)


def _parse_entry(index: int, entry: Any) -> EvalQuery | QueryError:
    if not isinstance(entry, dict):
        return QueryError(index, None, "query entry is not a mapping")

    raw_text = entry.get("text")
    text = raw_text if isinstance(raw_text, str) else None
    if text is None or not text.strip():
        return QueryError(index, text, "missing query text")
    if len(text) > TEXT_MAX_LEN:
        return QueryError(index, text, f"query text longer than {TEXT_MAX_LEN} characters")

    missing = [f for f in GT_FIELDS if entry.get(f) is None or entry.get(f) == ""]
    if missing:
        return QueryError(index, text, f"missing ground-truth field(s): {', '.join(missing)}")

    camera_id = entry["camera_id"]
    if not isinstance(camera_id, str):
        return QueryError(index, text, "camera_id must be a string")

    start = _parse_time(entry["start"])
    if start is None:
        return QueryError(index, text, "start is not an ISO 8601 date-time")
    end = _parse_time(entry["end"])
    if end is None:
        return QueryError(index, text, "end is not an ISO 8601 date-time")
    if end <= start:
        return QueryError(index, text, "ground-truth end time is not later than start time")

    return EvalQuery(index=index, text=text, camera_id=camera_id, start=start, end=end)


def load_queries(text: str) -> tuple[list[EvalQuery], list[QueryError]]:
    """Parse the YAML document; raise :class:`EvalFileError` for file-level problems (16.1, 16.8)."""
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise EvalFileError(f"cannot parse queries file: {exc}") from exc

    if not isinstance(doc, dict) or "queries" not in doc:
        raise EvalFileError("queries file must be a mapping with a top-level 'queries' list")
    entries = doc["queries"]
    if not isinstance(entries, list):
        raise EvalFileError("'queries' must be a list")
    if not MIN_QUERIES <= len(entries) <= MAX_QUERIES:
        raise EvalFileError(
            f"queries file holds {len(entries)} queries; expected {MIN_QUERIES} to {MAX_QUERIES}"
        )

    queries: list[EvalQuery] = []
    errors: list[QueryError] = []
    for i, entry in enumerate(entries):
        parsed = _parse_entry(i, entry)
        (queries if isinstance(parsed, EvalQuery) else errors).append(parsed)
    return queries, errors


def load_queries_file(path: str | Path) -> tuple[list[EvalQuery], list[QueryError]]:
    """Read ``path`` and delegate to :func:`load_queries`; a missing/unreadable file is an EvalFileError."""
    p = Path(path)
    try:
        content = p.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise EvalFileError(f"queries file not found: {p}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise EvalFileError(f"cannot read queries file {p}: {exc}") from exc
    return load_queries(content)


# --------------------------------------------------------------------------------------------
# Relevance and metrics


def is_relevant(ev_camera: str, ev_start: datetime, ev_end: datetime, gt: EvalQuery) -> bool:
    """Camera match plus inclusive overlap of the padded Event window with ground truth (16.2)."""
    if ev_camera != gt.camera_id:
        return False
    s = to_aware_local(ev_start)
    e = to_aware_local(ev_end)
    return s <= to_aware_local(gt.end) and e >= to_aware_local(gt.start)


def precision_at_5(flags: Sequence[bool]) -> float:
    """Relevant Events among the first 5 ranks divided by 5; missing ranks count as not relevant (16.3)."""
    return sum(1 for f in flags[:5] if f) / 5


def hit_at_1(flags: Sequence[bool]) -> int:
    """1 if the rank-1 Event is relevant, otherwise 0 (including no Events) (16.3)."""
    return 1 if len(flags) > 0 and flags[0] else 0


def aggregate(results: Sequence[QueryResult]) -> Aggregate:
    """Arithmetic means over non-error queries plus run/error counts (16.3, 16.7, 16.8)."""
    ok = [r for r in results if not r.is_error]
    errored = len(results) - len(ok)
    if not ok:
        return Aggregate(None, None, None, queries_run=len(results), queries_errored=errored)
    n = len(ok)
    return Aggregate(
        precision_at_5=sum(r.precision_at_5 or 0.0 for r in ok) / n,
        hit_at_1=sum(r.hit_at_1 or 0 for r in ok) / n,
        latency_ms_mean=sum(r.latency_ms or 0.0 for r in ok) / n,
        queries_run=len(results),
        queries_errored=errored,
    )
