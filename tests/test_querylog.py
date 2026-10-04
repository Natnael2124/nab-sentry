"""Unit tests for the Query_Log (task 14.2).

Requirements 10.10 (append-only, one line per request, earlier lines unchanged)
and 10.14 (a failed append does not raise; the caller keeps serving).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from nab_sentry.querylog import QueryLog, QueryRecord


def _record(i: int, status: int = 200) -> QueryRecord:
    return QueryRecord(
        status=status,
        q=f"white pickup truck {i} — ünïcode",
        filters={"camera": "GATE1", "start": None, "end": None, "cls": "truck", "limit": 20},
        result_count=0 if status == 422 else i,
        latency_ms=12.5 + i,
        ts=f"2025-01-01T08:05:{i % 60:02d}.000+01:00",
    )


# --- Requirement 10.10: append preserves earlier lines ----------------------------


def test_append_preserves_earlier_lines_byte_for_byte(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "query_log.jsonl"  # parent created on first append
    log = QueryLog(path)

    first = [_record(i) for i in range(5)]
    assert all(log.append(r) for r in first)
    before = path.read_bytes()
    assert before.count(b"\n") == 5

    second = [_record(i, status=422) for i in range(5, 8)]
    assert all(log.append(r) for r in second)
    after = path.read_bytes()

    # Earlier bytes are an unchanged prefix; exactly one new line per record.
    assert after.startswith(before)
    tail = after[len(before):].decode("utf-8").splitlines()
    assert len(tail) == len(second)
    assert [json.loads(line) for line in tail] == [json.loads(r.to_json()) for r in second]

    lines = after.decode("utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [json.loads(r.to_json()) for r in first + second]


def test_append_keeps_pre_existing_file_content(tmp_path: Path) -> None:
    path = tmp_path / "query_log.jsonl"
    existing = '{"legacy": true}\nnot json but must survive\n'.encode("utf-8")
    path.write_bytes(existing)

    # A fresh QueryLog instance (e.g. after a server restart) must not truncate.
    assert QueryLog(path).append(_record(1)) is True
    assert QueryLog(path).append(_record(2)) is True

    data = path.read_bytes()
    assert data.startswith(existing)
    new_lines = data[len(existing):].decode("utf-8").splitlines()
    assert [json.loads(line) for line in new_lines] == [
        json.loads(_record(1).to_json()),
        json.loads(_record(2).to_json()),
    ]


def test_each_record_is_one_line_even_with_newlines_in_query(tmp_path: Path) -> None:
    path = tmp_path / "query_log.jsonl"
    rec = QueryRecord(status=422, q="line1\nline2\r\n", filters={"limit": "abc"},
                      result_count=0, latency_ms=0.4)
    assert QueryLog(path).append(rec) is True
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["q"] == "line1\nline2\r\n"


# --- Requirement 10.14: unwritable path returns False without raising ------------


def _unwritable_dir_path(tmp_path: Path) -> Path:
    target = tmp_path / "query_log.jsonl"
    target.mkdir()  # the log path is a directory
    return target


def _unwritable_parent_is_file(tmp_path: Path) -> Path:
    parent = tmp_path / "not_a_dir"
    parent.write_text("regular file", encoding="utf-8")
    return parent / "query_log.jsonl"  # parent path is a regular file


@pytest.mark.parametrize("make_path", [_unwritable_dir_path, _unwritable_parent_is_file],
                         ids=["path-is-directory", "parent-is-file"])
def test_unwritable_path_returns_false_and_logs_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, make_path
) -> None:
    path = make_path(tmp_path)
    log = QueryLog(path)

    with caplog.at_level(logging.WARNING, logger="nab_sentry"):
        result = log.append(_record(1))  # must not raise

    assert result is False
    warnings = [r for r in caplog.records
                if r.levelno == logging.WARNING and r.name.startswith("nab_sentry")]
    assert warnings, "expected a warning on the nab_sentry logger"
    assert "Query_Log" in warnings[0].getMessage()

    # The log keeps accepting later calls (still failing, still not raising).
    assert log.append(_record(2)) is False


def test_failure_does_not_damage_later_appends_elsewhere(tmp_path: Path) -> None:
    bad = QueryLog(_unwritable_parent_is_file(tmp_path))
    good_path = tmp_path / "good.jsonl"
    good = QueryLog(good_path)

    assert good.append(_record(1)) is True
    assert bad.append(_record(2)) is False
    assert good.append(_record(3)) is True
    assert len(good_path.read_text(encoding="utf-8").splitlines()) == 2
