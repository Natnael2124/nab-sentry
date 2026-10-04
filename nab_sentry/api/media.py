"""Media serving helpers: HTTP Range parsing, safe path resolution, file streaming.

Everything here is framework-free so it can be property-tested directly. The
FastAPI routes in ``app.py`` combine ``parse_range`` + ``media_response_plan``
with ``iter_file`` to build 200/206/416 responses (Requirement 11), and use
``resolve_thumb`` / ``resolve_playback`` to turn untrusted names into paths that
are guaranteed to live directly inside their media directory (11.8, 18.6).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Union

__all__ = [
    "FullBody",
    "PartialBody",
    "Unsatisfiable",
    "RangeResult",
    "RANGE_RE",
    "THUMB_NAME_RE",
    "PLAYBACK_NAME_RE",
    "ResponsePlan",
    "parse_range",
    "media_response_plan",
    "resolve_thumb",
    "resolve_playback",
    "iter_file",
]


@dataclass(frozen=True)
class FullBody:
    """Serve the whole file with HTTP 200 (no or ignored Range header)."""


@dataclass(frozen=True)
class PartialBody:
    """Serve bytes ``start..end`` (zero-based, inclusive) with HTTP 206."""

    start: int
    end: int  # inclusive


@dataclass(frozen=True)
class Unsatisfiable:
    """Respond HTTP 416 with ``Content-Range: bytes */size`` and no file bytes."""


RangeResult = Union[FullBody, PartialBody, Unsatisfiable]

# ASCII digits only ([0-9], not \d, which also matches other Unicode digits).
# Always used with ``fullmatch`` so a trailing newline cannot sneak past ``$``.
RANGE_RE = re.compile(r"^bytes=([0-9]*)-([0-9]*)$")
THUMB_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}\.jpg$")
PLAYBACK_NAME_RE = re.compile(r"^v[0-9]+\.mp4$")

DEFAULT_CHUNK = 256 * 1024


def parse_range(header: str | None, size: int) -> RangeResult:
    """Map a ``Range`` header to a body plan for a file of ``size`` bytes.

    Rules (design table, Requirement 11):

    * ``None``, malformed, multi-range (contains ``,``), ``bytes=-`` or
      ``a > b`` -> ``FullBody`` (11.10)
    * ``bytes=a-b`` with ``a < size`` -> ``PartialBody(a, min(b, size-1))`` (11.2)
    * ``bytes=a-`` with ``a < size`` -> ``PartialBody(a, size-1)`` (11.3)
    * ``bytes=-n`` with ``n > 0`` -> ``PartialBody(max(0, size-n), size-1)`` (11.9)
    * start ``a >= size`` or ``bytes=-0`` -> ``Unsatisfiable`` (11.5)

    For ``size == 0`` every well-formed range is ``Unsatisfiable``; ignored
    headers still give ``FullBody`` (an empty 200).
    """
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}")
    if header is None or "," in header:
        return FullBody()
    m = RANGE_RE.fullmatch(header)
    if m is None:
        return FullBody()
    first, last = m.group(1), m.group(2)

    if first == "" and last == "":  # "bytes=-"
        return FullBody()

    if first == "":  # suffix range "bytes=-n"
        n = int(last)
        if n == 0 or size == 0:
            return Unsatisfiable()
        return PartialBody(max(0, size - n), size - 1)

    a = int(first)
    if last == "":  # open-ended "bytes=a-"
        if a >= size:
            return Unsatisfiable()
        return PartialBody(a, size - 1)

    b = int(last)
    if a > b:
        return FullBody()
    if a >= size:
        return Unsatisfiable()
    return PartialBody(a, min(b, size - 1))


@dataclass(frozen=True)
class ResponsePlan:
    """Status, headers, and byte span for a media response.

    ``start``/``end`` are inclusive; both are ``None`` when the body must
    contain no file bytes (416, or a 200 for an empty file).
    """

    status: int
    headers: dict[str, str]
    start: int | None
    end: int | None


def media_response_plan(result: RangeResult, size: int) -> ResponsePlan:
    """Build the status, ``Accept-Ranges``/``Content-Range``/``Content-Length``
    headers, and the inclusive byte span for a ``parse_range`` result."""
    if isinstance(result, PartialBody):
        length = result.end - result.start + 1
        return ResponsePlan(
            status=206,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {result.start}-{result.end}/{size}",
                "Content-Length": str(length),
            },
            start=result.start,
            end=result.end,
        )
    if isinstance(result, Unsatisfiable):
        return ResponsePlan(
            status=416,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes */{size}",
                "Content-Length": "0",
            },
            start=None,
            end=None,
        )
    # FullBody
    return ResponsePlan(
        status=200,
        headers={"Accept-Ranges": "bytes", "Content-Length": str(size)},
        start=0 if size > 0 else None,
        end=size - 1 if size > 0 else None,
    )


def _contained_file(base_dir: Path, name: str) -> Path | None:
    """Return ``(base_dir / name).resolve()`` if it is a regular file whose
    resolved parent is exactly ``base_dir.resolve()``; otherwise ``None``.

    Resolving follows symlinks, so a link pointing outside ``base_dir`` fails
    the parent check.
    """
    try:
        base = base_dir.resolve()
        candidate = (base_dir / name).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if candidate.parent != base:
        return None
    try:
        if not candidate.is_file():
            return None
    except (OSError, ValueError):
        return None
    return candidate


def resolve_thumb(thumbs_dir: Path, name: str) -> Path | None:
    """Resolve a requested Thumbnail name to a file directly inside ``thumbs_dir``.

    Only bare names matching ``^[A-Za-z0-9_-]{1,128}\\.jpg$`` are accepted, which
    rules out ``..``, ``/``, ``\\``, ``:`` (drive letters), ``%``-encodings and NUL.
    """
    if not isinstance(name, str) or THUMB_NAME_RE.fullmatch(name) is None:
        return None
    return _contained_file(thumbs_dir, name)


def resolve_playback(playback_dir: Path, stored_name: str | None) -> Path | None:
    """Resolve a stored ``videos.playback_path`` name (``v<id>.mp4``) to a file
    directly inside ``playback_dir``; ``None`` if invalid or missing."""
    if not isinstance(stored_name, str) or PLAYBACK_NAME_RE.fullmatch(stored_name) is None:
        return None
    return _contained_file(playback_dir, stored_name)


def iter_file(path: Path, start: int, end: int, chunk: int = DEFAULT_CHUNK) -> Iterator[bytes]:
    """Yield bytes ``start..end`` (inclusive) of ``path`` in chunks of at most
    ``chunk`` bytes. Stops early if the file is shorter than expected."""
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    if start < 0 or end < start:
        return
    remaining = end - start + 1
    with open(path, "rb") as fh:
        fh.seek(start)
        while remaining > 0:
            data = fh.read(min(chunk, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data
