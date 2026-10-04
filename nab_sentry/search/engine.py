"""Search engine: filter validation, readiness check, filtered search and clustering.

Requirements 8.2, 8.3, 8.5, 8.12-8.15 and 9.15. ``search`` runs in the design's order:
``validate_filter`` -> ``encoder.encode_text`` -> allowed set (empty -> ``[]`` without touching
the index) -> ``index.search`` -> ``db.hit_rows`` -> ``cluster_hits`` -> ``[:limit]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from nab_sentry.ingest.detector import TARGET_CLASSES
from nab_sentry.search.clustering import ClusterHit, ClusterParams, Event, VideoInfo, cluster_hits
from nab_sentry.store.db import MetadataStore, as_aware

if TYPE_CHECKING:
    from nab_sentry.config import Config
    from nab_sentry.embed.clip_encoder import Encoder
    from nab_sentry.store.vector_index import VectorIndex

__all__ = [
    "SearchFilter",
    "FilterError",
    "SearchUnavailable",
    "IndexInconsistent",
    "VALID_CLASSES",
    "validate_filter",
    "SearchEngine",
]

# Valid ``cls`` filter values: the Target_Classes names (8.14).
VALID_CLASSES: frozenset[str] = frozenset(TARGET_CLASSES.values())


@dataclass(frozen=True)
class SearchFilter:
    """Optional search restrictions; set parts are combined with logical AND (8.3).

    ``start``/``end`` are aware datetimes; either may be omitted (unbounded on that side, 8.9).
    A naive value is interpreted in the local time zone, as everywhere else in the store (1.12).
    ``cls`` restricts to ``crop`` vectors of that detection class (8.8).
    """

    camera_id: str | None = None
    start: datetime | None = None
    end: datetime | None = None
    cls: str | None = None

    def is_empty(self) -> bool:
        return self.camera_id is None and self.start is None and self.end is None and self.cls is None


class FilterError(ValueError):
    """An invalid filter part; ``part`` is ``"time_range"`` or ``"cls"`` (8.14)."""

    def __init__(self, part: str, message: str) -> None:
        super().__init__(message)
        self.part = part
        self.message = message


class SearchUnavailable(RuntimeError):
    """Searches are refused: encoder not loaded or index unusable/inconsistent (8.15, 10.13)."""


class IndexInconsistent(SearchUnavailable):
    """The FAISS ID set differs from the ``vectors`` table ID set (8.15)."""


def validate_filter(f: SearchFilter) -> None:
    """Raise ``FilterError`` for ``start > end`` (``time_range``) or an unknown class (``cls``)."""
    if f.start is not None and f.end is not None and as_aware(f.start) > as_aware(f.end):
        raise FilterError(
            "time_range",
            f"time range start {f.start.isoformat()} is later than end {f.end.isoformat()}",
        )
    if f.cls is not None and f.cls not in VALID_CLASSES:
        raise FilterError(
            "cls", f"unknown class {f.cls!r}; expected one of {sorted(VALID_CLASSES)}"
        )


class SearchEngine:
    def __init__(
        self,
        cfg: Config,
        db: MetadataStore,
        index: VectorIndex,
        encoder: Encoder | None,
    ) -> None:
        self.cfg = cfg
        self.db = db
        self.index = index
        self.encoder = encoder
        self.params = ClusterParams(
            merge_gap_s=float(cfg.merge_gap_s),
            padding_s=float(cfg.event_padding_s),
            label_boost=float(cfg.label_boost),
        )
        self._consistent = False  # ID sets compared once, on first successful check (design)

    # -- readiness -----------------------------------------------------------------------------

    def check_ready(self) -> None:
        """Raise ``SearchUnavailable`` if the encoder is missing or the ID sets differ (8.15)."""
        if self.encoder is None:
            raise SearchUnavailable("search unavailable: text encoder is not loaded")
        if self._consistent:
            return
        if self.index is None:
            raise SearchUnavailable("search unavailable: vector index is not loaded")
        faiss_ids = np.unique(np.asarray(self.index.ids(), dtype=np.int64))
        db_ids = np.unique(np.asarray(self.db.vector_ids(), dtype=np.int64))
        if faiss_ids.shape != db_ids.shape or not np.array_equal(faiss_ids, db_ids):
            missing = np.setdiff1d(db_ids, faiss_ids).size
            orphan = np.setdiff1d(faiss_ids, db_ids).size
            raise IndexInconsistent(
                "search unavailable: vector index is inconsistent with the metadata store "
                f"({orphan} index IDs without rows, {missing} rows without index IDs); "
                "run scripts/ingest.py --repair"
            )
        self._consistent = True

    # -- search --------------------------------------------------------------------------------

    def search_vector(self, qvec: np.ndarray, f: SearchFilter, query_text: str) -> list[Event]:
        """Filtered vector search followed by clustering (8.2, 8.3, 8.5, 8.12, 9.15)."""
        validate_filter(f)
        allowed: np.ndarray | None = None
        if not f.is_empty():
            allowed = self.db.allowed_vector_ids(f)
            if allowed.size == 0:
                return []  # 8.12, 8.13: no index call
        hits = self.index.search(qvec, int(self.cfg.top_k), allowed)
        if allowed is not None and hits:
            # Defensive: the index already restricts, but never return an ID outside the set (8.5).
            allowed_set = set(allowed.tolist())
            hits = [h for h in hits if h.vector_id in allowed_set]
        if not hits:
            return []  # 9.15

        rows = self.db.hit_rows([h.vector_id for h in hits])
        cluster_in: list[ClusterHit] = []
        videos: dict[int, VideoInfo] = {}
        for h in hits:
            row = rows.get(h.vector_id)
            if row is None:  # row vanished (e.g. concurrent repair): drop the hit
                continue
            cluster_in.append(
                ClusterHit(
                    vector_id=row.vector_id,
                    camera_id=row.camera_id,
                    video_id=row.video_id,
                    frame_offset_s=float(row.offset_s),
                    similarity=float(h.similarity),
                    thumb_path=row.thumb_path,
                )
            )
            if row.video_id not in videos:
                videos[row.video_id] = VideoInfo(
                    duration_s=float(row.duration_s),
                    start_epoch_ms=int(row.start_epoch_ms),
                    start_time=row.start_ts,
                    camera_label=row.camera_label,
                )
        return cluster_hits(cluster_in, videos, query_text, self.params)

    def search(self, query: str, f: SearchFilter, limit: int) -> list[Event]:
        """Validate, encode, search, cluster and return at most ``limit`` Events."""
        validate_filter(f)
        self.check_ready()
        assert self.encoder is not None  # check_ready guarantees it
        qvec = self.encoder.encode_text(query)  # EmptyQueryError propagates
        events = self.search_vector(qvec, f, query)
        return events[: max(0, int(limit))]
