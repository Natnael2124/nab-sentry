"""``/ask`` search for the Notifier (Step 3, Requirement 20.2).

Runs the V1 text search over a time window. The OpenCLIP encoder is loaded on the first
question only (a Notifier nobody asks never pays its memory), and the Vector_Index is read from
disk on every question because the Watcher keeps changing it. ``search_vector`` is used with
an allowed-ID filter from the store, so vectors the Watcher saved but has not committed yet are
simply not found.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from nab_sentry.config import Config
from nab_sentry.embed.compat import check_embedder
from nab_sentry.logging_setup import get_logger
from nab_sentry.search.engine import SearchEngine, SearchFilter
from nab_sentry.store.db import MetadataStore
from nab_sentry.store.vector_index import DEFAULT_DIM, VectorIndex

log = get_logger("notify.search")


class TodaySearch:
    def __init__(self, cfg: Config, db: MetadataStore, encoder_factory: Callable[[], Any],
                 limit: int = 5) -> None:
        self.cfg = cfg
        self.db = db
        self.encoder_factory = encoder_factory
        self.limit = limit
        self._encoder: Any = None

    def __call__(self, words: str, from_s: float, to_s: float) -> list[dict]:
        if self._encoder is None:
            self._encoder = self.encoder_factory()
            check_embedder(self.db, self._encoder)  # never search another embedder's vectors
        if not self.cfg.index_path.is_file():
            return []
        index = VectorIndex.load(self.cfg.index_path, DEFAULT_DIM)
        engine = SearchEngine(self.cfg, self.db, index, self._encoder)
        f = SearchFilter(start=datetime.fromtimestamp(from_s).astimezone(),
                         end=datetime.fromtimestamp(to_s).astimezone())
        events = engine.search_vector(self._encoder.encode_text(words), f, words)[: self.limit]
        out = []
        for e in events:
            start_ms = self.db.video_start_ms(e.video_id)
            if start_ms is None:
                continue
            out.append({"camera_id": e.camera_id, "start_s": start_ms / 1000 + e.start_s,
                        "thumb_path": e.thumb_path, "score": e.score})
        return out
