"""Exceptions shared by more than one component.

``ModelMissingError`` is raised by both the Embedder (``embed/clip_encoder.py``,
Requirement 5.8) and the Detector (``ingest/detector.py``, Requirement 4.6), so it
lives here rather than in either module. Stdlib-only, like ``startup.py``.
"""

from __future__ import annotations

from pathlib import Path

from nab_sentry.startup import FETCH_HINT


class ModelMissingError(RuntimeError):
    """A model weight file is absent or unreadable at load time; never triggers a download."""

    def __init__(self, path: Path | str, detail: str = "missing or unreadable") -> None:
        self.path = Path(path)
        self.detail = detail
        super().__init__(
            f"Model weights {detail}: {self.path}. Run the Model_Fetcher: {FETCH_HINT}"
        )
