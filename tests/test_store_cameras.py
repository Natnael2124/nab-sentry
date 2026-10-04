"""Property 6: Camera rows are created once and never relabelled.

**Validates: Requirements 1.9, 1.10**
"""

from __future__ import annotations

import logging

from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.store.db import MetadataStore
from tests.strategies import camera_ids, labels

STORE_LOGGER = "nab_sentry.store.db"


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


# Small pools so the sequence repeats IDs and labels often (exercising "exists" and "label_conflict").
upsert_sequences = st.tuples(
    st.lists(camera_ids, min_size=1, max_size=5, unique=True),
    st.lists(labels, min_size=1, max_size=4, unique=True),
).flatmap(
    lambda pools: st.lists(
        st.tuples(st.sampled_from(pools[0]), st.sampled_from(pools[1])), min_size=1, max_size=30
    )
)


@given(upsert_sequences)
def test_camera_rows_created_once_never_relabelled(ops: list[tuple[str, str]]) -> None:
    """Feature: nab-sentry, Property 6: Camera rows are created once and never relabelled."""
    logger = logging.getLogger(STORE_LOGGER)
    handler = _ListHandler()
    logger.addHandler(handler)
    store = MetadataStore(":memory:")
    try:
        model: dict[str, str] = {}
        expected_conflicts: list[tuple[str, str, str]] = []
        for camera_id, label in ops:
            if camera_id not in model:
                expected = "created"
                model[camera_id] = label
            elif model[camera_id] == label:
                expected = "exists"
            else:
                expected = "label_conflict"
                expected_conflicts.append((camera_id, model[camera_id], label))

            assert store.upsert_camera(camera_id, label) == expected
            # The stored label is always the first one seen for this ID (1.10).
            assert store.camera_label(camera_id) == model[camera_id]

        # Exactly one row per distinct camera ID, holding its first label (1.9).
        assert store.cameras() == sorted(model.items())
        assert store.counts().cameras == len(model)

        # One warning per conflict naming the camera ID, stored label and new label (1.10).
        conflict_logs = [r.args for r in handler.records if "label conflict" in r.msg]
        assert conflict_logs == expected_conflicts
    finally:
        store.close()
        logger.removeHandler(handler)


def test_conflict_keeps_stored_label_and_warns(caplog) -> None:
    with MetadataStore(":memory:") as store, caplog.at_level(logging.WARNING, logger=STORE_LOGGER):
        assert store.upsert_camera("cam-1", "Front Door") == "created"
        assert store.upsert_camera("cam-1", "Front Door") == "exists"
        assert store.upsert_camera("cam-1", "Back Yard") == "label_conflict"
        assert store.cameras() == [("cam-1", "Front Door")]
        assert store.camera_label("missing") is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "cam-1" in warnings[0] and "Front Door" in warnings[0] and "Back Yard" in warnings[0]


def test_upsert_inside_transaction_rolls_back(tmp_path) -> None:
    with MetadataStore(tmp_path / "meta.db") as store:
        try:
            with store.transaction():
                assert store.upsert_camera("cam-2", "Garage") == "created"
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert store.cameras() == []
        assert store.upsert_camera("cam-2", "Garage") == "created"
