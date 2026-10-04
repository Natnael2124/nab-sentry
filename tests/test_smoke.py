"""Smoke tests for the scaffold: package import, workspace fixture, shared strategies."""

import string

from hypothesis import given

import nab_sentry
from tests import strategies as S


def test_package_imports_without_heavy_deps():
    assert nab_sentry.__version__


def test_workspace_fixture_lays_out_data_and_models(workspace):
    assert workspace.data.is_dir() and workspace.data.name == "data"
    assert workspace.models.is_dir() and workspace.models.name == "models"
    for sub in (workspace.thumbs, workspace.playback, workspace.logs):
        assert sub.is_dir() and sub.parent == workspace.data


@given(S.camera_ids)
def test_camera_ids_are_valid(cid):
    assert 1 <= len(cid) <= 64
    assert set(cid) <= set(string.ascii_letters + string.digits + "-")


@given(S.labels)
def test_labels_are_valid(label):
    assert 1 <= len(label) <= 128 and label.strip()


@given(S.naive_datetimes, S.aware_datetimes)
def test_datetimes_are_whole_second(naive, aware):
    assert naive.tzinfo is None and naive.microsecond == 0
    assert aware.utcoffset() is not None and aware.microsecond == 0
