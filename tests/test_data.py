"""Tests for model data and transition buckets."""

import pytest
import torch

from jmstate.types._data import ModelDataUnchecked
from jmstate.utils._checks import check_trajectories
from jmstate.utils._surv import build_buckets, build_remaining_buckets

from ._helpers import _data, _model


def test_indexing():
    data = _data()
    assert len(data) == 1
    assert data[0].x.shape == (1, 1)


def test_trajectories():
    with pytest.raises(ValueError, match="sorted"):
        check_trajectories([[(1.0, 1), (0.0, 2)]], None)


def test_buckets():
    buckets = build_buckets([[(0.0, 1), (1.0, 2)], [(0.0, 1), (2.0, 2)]])
    torch.testing.assert_close(buckets[(1, 2)].t1, torch.tensor([[1.0], [2.0]]))
    assert (1, 2) in build_remaining_buckets(
        _model(), [[(0.0, 1)]], torch.tensor([[2.0]])
    )


def test_preparation():
    data, model = _data(), _model()
    prepared = ModelDataUnchecked(
        data.x, data.t, data.y, data.trajectories, data.c
    ).prepare(model)
    assert prepared.valid_y.shape == data.y.shape
