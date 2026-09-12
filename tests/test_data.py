"""Tests for model data and transition buckets."""

import pytest
import torch

from jmstate.types import ModelData, SampleData
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
    with pytest.raises(ValueError, match="censoring"):
        build_remaining_buckets(_model(), [[(0.0, 1)]], torch.tensor([[1.0], [2.0]]))


def test_bucket_labels():
    buckets = build_buckets([[(0.0, "a"), (1.0, 2)]])
    assert ("a", 2) in buckets


def test_checks():
    with pytest.raises(ValueError, match="empty"):
        check_trajectories([[]], None)
    with pytest.raises(ValueError, match="censoring"):
        check_trajectories([[(0.0, 1), (2.0, 2)]], torch.tensor([[1.0]]))


def test_checks_float_rounding():
    # A transition at the censoring time may overshoot by a few ULPs.
    limit = 1.8677299
    check_trajectories([[(0.0, 1), (limit + 1e-7, 2)]], torch.tensor([[limit]]))
    with pytest.raises(ValueError, match="censoring"):
        check_trajectories([[(0.0, 1), (limit + 1e-3, 2)]], torch.tensor([[limit]]))


def test_nan_times():
    with pytest.raises(ValueError, match="NaN"):
        ModelData(
            torch.zeros(1, 1),
            torch.tensor([[float("nan")]]),
            torch.tensor([[[0.1]]]),
            [[(0.0, 1)]],
            torch.tensor([[2.0]]),
        )


def test_sample_data():
    data = _data()
    sample = SampleData(data.x, data.trajectories, torch.ones(1, 3), data.c)
    assert sample[0].x.shape == (1, 1)
    assert sample.to(torch.float64).x.dtype == torch.float64


def test_preparation():
    data, model = _data(), _model()
    prepared = ModelDataUnchecked(
        data.x, data.t, data.y, data.trajectories, data.c
    ).prepare(model)
    assert prepared.valid_y.shape == data.y.shape
