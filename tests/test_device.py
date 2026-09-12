"""Tests for dtype and device handling."""

import pytest
import torch

from jmstate.functions.base_hazards import Exponential
from jmstate.types import SampleData
from jmstate.types._data import ModelDataUnchecked
from jmstate.utils import _surv as surv
from jmstate.utils._checks import check_finite
from jmstate.utils._dtype import dtype_device
from jmstate.utils._surv import build_buckets
from jmstate.utils._surv_ext import (
    _build_buckets,
    _build_quad_buckets,
    _build_remaining_buckets,
)

from ._helpers import _data, _model

HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()


def test_extension():
    trajs = [
        [(0.0, 1), (1.5, 2)],
        [(0.0, 1), (0.5, 3), (1.2, 2)],
        [(0.0, 3)],
        [(0.0, 1), (2.0, 1), (2.5, 2)],
    ]
    keys = [(1, 2), (1, 3), (3, 2)]
    censoring = [2.0, 2.0, 1.0, 2.0]
    assert set(_build_buckets(trajs)) == {(1, 1), (1, 2), (1, 3), (3, 2)}
    assert set(_build_quad_buckets(trajs, keys, censoring)) == {
        (1, 2),
        (1, 3),
        (3, 2),
    }
    assert set(_build_remaining_buckets(trajs, keys, censoring)) == {(3, 2)}
    with pytest.raises(ValueError, match="empty"):
        _build_buckets([[]])


def test_numpy_bridge_fallback(monkeypatch):
    """Falls back to list conversion when torch cannot read NumPy arrays."""
    def _unavailable(*_args, **_kwargs):
        raise RuntimeError("Numpy is not available")

    monkeypatch.setattr(surv.torch, "from_numpy", _unavailable)
    buckets = build_buckets([[(0.0, 1), (1.5, 2)], [(0.0, 1), (2.0, 2)]])
    assert set(buckets) == {(1, 2)}
    data = buckets[(1, 2)]
    assert data.idxs.tolist() == [0, 1]
    assert data.t0.flatten().tolist() == [0.0, 0.0]
    assert data.t1.flatten().tolist() == [1.5, 2.0]


def test_canonical():
    model = _model()
    assert model.dtype == torch.float32
    assert model.device == torch.device("cpu")
    assert dtype_device(model.params) == (
        torch.float32,
        torch.device("cpu"),
    )


def test_canonical_fallback():
    assert dtype_device(torch.nn.Module()) == (
        torch.get_default_dtype(),
        torch.device("cpu"),
    )


def test_double():
    model = _model(max_iter=1).to(torch.float64)
    assert model.dtype == torch.float64
    data = _data()
    model.fit(data)
    prepared = ModelDataUnchecked(
        data.x, data.t, data.y, data.trajectories, data.c
    ).prepare(model)
    assert prepared.x.dtype == torch.float64
    assert prepared.quad_buckets[(1, 2)][1].dtype == torch.float64


def test_bfloat16():
    model = _model(max_iter=1).to(torch.bfloat16)
    assert model.dtype == torch.bfloat16
    data = _data()
    model.fit(data)
    prepared = ModelDataUnchecked(
        data.x, data.t, data.y, data.trajectories, data.c
    ).prepare(model)
    assert prepared.x.dtype == torch.bfloat16
    assert prepared.quad_buckets[(1, 2)][1].dtype == torch.bfloat16
    assert prepared.quad_buckets[(1, 2)][4].dtype == torch.bfloat16


def test_frozen():
    hazard = Exponential(1.0, frozen=True).to(torch.bfloat16)
    assert hazard.log_lmda.dtype == torch.bfloat16


def test_to():
    data = _data().to(dtype=torch.float64)
    assert data.x.dtype == torch.float64
    assert data[0].x.shape == (1, 1)


def test_finite():
    check_finite(torch.zeros(2), "x")
    check_finite(None, "x")
    check_finite(torch.tensor([float("nan")]), "x", allow_nan=True)
    with pytest.raises(ValueError, match="infinite"):
        check_finite(torch.tensor([float("inf")]), "x")
    with pytest.raises(ValueError, match="NaN"):
        check_finite(torch.tensor([float("nan")]), "x")


def test_chains():
    model, data = _model(), _data()
    sample = SampleData(data.x, data.trajectories, torch.ones(2, 1, 3), data.c)
    out = model.sample_trajectories(sample, torch.tensor([[2.0]]), max_length=1)
    assert len(out) == 2
    assert len(out[0]) == 1


@pytest.mark.skipif(not HAS_XPU, reason="no XPU device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_xpu(dtype):
    model = _model(max_iter=1)
    model.n_subsample = 1
    model.to(device="xpu", dtype=dtype)
    assert model.device == torch.device("xpu:0")
    model.fit(_data())
    data = _data()
    sample = SampleData(data.x, data.trajectories, torch.ones(1, 3), data.c)
    surv = model.compute_surv_logps(sample, torch.tensor([[1.0, 2.0]]))
    assert surv.device == torch.device("xpu:0")
    assert model.sample_trajectories(sample, torch.tensor([[2.0]]), max_length=1)
