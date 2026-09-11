"""Tests for dtype and device handling."""

import pytest
import torch

from jmstate.functions.base_hazards import Exponential
from jmstate.types import SampleData
from jmstate.types._data import ModelDataUnchecked
from jmstate.utils._checks import check_finite
from jmstate.utils._dtype import canonical_dtype_device, resolve_dtype
from jmstate.utils._surv import (
    build_buckets_raw,
    build_quad_buckets_raw,
    build_remaining_buckets_raw,
)

from ._helpers import _data, _model


def test_extension():
    trajs = [
        [(0.0, 1), (1.5, 2)],
        [(0.0, 1), (0.5, 3), (1.2, 2)],
        [(0.0, 3)],
        [(0.0, 1), (2.0, 1), (2.5, 2)],
    ]
    keys = [(1, 2), (1, 3), (3, 2)]
    censoring = [2.0, 2.0, 1.0, 2.0]
    assert set(build_buckets_raw(trajs)) == {(1, 1), (1, 2), (1, 3), (3, 2)}
    assert set(build_quad_buckets_raw(trajs, keys, censoring)) == {
        (1, 2),
        (1, 3),
        (3, 2),
    }
    assert set(build_remaining_buckets_raw(trajs, keys, censoring)) == {(3, 2)}
    with pytest.raises(ValueError, match="empty"):
        build_buckets_raw([[]])


def test_dtypes():
    assert resolve_dtype(torch.float32, torch.bfloat16) == torch.float32
    assert resolve_dtype(torch.bfloat16) == torch.float32
    assert resolve_dtype(torch.float32, torch.float64) == torch.float64
    assert resolve_dtype(torch.float64) == torch.float64


def test_canonical():
    model = _model()
    assert model.dtype == torch.float32
    assert model.device == torch.device("cpu")
    assert canonical_dtype_device(model.params) == (
        torch.float32,
        torch.device("cpu"),
    )


def test_canonical_fallback():
    assert canonical_dtype_device(torch.nn.Module()) == (
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
    assert prepared.x.dtype == torch.float32
    assert prepared.quad_buckets[(1, 2)][1].dtype == torch.float32
    assert prepared.quad_buckets[(1, 2)][4].dtype == torch.float32


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


@pytest.mark.skipif(not torch.xpu.is_available(), reason="no XPU device")
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
