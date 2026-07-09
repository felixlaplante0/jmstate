"""Tests for MCMC sampling."""

import pytest
from sklearn.exceptions import ConvergenceWarning

from jmstate.types._data import ModelDataUnchecked

from ._helpers import _data, _model


def test_step():
    model, data = _model(tol=0.0), _data()
    prepared = ModelDataUnchecked(
        data.x, data.t, data.y, data.trajectories, data.c
    ).prepare(model)
    sampler = model._init_sampler(prepared)
    sampler.step().run(1)
    assert len(sampler.diagnostics_["mean_accept_rate"]) == 2


def test_fit_loop():
    model, data = _model(tol=0.0), _data()
    prepared = ModelDataUnchecked(
        data.x, data.t, data.y, data.trajectories, data.c
    ).prepare(model)
    with pytest.warns(ConvergenceWarning, match="max_iter"):
        model._fit(prepared, model._init_sampler(prepared))
