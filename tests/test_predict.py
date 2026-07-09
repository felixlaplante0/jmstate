"""Tests for prediction methods."""

import torch

from jmstate.types._data import SampleData

from ._helpers import _data, _model


def test_longitudinal():
    model, data = _model(), _data()
    model.fit(data)
    assert model.predict_y(data, torch.tensor([[1.0, 1.5]]), n_samples=1).shape == (
        1,
        1,
        2,
        1,
    )


def test_survival():
    model, data = _model(), _data()
    model.fit(data)
    assert model.predict_surv_logps(
        data, torch.tensor([[1.6, 2.0]]), n_samples=1
    ).shape == (1, 1, 2)


def test_trajectories():
    model, data = _model(), _data()
    model.fit(data)
    assert (
        len(
            model.predict_trajectories(
                data, torch.tensor([[2.0]]), n_samples=1, max_length=2
            )
        )
        == 1
    )


def test_sampling():
    model = _model()
    sample = SampleData(
        torch.zeros(1, 1), [[(0.0, 1)]], torch.ones(1, 3), torch.tensor([[0.5]])
    )
    assert model.compute_surv_logps(sample, torch.tensor([[1.0, 2.0]])).shape == (1, 2)
    assert (
        len(model.sample_trajectories(sample, torch.tensor([[2.0]]), max_length=1)) == 1
    )
