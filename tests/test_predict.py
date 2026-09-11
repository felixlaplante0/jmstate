"""Tests for prediction methods."""

import pytest
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


def test_double_monte_carlo():
    torch.manual_seed(0)
    model, data = _model(), _data()
    model.n_subsample = 1
    model.fit(data)
    model.compute_summary(
        n_posterior_samples=4,
        n_importance_samples=4,
        importance_batch_size=2,
    )

    assert (
        model.predict_y(
            data, torch.tensor([[1.0, 1.5]]), n_samples=1, double_monte_carlo=True
        ).shape[0]
        == 1
    )
    assert (
        model.predict_surv_logps(
            data, torch.tensor([[1.6, 2.0]]), n_samples=1, double_monte_carlo=True
        ).shape[0]
        == 1
    )
    assert (
        len(
            model.predict_trajectories(
                data,
                torch.tensor([[2.0]]),
                n_samples=1,
                max_length=2,
                double_monte_carlo=True,
            )
        )
        == 1
    )


def test_double_monte_carlo_requires_summary():
    model, data = _model(), _data()
    with pytest.raises(ValueError, match="summary"):
        model.predict_y(
            data, torch.tensor([[1.0]]), n_samples=1, double_monte_carlo=True
        )


def test_singular_params():
    model = _model()
    model.fim_ = torch.zeros(model.params.numel(), model.params.numel())
    assert model._sample_params(2).shape == (2, model.params.numel())
