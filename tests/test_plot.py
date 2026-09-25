"""Tests for plotting utilities."""

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

from jmstate.utils import plot_mcmc_diagnostics, plot_params_history

from ._helpers import _data, _model

matplotlib.use("Agg")


def test_params_history():
    model, data = _model(max_iter=2, tol=0.0), _data()
    model.n_subsample = 1
    model.to(torch.float64)
    model.fit(data)
    figure, axes = plot_params_history(model)
    assert len(axes) >= 1
    plt.close(figure)


def test_mcmc_diagnostics():
    model, data = _model(max_iter=2, tol=0.0), _data()
    model.n_subsample = 1
    model.fit(data)
    figure, axes = plot_mcmc_diagnostics(model)
    assert len(axes) == 2
    plt.close(figure)


def test_plot_errors():
    model = _model()
    with pytest.raises(ValueError, match="recorded parameter"):
        plot_params_history(model)
    with pytest.raises(ValueError, match="sampler"):
        plot_mcmc_diagnostics(model)

    model.fit(_data())
    with pytest.raises(ValueError, match="MCMC step"):
        plot_mcmc_diagnostics(model)


def test_plot_unused_subplots():
    model = _model()
    del model.params.x_coefs["(1, 2)"]
    vector = model.params.to_vector()
    model.params_history_ = [vector, vector]
    figure, axes = plot_params_history(model)
    assert len(axes) == 6
    plt.close(figure)
