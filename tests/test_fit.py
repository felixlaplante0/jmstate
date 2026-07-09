"""Tests for model fitting."""

import warnings

import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._param_validation import InvalidParameterError

from ._helpers import _data, _model


def test_warning():
    with pytest.warns(ConvergenceWarning, match="max_iter"):
        _model(max_iter=1, window_size=3).fit(_data())


def test_optimizer():
    with pytest.raises(ValueError, match="Optimizer is not initialized"):
        _model(fit=False).fit(_data())


def test_parameters():
    with pytest.raises(InvalidParameterError, match="max_iter"):
        _model(max_iter=-1).fit(_data())


def test_convergence():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _model().fit(_data())
    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, ConvergenceWarning)
    ]
