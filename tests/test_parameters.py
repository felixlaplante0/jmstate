"""Tests for parameter containers."""

import pytest
import torch

from jmstate.types import PrecisionParameters


def test_precision():
    full = PrecisionParameters.from_covariance(torch.eye(2), "full")
    diag = PrecisionParameters.from_precision(torch.eye(2), "diag")
    spherical = PrecisionParameters.from_covariance(torch.eye(2) * 4.0, "spherical")
    torch.testing.assert_close(full.precision, torch.eye(2))
    torch.testing.assert_close(diag.covariance, torch.eye(2))
    torch.testing.assert_close(spherical.precision, torch.eye(2) * 0.25)
    torch.testing.assert_close(spherical.covariance, torch.eye(2) * 4.0)
    assert full.get_params()["dim"] == 2
    with pytest.raises(ValueError, match="incompatible"):
        PrecisionParameters(torch.ones(2), 2, "full")
