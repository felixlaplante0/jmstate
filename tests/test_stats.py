"""Tests for confidence interval utilities."""

import torch
from torch.nn.utils import parameters_to_vector

from jmstate.utils import confidence_interval

from ._helpers import _model

QUANTILE_95 = 1.959963984540054
QUANTILE_90 = 1.6448536269514722


def test_confidence_interval():
    estimate = torch.tensor([0.0, 1.0, -2.0])
    stderr = torch.tensor([1.0, 0.5, 0.1])
    lower, upper = confidence_interval(estimate, stderr)
    torch.testing.assert_close(lower, estimate - QUANTILE_95 * stderr)
    torch.testing.assert_close(upper, estimate + QUANTILE_95 * stderr)


def test_confidence_interval_level():
    estimate = torch.zeros(2)
    stderr = torch.ones(2)
    lower, upper = confidence_interval(estimate, stderr, level=0.9)
    torch.testing.assert_close(lower, torch.full((2,), -QUANTILE_90))
    torch.testing.assert_close(upper, torch.full((2,), QUANTILE_90))


def test_conf_int():
    model = _model()
    model.fim_ = torch.eye(sum(p.numel() for p in model.params.parameters()))
    lower, upper = model.conf_int()
    vector = parameters_to_vector(model.params.parameters())
    torch.testing.assert_close(lower, vector - QUANTILE_95)
    torch.testing.assert_close(upper, vector + QUANTILE_95)
