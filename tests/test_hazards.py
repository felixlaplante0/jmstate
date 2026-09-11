"""Tests for baseline hazards."""

import pytest
import torch

from jmstate.functions.base_hazards import (
    Exponential,
    Gompertz,
    LogNormal,
    Neural,
    Weibull,
)


@pytest.mark.parametrize(
    ("hazard", "expected"),
    [
        (Exponential(2.0), torch.tensor(0.6931471825)),
        (Weibull(2.0, 1.5), torch.tensor([[1.0986123085, 1.6479184628]])),
        (Gompertz(1.2, 0.2), torch.tensor([[0.2823216021, 0.4823216200]])),
        (LogNormal(0.0, 1.0), torch.tensor([[-0.1861602962, -0.3353189230]])),
    ],
)
def test_parametric_hazards(hazard, expected):
    t0 = torch.tensor([[0.5]])
    t1 = torch.tensor([[1.0, 2.0]])
    torch.testing.assert_close(hazard(t0, t1), expected)


@pytest.mark.parametrize(
    "hazard",
    [
        Weibull(2.0, 1.5, clock_type="absolute", frozen=True),
        Gompertz(1.2, 0.2, clock_type="absolute", frozen=True),
        LogNormal(0.0, 1.0, clock_type="absolute", frozen=True),
    ],
)
def test_time_dependent_hazard_options(hazard):
    t0 = torch.tensor([[0.5]])
    t1 = torch.tensor([[1.0, 2.0]])
    assert hazard(t0, t1).shape == t1.shape
    assert not any(parameter.requires_grad for parameter in hazard.parameters())


def test_hazard_properties():
    torch.testing.assert_close(Exponential(2.0).lmda, torch.tensor(2.0))
    weibull = Weibull(2.0, 1.5)
    torch.testing.assert_close(weibull.lmda, torch.tensor(2.0))
    torch.testing.assert_close(weibull.k, torch.tensor(1.5))
    torch.testing.assert_close(Gompertz(1.2, 0.2).a, torch.tensor(1.2))
    torch.testing.assert_close(LogNormal(0.0, 2.0).scale, torch.tensor(2.0))


def test_neural_hazard():
    network = torch.nn.Linear(1, 1)
    with torch.no_grad():
        network.weight.fill_(2.0)
        network.bias.fill_(-1.0)

    t0 = torch.tensor([[0.5], [1.0]])
    t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    hazard = Neural(network)
    torch.testing.assert_close(hazard(t0, t1), 2 * (t1 - t0) - 1)

    t1_3d = t1.unsqueeze(0).expand(3, -1, -1)
    output = Neural(network, clock_type="absolute")(t0, t1_3d)
    torch.testing.assert_close(output, 2 * t1_3d - 1)
