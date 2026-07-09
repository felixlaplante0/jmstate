"""Tests for baseline hazards."""

import torch

from jmstate.functions.base_hazards import Exponential, Gompertz, LogNormal, Weibull


def test_hazards():
    t0, t1 = torch.tensor([[0.5]]), torch.tensor([[1.0, 2.0]])
    torch.testing.assert_close(Exponential(2.0)(t0, t1), torch.log(torch.tensor(2.0)))
    for hazard in (Weibull(2.0, 1.5), Gompertz(1.2, 0.2), LogNormal(0.0, 1.0)):
        assert hazard(t0, t1).shape == t1.shape
    for hazard in (
        Weibull(2.0, 1.5, clock_type="absolute", frozen=True),
        Gompertz(1.2, 0.2, clock_type="absolute", frozen=True),
        LogNormal(0.0, 1.0, clock_type="absolute", frozen=True),
    ):
        assert hazard(t0, t1).shape == t1.shape
