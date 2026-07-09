"""Private fixtures for jmstate tests."""

import torch

from jmstate.functions.base_hazards import Exponential
from jmstate.model import MultiStateJointModel
from jmstate.types import ModelData, ModelDesign, ModelParameters, PrecisionParameters


def _indiv_effects(
    fixed: torch.Tensor,
    _x: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return fixed * torch.exp(b)


def _marker(t: torch.Tensor, indiv_params: torch.Tensor) -> torch.Tensor:
    a, k, ka = indiv_params.chunk(3, dim=-1)
    return (a * (torch.exp(-k * t) - torch.exp(-ka * t))).unsqueeze(-1)


def _marker_integral(t: torch.Tensor, indiv_params: torch.Tensor) -> torch.Tensor:
    a, k, ka = indiv_params.chunk(3, dim=-1)
    return (a * (1 - torch.exp(-k * t)) / k - (1 - torch.exp(-ka * t)) / ka).unsqueeze(
        -1
    )


def _model(
    *, max_iter: int = 1, window_size: int = 2, tol: float = 1.0, fit: bool = True
) -> MultiStateJointModel:
    design = ModelDesign(_indiv_effects, _marker, {(1, 2): _marker_integral})
    params = ModelParameters(
        torch.tensor([1.0, 0.5, 1.0]),
        PrecisionParameters.from_covariance(torch.eye(3), "diag"),
        PrecisionParameters.from_covariance(torch.eye(1), "spherical"),
        {(1, 2): Exponential(1.0)},
        {(1, 2): torch.zeros(1)},
        {(1, 2): torch.zeros(1)},
    )
    optimizer = torch.optim.SGD(params.parameters(), lr=0.0) if fit else None
    return MultiStateJointModel(
        design,
        params,
        optimizer,
        n_chains=1,
        n_warmup=0,
        n_subsample=0,
        max_iter=max_iter,
        window_size=window_size,
        tol=tol,
        verbose=False,
    )


def _data() -> ModelData:
    return ModelData(
        torch.zeros(1, 1),
        torch.tensor([[0.5, 1.0]]),
        torch.tensor([[[0.1], [0.2]]]),
        [[(0.0, 1), (1.5, 2)]],
        torch.tensor([[2.0]]),
    )
