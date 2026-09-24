"""Simulation study for the jmstate paper.

Four candidate models are fitted to synthetic longitudinal and multistate data
of increasing sample size: a correctly specified model, an under-specified one,
an over-specified one, and a model with a misspecified link. Tables are built
by ``tables.py``; the study loop itself runs inline in
``scripts/fitting-test.ipynb`` so partial results can be checkpointed.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any
from warnings import warn

import torch
from torch.distributions import MultivariateNormal
from torch.nn.utils import parameters_to_vector
from tqdm import trange

try:
    from .utils import resolve_device
except ImportError:  # run as a script: python scripts/utils/simulation.py
    from utils import resolve_device

from jmstate import MultiStateJointModel
from jmstate.functions.base_hazards import Exponential
from jmstate.types import (
    ModelData,
    ModelDesign,
    ModelParameters,
    PrecisionParameters,
    SampleData,
)

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"

N_VALUES = (100, 500, 2000)
N_REPS = 100
N_TIMES = 20
SEED = 42
COVERAGE_LEVEL = 0.95
KEYS = ((1, 1), (1, 2))


def pk_fn(t: torch.Tensor, indiv_params: torch.Tensor) -> torch.Tensor:
    """Evaluate the bi-exponential pharmacokinetic concentration.

    Args:
        t (torch.Tensor): Evaluation times.
        indiv_params (torch.Tensor): Individual parameters ``(A, k, ka)``.

    Returns:
        torch.Tensor: Concentrations with a trailing singleton dimension.
    """
    amplitude, k, ka = indiv_params.chunk(3, dim=-1)
    concentration = amplitude * (torch.exp(-k * t) - torch.exp(-ka * t))
    return concentration.unsqueeze(-1)


def pk_integral_fn(t: torch.Tensor, indiv_params: torch.Tensor) -> torch.Tensor:
    """Evaluate the cumulative pharmacokinetic exposure.

    Args:
        t (torch.Tensor): Evaluation times.
        indiv_params (torch.Tensor): Individual parameters ``(A, k, ka)``.

    Returns:
        torch.Tensor: Cumulative exposures with a trailing singleton dimension.
    """
    amplitude, k, ka = indiv_params.chunk(3, dim=-1)
    integral = amplitude * (
        (1.0 / k) * (1 - torch.exp(-k * t)) - (1.0 / ka) * (1 - torch.exp(-ka * t))
    )
    return integral.unsqueeze(-1)


def indiv_params_fn(
    fixed: torch.Tensor,
    x: torch.Tensor,  # noqa: ARG001
    b: torch.Tensor,
) -> torch.Tensor:
    """Map fixed effects and random effects to individual parameters.

    Args:
        fixed (torch.Tensor): Fixed effects.
        x (torch.Tensor): Covariates (used for broadcasting only).
        b (torch.Tensor): Log-random effects.

    Returns:
        torch.Tensor: Individual parameters.
    """
    return fixed * b.exp()


def gen_data(n: int, m: int, model: MultiStateJointModel):
    """Sample synthetic longitudinal and multistate data.

    Args:
        n (int): Number of individuals.
        m (int): Number of longitudinal time points.
        model (MultiStateJointModel): Model used for sampling.

    Returns:
        tuple: ``(x, t, y, trajectories, c)`` model inputs.
    """
    censoring = torch.rand(n, 1) * 5 + 10
    x = torch.randn(n, 4)

    random_prec = model.params.random_prec.precision.detach()
    noise_prec = model.params.noise_prec.precision.detach()
    random_dist = MultivariateNormal(
        torch.zeros(random_prec.size(0)), precision_matrix=random_prec
    )
    noise_dist = MultivariateNormal(
        torch.zeros(noise_prec.size(0)), precision_matrix=noise_prec
    )

    b = random_dist.sample((n,))
    indiv_params = model.design.indiv_params_fn(
        model.params.fixed_effects.detach(), x, b
    )

    sample_data = SampleData(x[:, [0]], [[(0.0, 1)] for _ in range(n)], indiv_params)
    trajectories = model.sample_trajectories(sample_data, censoring)

    t = torch.linspace(0, 15, m)
    y = model.design.regression_fn(t, indiv_params)
    y += noise_dist.sample(y.shape[:2])
    y[t > censoring] = torch.nan

    return x, t, y, trajectories, censoring


def true_parameters() -> ModelParameters:
    """Build the parameters of the data-generating model.

    Returns:
        ModelParameters: True simulation parameters.
    """
    return ModelParameters(
        torch.tensor([2.0, 0.2, 1.0]),
        PrecisionParameters.from_covariance(
            torch.diag(torch.tensor([0.15, 0.05, 0.1])), "diag"
        ),
        PrecisionParameters.from_covariance(torch.tensor([[0.05]]), "spherical"),
        {(1, 1): Exponential(2e-1), (1, 2): Exponential(1e-2)},
        {(1, 1): torch.tensor([-1.0]), (1, 2): torch.tensor([0.5])},
        {(1, 1): torch.tensor([-1.0]), (1, 2): torch.tensor([0.5])},
    )


def correct_design() -> ModelDesign:
    """Build the correctly specified model design.

    Returns:
        ModelDesign: Design whose link functions integrate the exposure.
    """
    return ModelDesign(indiv_params_fn, pk_fn, dict.fromkeys(KEYS, pk_integral_fn))


def misspecified_design() -> ModelDesign:
    """Build the design with a misspecified link function.

    Returns:
        ModelDesign: Design linking on the concentration instead of its integral.
    """
    return ModelDesign(indiv_params_fn, pk_fn, dict.fromkeys(KEYS, pk_fn))


def _init_params(
    link_coefs: dict[tuple[int, int], torch.Tensor] | None = None, n_x: int = 1
) -> ModelParameters:
    """Build initial parameters for the candidate models.

    Args:
        link_coefs (dict[tuple[int, int], torch.Tensor] | None, optional): Link
            coefficients per transition. Defaults to None (independent zeros).
        n_x (int, optional): Number of covariate coefficients per transition.
            Defaults to 1.

    Returns:
        ModelParameters: Initial parameters.
    """
    return ModelParameters(
        torch.ones(3),
        PrecisionParameters.from_covariance(torch.eye(3), "diag"),
        PrecisionParameters.from_covariance(torch.eye(1), "spherical"),
        {key: Exponential(1.0) for key in KEYS},
        link_coefs or {key: torch.zeros(1) for key in KEYS},
        {key: torch.zeros(n_x) for key in KEYS},
    )


def init_params_correct() -> ModelParameters:
    """Build correctly specified initial parameters.

    Returns:
        ModelParameters: Initial parameters with the true structure.
    """
    return _init_params()


def init_params_less() -> ModelParameters:
    """Build initial parameters with a shared link coefficient.

    Returns:
        ModelParameters: Initial parameters with too few link parameters.
    """
    return _init_params(dict.fromkeys(KEYS, torch.nn.Parameter(torch.zeros(1))))


def init_params_more() -> ModelParameters:
    """Build initial parameters with extra covariate coefficients.

    Returns:
        ModelParameters: Initial parameters with too many coefficients.
    """
    return _init_params(n_x=4)


TRUE_PARAMETERS = true_parameters()
DESIGN = correct_design()
DESIGN_MIS = misspecified_design()
TRUE_MODEL = MultiStateJointModel(DESIGN, TRUE_PARAMETERS)
MODELS: dict[str, tuple[Callable[[], ModelParameters], ModelDesign]] = {
    "correct": (init_params_correct, DESIGN),
    "less": (init_params_less, DESIGN),
    "more": (init_params_more, DESIGN),
    "mis": (init_params_correct, DESIGN_MIS),
}


def get_vector_and_scores(
    parameters_factory: Callable[[], ModelParameters],
    design: ModelDesign,
    data: ModelData,
    device: torch.device,
):
    """Fit one candidate model and return its parameter vector and scores.

    Args:
        parameters_factory (callable): Factory returning fresh parameters.
        design (ModelDesign): Model design to fit.
        data (ModelData): Training data.
        device (torch.device): Device used for fitting.

    Returns:
        tuple: ``(parameter vector, standard errors, AIC, BIC, fit time,
            summary time)``. Failed fits return NaN entries (and ``None`` for
            BIC) instead of raising, so one divergent replication cannot abort
            a long study.
    """
    parameters = parameters_factory()
    optimizer = torch.optim.Adam(parameters.parameters(), lr=0.05)
    model = MultiStateJointModel(
        design, parameters, optimizer, max_iter=10000, verbose=False
    ).to(device)

    try:
        start = perf_counter()
        model.fit(data)
        fit_time = perf_counter() - start

        start = perf_counter()
        model.compute_summary()
        summary_time = perf_counter() - start

        vector = parameters_to_vector(model.parameters()).detach().cpu()
        try:
            stderr = model.stderr.detach().cpu()
        except (RuntimeError, ValueError):
            stderr = torch.full_like(vector, float("nan"))
        return (
            vector,
            stderr,
            model.aic_,
            model.bic_,
            fit_time,
            summary_time,
        )
    except Exception as exc:  # one bad fit must not kill the study
        warn(f"Fit failed and is recorded as NaN: {exc!r}", stacklevel=2)
        nan_vector = torch.full((parameters.numel(),), float("nan"))
        nan = float("nan")
        return nan_vector, nan_vector.clone(), nan, None, nan, nan


def run_replications(
    n: int,
    n_reps: int = N_REPS,
    device: torch.device | str | None = None,
    seed: int = SEED,
) -> dict[str, list[dict[str, Any]]]:
    """Fit every candidate model on ``n_reps`` synthetic samples of size ``n``.

    Each replication draws one dataset shared by all candidate models.

    Args:
        n (int): Number of individuals per replication.
        n_reps (int): Number of replications. Defaults to ``N_REPS``.
        device (torch.device | str | None): Target device; auto-selected when None.
        seed (int): Random seed. Defaults to ``SEED``.

    Returns:
        dict[str, list[dict[str, Any]]]: One record list per candidate model.
    """
    device = resolve_device(device)
    torch.manual_seed(seed)

    results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _ in trange(n_reps, desc=f"n={n}", leave=False):
        data_more = ModelData(*gen_data(n, N_TIMES, TRUE_MODEL))
        data = replace(data_more, x=data_more.x[:, [0]])

        for name, (factory, design) in MODELS.items():
            vector, stderr, aic, bic, fit_time, summary_time = get_vector_and_scores(
                factory, design, data_more if name == "more" else data, device
            )
            results[name].append(
                {
                    "vec": vector,
                    "se": stderr,
                    "aic": aic,
                    "bic": bic,
                    "fit_time": fit_time,
                    "summary_time": summary_time,
                }
            )
    return results
