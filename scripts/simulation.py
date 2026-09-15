"""Simulation study for the jmstate paper.

Four candidate models are fitted to synthetic longitudinal and multistate data
of increasing sample size: a correctly specified model, an under-specified one,
an over-specified one, and a model with a misspecified link. Two tables are
written to ``results/``:

- ``convergence-results.csv``: bias, RMSE and empirical coverage of the
  correctly specified model, each reported with its Monte Carlo standard error;
- ``selection-results.csv``: AIC/BIC selection frequencies and fitting times.

Both tables carry the sample size ``n`` so that they can be rebuilt later. Run
the whole study with::

    python scripts/simulation.py --n 100 500 2000 --reps 100
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.distributions import MultivariateNormal
from torch.nn.utils import parameters_to_vector
from tqdm import trange
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
from jmstate.utils import confidence_interval

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

N_VALUES = (100, 500, 2000)
N_REPS = 100
N_TIMES = 20
MAX_ITER = 2000
LEARNING_RATE = 0.1
SEED = 42
COVERAGE_LEVEL = 0.95


# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
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
    fixed: torch.Tensor, x: torch.Tensor, b: torch.Tensor
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


# --------------------------------------------------------------------------- #
# Model specification
# --------------------------------------------------------------------------- #
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
    surv_fns = {(1, 1): pk_integral_fn, (1, 2): pk_integral_fn}
    return ModelDesign(indiv_params_fn, pk_fn, surv_fns)


def misspecified_design() -> ModelDesign:
    """Build the design with a misspecified link function.

    Returns:
        ModelDesign: Design linking on the concentration instead of its integral.
    """
    surv_fns = {(1, 1): pk_integral_fn, (1, 2): pk_integral_fn}
    return ModelDesign(indiv_params_fn, pk_fn, dict.fromkeys(surv_fns, pk_fn))


def init_params_correct() -> ModelParameters:
    """Build correctly specified initial parameters.

    Returns:
        ModelParameters: Initial parameters with the true structure.
    """
    return ModelParameters(
        torch.ones(3),
        PrecisionParameters.from_covariance(torch.eye(3), "diag"),
        PrecisionParameters.from_covariance(torch.eye(1), "spherical"),
        {(1, 1): Exponential(1.0), (1, 2): Exponential(1.0)},
        {key: torch.zeros(1) for key in ((1, 1), (1, 2))},
        {key: torch.zeros(1) for key in ((1, 1), (1, 2))},
    )


def init_params_less() -> ModelParameters:
    """Build initial parameters with a shared link coefficient.

    Returns:
        ModelParameters: Initial parameters with too few link parameters.
    """
    shared_coef = torch.nn.Parameter(torch.zeros(1))
    return ModelParameters(
        torch.ones(3),
        PrecisionParameters.from_covariance(torch.eye(3), "diag"),
        PrecisionParameters.from_covariance(torch.eye(1), "spherical"),
        {(1, 1): Exponential(1.0), (1, 2): Exponential(1.0)},
        {(1, 1): shared_coef, (1, 2): shared_coef},
        {key: torch.zeros(1) for key in ((1, 1), (1, 2))},
    )


def init_params_more() -> ModelParameters:
    """Build initial parameters with extra covariate coefficients.

    Returns:
        ModelParameters: Initial parameters with too many coefficients.
    """
    return ModelParameters(
        torch.ones(3),
        PrecisionParameters.from_covariance(torch.eye(3), "diag"),
        PrecisionParameters.from_covariance(torch.eye(1), "spherical"),
        {(1, 1): Exponential(1.0), (1, 2): Exponential(1.0)},
        {key: torch.zeros(1) for key in ((1, 1), (1, 2))},
        {key: torch.zeros(4) for key in ((1, 1), (1, 2))},
    )


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


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
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
            summary time)``.
    """
    parameters = parameters_factory()
    optimizer = torch.optim.Adam(parameters.parameters(), lr=LEARNING_RATE)
    model = MultiStateJointModel(
        design, parameters, optimizer, max_iter=MAX_ITER, verbose=False
    ).to(device)

    start = perf_counter()
    model.fit(data)
    fit_time = perf_counter() - start

    start = perf_counter()
    model.compute_summary()
    summary_time = perf_counter() - start

    return (
        parameters_to_vector(model.parameters()).detach().cpu(),
        model.stderr.detach().cpu(),
        model.aic_,
        model.bic_,
        fit_time,
        summary_time,
    )


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


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def convergence_table(results: dict[str, list[dict[str, Any]]], n: int) -> pd.DataFrame:
    """Compute bias, RMSE and coverage with Monte Carlo standard errors.

    The Monte Carlo standard error of a statistic is its empirical standard
    deviation divided by the square root of the number of replications. For the
    RMSE it is obtained from the MSE by the delta method and for the coverage
    proportion it is ``sqrt(p (1 - p) / R)``.

    Args:
        results (dict[str, list[dict[str, Any]]]): Output of ``run_replications``.
        n (int): Sample size, stored as a column.

    Returns:
        pd.DataFrame: One row per parameter.
    """
    vectors = torch.stack([record["vec"] for record in results["correct"]])
    stderrs = torch.stack([record["se"] for record in results["correct"]])
    truth = parameters_to_vector(TRUE_PARAMETERS.parameters()).detach()
    errors = (vectors - truth).numpy()
    n_reps = errors.shape[0]

    bias = errors.mean(axis=0)
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    bias_se = errors.std(axis=0, ddof=1) / np.sqrt(n_reps)
    mse_se = (errors**2).std(axis=0, ddof=1) / np.sqrt(n_reps)
    rmse_se = mse_se / (2.0 * rmse)

    lower, upper = confidence_interval(vectors, stderrs, level=COVERAGE_LEVEL)
    available = torch.isfinite(stderrs)
    covered = (truth >= lower) & (truth <= upper) & available
    successes = covered.sum(dim=0).numpy()
    totals = available.sum(dim=0).numpy()
    coverage = np.divide(
        successes,
        totals,
        out=np.full(successes.shape, np.nan),
        where=totals > 0,
    )
    coverage_mcse = np.full(coverage.shape, np.nan)
    covered_totals = totals > 0
    coverage_mcse[covered_totals] = np.sqrt(
        coverage[covered_totals]
        * (1.0 - coverage[covered_totals])
        / totals[covered_totals]
    )

    names = [
        f"{name}[{j}]"
        for name, param in TRUE_PARAMETERS.named_parameters()
        for j in range(param.numel())
    ]
    return pd.DataFrame(
        {
            "n": n,
            "parameter": names,
            "True value": truth.numpy(),
            "Bias": bias,
            "Bias sd": bias_se,
            "RMSE": rmse,
            "RMSE sd": rmse_se,
            "Coverage": coverage,
            "Coverage mcse": coverage_mcse,
        }
    )


def selection_table(results: dict[str, list[dict[str, Any]]], n: int) -> pd.DataFrame:
    """Count AIC/BIC wins and summarise fitting times per candidate.

    Args:
        results (dict[str, list[dict[str, Any]]]): Output of ``run_replications``.
        n (int): Sample size, stored as a column.

    Returns:
        pd.DataFrame: One row per candidate model.
    """
    names = list(MODELS)
    aic = pd.DataFrame({name: [r["aic"] for r in results[name]] for name in names})
    bic = pd.DataFrame({name: [r["bic"] for r in results[name]] for name in names})
    fit_times = pd.DataFrame(
        {name: [r["fit_time"] for r in results[name]] for name in names}
    )
    summary_times = pd.DataFrame(
        {name: [r["summary_time"] for r in results[name]] for name in names}
    )

    counts = pd.DataFrame(
        {
            "AIC": aic.idxmin(axis=1).value_counts(),
            "BIC": bic.idxmin(axis=1).value_counts(),
        }
    )
    counts = counts.reindex(names, fill_value=0).fillna(0).astype(int)
    counts.index.name = "model"
    counts["Fit mean (s)"] = fit_times.mean()
    counts["Fit sd (s)"] = fit_times.std()
    counts["Summary mean (s)"] = summary_times.mean()
    counts["Summary sd (s)"] = summary_times.std()

    counts.insert(0, "n_reps", len(aic))
    counts.insert(0, "n", n)
    return counts.reset_index()


def run_study(
    n_values: Sequence[int] = N_VALUES,
    n_reps: int = N_REPS,
    device: torch.device | str | None = None,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the study for every sample size and write both result tables.

    Args:
        n_values (Sequence[int]): Sample sizes to simulate. Defaults to ``N_VALUES``.
        n_reps (int): Number of replications per sample size. Defaults to ``N_REPS``.
        device (torch.device | str | None): Target device; auto-selected when None.
        seed (int): Random seed. Defaults to ``SEED``.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The convergence and selection tables.
    """
    device = resolve_device(device)
    print(f"Using device: {device}")

    convergence, selection = [], []
    for n in n_values:
        results = run_replications(n, n_reps, device, seed)
        convergence.append(convergence_table(results, n))
        selection.append(selection_table(results, n))

    convergence_df = pd.concat(convergence, ignore_index=True)
    selection_df = pd.concat(selection, ignore_index=True)
    convergence_df.to_csv(RESULTS / "convergence-results.csv", index=False)
    selection_df.to_csv(RESULTS / "selection-results.csv", index=False)
    return convergence_df, selection_df


def main() -> None:
    """Parse command-line arguments and run the study."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=list(N_VALUES))
    parser.add_argument("--reps", type=int, default=N_REPS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    convergence, selection = run_study(args.n, args.reps, args.device, args.seed)
    print(convergence)
    print(selection)


if __name__ == "__main__":
    main()
