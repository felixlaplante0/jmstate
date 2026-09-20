"""Result tables for the jmstate simulation study.

The study loop itself runs inline in ``scripts/fitting-test.ipynb`` (with a
CSV checkpoint after each sample size); this module only builds the tables.
"""

from collections.abc import Sequence
from typing import Any
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import parameters_to_vector

from jmstate.types import ModelParameters
from jmstate.utils import confidence_interval


def convergence_table(
    results: dict[str, list[dict[str, Any]]],
    n: int,
    parameters: ModelParameters,
    level: float,
) -> pd.DataFrame:
    """Compute bias, RMSE and coverage with Monte Carlo standard errors.

    The Monte Carlo standard error of a statistic is its empirical standard
    deviation divided by the square root of the number of replications. For the
    RMSE it is obtained from the MSE by the delta method and for the coverage
    proportion it is ``sqrt(p (1 - p) / R)``.

    Args:
        results (dict[str, list[dict[str, Any]]]): Output of ``run_replications``.
        n (int): Sample size, stored as a column.
        parameters (ModelParameters): True parameters used for simulation.
        level (float): Coverage level of the confidence intervals.

    Returns:
        pd.DataFrame: One row per parameter.
    """
    vectors = torch.stack([record["vec"] for record in results["correct"]])
    stderrs = torch.stack([record["se"] for record in results["correct"]])
    truth = parameters_to_vector(parameters.parameters()).detach()
    errors = (vectors - truth).numpy()
    # Failed replications are stored as NaN; use NaN-aware statistics so one
    # bad fit cannot poison the whole table.
    with catch_warnings():
        simplefilter("ignore", RuntimeWarning)
        valid = np.count_nonzero(~np.isnan(errors), axis=0)
        bias = np.nanmean(errors, axis=0)
        mse = np.nanmean(errors**2, axis=0)
        rmse = np.sqrt(mse)
        bias_se = np.nanstd(errors, axis=0, ddof=1) / np.sqrt(valid)
        mse_se = np.nanstd(errors**2, axis=0, ddof=1) / np.sqrt(valid)
    rmse_se = np.divide(
        mse_se,
        2.0 * rmse,
        out=np.full(mse_se.shape, np.nan),
        where=rmse > 0,
    )

    lower, upper = confidence_interval(vectors, stderrs, level=level)
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
        for name, param in parameters.named_parameters()
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


def selection_table(
    results: dict[str, list[dict[str, Any]]], n: int, names: Sequence[str]
) -> pd.DataFrame:
    """Count AIC/BIC wins and summarise fitting times per candidate.

    Args:
        results (dict[str, list[dict[str, Any]]]): Output of ``run_replications``.
        n (int): Sample size, stored as a column.
        names (Sequence[str]): Candidate model names, in display order.

    Returns:
        pd.DataFrame: One row per candidate model.
    """
    names = list(names)
    # Failed fits are stored as NaN/None; coerce to NaN so they are skipped by
    # idxmin instead of raising.
    aic = pd.DataFrame({name: [r["aic"] for r in results[name]] for name in names})
    bic = pd.DataFrame({name: [r["bic"] for r in results[name]] for name in names})
    aic = aic.apply(pd.to_numeric, errors="coerce")
    bic = bic.apply(pd.to_numeric, errors="coerce")
    fit_times = pd.DataFrame(
        {name: [r["fit_time"] for r in results[name]] for name in names}
    )
    summary_times = pd.DataFrame(
        {name: [r["summary_time"] for r in results[name]] for name in names}
    )

    def _wins(frame: pd.DataFrame) -> pd.Series:
        winners = frame.apply(
            lambda row: row.idxmin() if row.notna().any() else None, axis=1
        )
        return winners.value_counts()

    counts = pd.DataFrame({"AIC": _wins(aic), "BIC": _wins(bic)})
    counts = counts.reindex(names, fill_value=0).fillna(0).astype(int)
    counts.index.name = "model"
    counts["Fit mean (s)"] = fit_times.mean()
    counts["Fit sd (s)"] = fit_times.std()
    counts["Summary mean (s)"] = summary_times.mean()
    counts["Summary sd (s)"] = summary_times.std()

    counts.insert(0, "n_reps", len(aic))
    counts.insert(0, "n", n)
    return counts.reset_index()


def aggregate_metrics(
    records: Sequence[dict[str, Any]], group_columns: Sequence[str]
) -> pd.DataFrame:
    """Aggregate metric records by mean, standard deviation, and valid count.

    Args:
        records (Sequence[dict[str, Any]]): Per-fold metric records.
        group_columns (Sequence[str]): Columns used for grouping.

    Returns:
        pd.DataFrame: Aggregated frame with ``mean_*``, ``sd_*``, and
            ``n_valid_*`` columns.
    """
    frame = pd.DataFrame(records)
    metric_columns = ["auc_ipcw", "c_index_ipcw", "brier_ipcw"]
    grouped = frame.groupby(list(group_columns), dropna=False)
    result = grouped.size().rename("n_records").to_frame()
    for metric in metric_columns:
        result[f"mean_{metric}"] = grouped[metric].mean()
        result[f"sd_{metric}"] = grouped[metric].std()
        result[f"n_valid_{metric}"] = grouped[metric].count()
    return result.reset_index()
