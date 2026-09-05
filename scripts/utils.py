"""Shared utilities for the jmstate experiment notebooks."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from efficient_kan import KANLinear
from sksurv.metrics import brier_score, concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv
from torch import nn

from jmstate.functions.base_hazards import Neural
from jmstate.utils import plot_mcmc_diagnostics, plot_params_history

PLOT_STYLE = {
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
}
METRIC_SPECS = (
    ("auc_ipcw", "IPCW AUC ↑", "C0"),
    ("c_index_ipcw", "IPCW C-index ↑", "C1"),
    ("brier_ipcw", "IPCW Brier score ↓", "C2"),
)
LANDMARK_QUANTILES = (0.25, 0.5, 0.75)
HORIZON_FRACTIONS = (0.2, 0.3, 0.5)


class SplineBaseline(nn.Module):
    """Evaluate an unpenalized quadratic B-spline log-hazard basis.

    The basis uses an ``efficient-kan`` quadratic spline with 10 grid points and
    no penalty. The linear base weight is frozen at zero so only the spline
    contributes to the log-hazard.
    """

    def __init__(self, max_time: float):
        """Initialize the spline basis on ``[0, max_time]``.

        Args:
            max_time (float): Upper endpoint of the baseline-hazard interval.
        """
        super().__init__()
        self.layer = KANLinear(
            1,
            1,
            grid_size=10,
            spline_order=2,
            scale_noise=0.01,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=False,
            base_activation=nn.Identity,
            grid_range=(0.0, max_time + max(1e-4, max_time * 1e-5)),
        )
        with torch.no_grad():
            self.layer.base_weight.zero_()
        self.layer.base_weight.requires_grad_(False)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Evaluate the log-hazard at ``time``.

        Args:
            time (torch.Tensor): Evaluation times of any shape.

        Returns:
            torch.Tensor: Log-hazard values with the same shape as ``time``.
        """
        return self.layer(time.reshape(-1, 1)).reshape(time.shape)


def spline_hazard(max_time: float) -> Neural:
    """Build a spline-only neural base hazard.

    Args:
        max_time (float): Upper endpoint of the baseline-hazard interval.

    Returns:
        Neural: Neural base hazard with a sojourn clock.
    """
    return Neural(SplineBaseline(max_time), clock_type="sojourn")


def write_prediction_grid(
    output_path: Path,
    landmarks: Sequence[float] | np.ndarray,
    horizons: np.ndarray,
) -> None:
    """Write the shared landmark and prediction-horizon grid to CSV.

    Args:
        output_path (Path): Destination CSV path.
        landmarks (Sequence[float] | np.ndarray): Landmark times.
        horizons (np.ndarray): Horizon matrix with one row per landmark.
    """
    landmarks = np.asarray(landmarks, dtype=float)
    horizons = np.asarray(horizons, dtype=float)
    frame = pd.DataFrame(
        {
            "landmark_index": np.repeat(np.arange(landmarks.size), horizons.shape[1]),
            "landmark": np.repeat(landmarks, horizons.shape[1]),
            "horizon": horizons.ravel(),
        }
    )
    frame.to_csv(output_path, index=False)


def plot_metric_grid(
    metrics: pd.DataFrame,
    landmarks: Sequence[float] | np.ndarray,
    output_path: Path,
    model_styles: Mapping[str, str],
) -> plt.Figure:
    """Plot cross-validated IPCW metrics for each landmark.

    Args:
        metrics (pd.DataFrame): Aggregated frame with ``model``, ``landmark``,
            ``horizon``, and ``mean_*``/``sd_*`` metric columns.
        landmarks (Sequence[float] | np.ndarray): Landmark times, one column
            per landmark in the figure.
        output_path (Path): Destination figure path.
        model_styles (Mapping[str, str]): Mapping from model names to line
            styles.

    Returns:
        plt.Figure: The Matplotlib figure.
    """
    landmarks = np.asarray(landmarks, dtype=float)
    models = sorted(metrics["model"].dropna().unique())
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(
            len(METRIC_SPECS),
            len(landmarks),
            figsize=(15, 10),
            sharex="col",
            squeeze=False,
        )
        for row, (metric, label, color) in enumerate(METRIC_SPECS):
            for column, landmark in enumerate(landmarks):
                axis = axes[row, column]
                for model_name in models:
                    subset = metrics[
                        (metrics.model == model_name)
                        & np.isclose(metrics.landmark, landmark)
                    ].sort_values("horizon")
                    x = (subset.horizon - landmark).to_numpy() * 10
                    y = subset[f"mean_{metric}"].to_numpy()
                    spread = 1.96 * subset[f"sd_{metric}"].fillna(0).to_numpy()
                    axis.plot(
                        x,
                        y,
                        label=model_name,
                        color=color,
                        linestyle=model_styles[model_name],
                    )
                    axis.fill_between(
                        x,
                        y - spread,
                        y + spread,
                        alpha=0.12,
                        color=color,
                    )
                axis.set_ylabel(label)
                axis.grid(alpha=0.25)
                if row == 0:
                    axis.set_title(f"Landmark: {landmark * 10:.1f} years")
                if row == len(METRIC_SPECS) - 1:
                    axis.set_xlabel("Prediction horizon (years)")
        axes[0, 0].legend()
        figure.tight_layout()
        figure.savefig(output_path)
    return figure


def save_fit_diagnostics(fitted: Any, output_dir: Path, prefix: str) -> None:
    """Save the optimization and MCMC diagnostic figures.

    Args:
        fitted (Any): Fitted ``MultiStateJointModel`` instance.
        output_dir (Path): Directory receiving the figures.
        prefix (str): Filename prefix for the fitted model.
    """
    with plt.rc_context(PLOT_STYLE):
        figure, _ = plot_params_history(fitted, figsize=(15, 12))
        convergence = len(fitted.params_history_) - fitted.window_size
        for axis in figure.axes:
            axis.axvline(convergence, color="gray", linestyle="--")
            axis.get_legend().remove()
        figure.savefig(output_dir / f"{prefix}-optimization.pdf")
        plt.close(figure)
        figure, _ = plot_mcmc_diagnostics(fitted)
        figure.savefig(output_dir / f"{prefix}-diagnostics.pdf")
        plt.close(figure)


def prediction_grid(
    censoring_times: Sequence[float] | np.ndarray,
    landmark_quantiles: Sequence[float] = LANDMARK_QUANTILES,
    horizon_fractions: Sequence[float] = HORIZON_FRACTIONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute landmark times and prediction horizons.

    Landmarks are quantiles of the censoring times. Each horizon is the
    landmark plus a fraction of the remaining follow-up up to the maximum
    censoring time.

    Args:
        censoring_times (Sequence[float] | np.ndarray): Observed censoring or
            last-follow-up times.
        landmark_quantiles (Sequence[float]): Quantiles used for landmark
            times. Defaults to ``LANDMARK_QUANTILES``.
        horizon_fractions (Sequence[float]): Fractions of the remaining
            follow-up used after each landmark. Defaults to
            ``HORIZON_FRACTIONS``.

    Returns:
        tuple[np.ndarray, np.ndarray]: Landmark vector and landmark-by-horizon
            matrix of absolute horizons.
    """
    censoring = np.asarray(censoring_times, dtype=float)
    landmarks = np.quantile(censoring, landmark_quantiles)
    fractions = np.asarray(horizon_fractions, dtype=float)
    horizons = landmarks[:, None] + fractions[None, :] * (
        censoring.max() - landmarks[:, None]
    )
    return landmarks, horizons


def score_survival_predictions(
    train_times: Sequence[float] | np.ndarray,
    train_events: Sequence[bool] | np.ndarray,
    test_times: Sequence[float] | np.ndarray,
    test_events: Sequence[bool] | np.ndarray,
    survival: np.ndarray,
    horizons: Sequence[float] | np.ndarray,
) -> list[dict[str, Any]]:
    """Compute KM-IPCW AUC, C-index, and Brier scores for one landmark.

    Scores use a Kaplan-Meier IPCW estimator fitted on the training subjects
    at risk at the landmark. Horizons that cannot be scored keep NaN values.

    Args:
        train_times (Sequence[float] | np.ndarray): Training event or
            censoring times relative to the landmark.
        train_events (Sequence[bool] | np.ndarray): Training event indicators.
        test_times (Sequence[float] | np.ndarray): Test event or censoring
            times relative to the landmark.
        test_events (Sequence[bool] | np.ndarray): Test event indicators.
        survival (np.ndarray): Test survival probabilities with shape
            ``(n_test, n_horizons)``.
        horizons (Sequence[float] | np.ndarray): Prediction times relative to
            the landmark.

    Returns:
        list[dict[str, Any]]: One metric record per prediction horizon.
    """
    train_times = np.asarray(train_times, dtype=float)
    train_events = np.asarray(train_events, dtype=bool)
    test_times = np.asarray(test_times, dtype=float)
    test_events = np.asarray(test_events, dtype=bool)
    survival = np.asarray(survival, dtype=float)
    horizons = np.asarray(horizons, dtype=float)

    train_y = Surv.from_arrays(event=train_events, time=train_times)
    test_y = Surv.from_arrays(event=test_events, time=test_times)
    records: list[dict[str, Any]] = []
    for index, horizon in enumerate(horizons):
        estimate = np.clip(survival[:, index], 0.0, 1.0)
        risk = 1.0 - estimate
        record: dict[str, Any] = {
            "horizon": float(horizon),
            "n_train": int(train_times.size),
            "n_test": int(test_times.size),
            "n_train_events": int(train_events.sum()),
            "n_test_events": int(((test_times <= horizon) & test_events).sum()),
            "auc_ipcw": np.nan,
            "c_index_ipcw": np.nan,
            "brier_ipcw": np.nan,
        }
        try:
            auc, _ = cumulative_dynamic_auc(
                train_y, test_y, risk[:, None], np.asarray([horizon])
            )
            record["auc_ipcw"] = float(auc[0])
            record["c_index_ipcw"] = float(
                concordance_index_ipcw(train_y, test_y, risk, tau=horizon)[0]
            )
            record["brier_ipcw"] = float(
                brier_score(train_y, test_y, estimate[:, None], np.asarray([horizon]))[
                    1
                ][0]
            )
        except ValueError:
            pass
        records.append(record)
    return records


def state_at(trajectory: Sequence[tuple[float, Any]], time: float) -> int:
    """Get the last observed state no later than ``time``.

    Args:
        trajectory (Sequence[tuple[float, Any]]): List of ``(time, state)``
            pairs in chronological order.
        time (float): Query time.

    Returns:
        int: State occupied at ``time``.
    """
    state = int(trajectory[0][1])
    for transition_time, transition_state in trajectory:
        if transition_time > time:
            break
        state = int(transition_state)
    return state


def transition_survival_data(
    trajectories: Sequence[Sequence[tuple[float, Any]]],
    censoring_times: Sequence[float] | np.ndarray,
    landmark: float,
    target_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a left-truncated binary target for reaching a state.

    Subjects are eligible when still under follow-up at the landmark and not
    yet in ``target_state``. Times are measured from the landmark.

    Args:
        trajectories (Sequence[Sequence[tuple[float, Any]]]): One trajectory
            per subject.
        censoring_times (Sequence[float] | np.ndarray): Censoring or
            last-follow-up times.
        landmark (float): Conditioning time.
        target_state (int): State whose first hitting time is the event.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple of
            ``(eligible, times, events, initial_states)``.
    """
    censoring = np.asarray(censoring_times, dtype=float)
    eligible = np.zeros(len(trajectories), dtype=bool)
    times = np.zeros(len(trajectories), dtype=float)
    events = np.zeros(len(trajectories), dtype=bool)
    initial_states = np.zeros(len(trajectories), dtype=int)
    for index, trajectory in enumerate(trajectories):
        initial_states[index] = state_at(trajectory, landmark)
        eligible[index] = (
            censoring[index] > landmark and initial_states[index] < target_state
        )
        event_time = next(
            (
                time
                for time, state in trajectory
                if time > landmark and state >= target_state
            ),
            np.inf,
        )
        events[index] = bool(np.isfinite(event_time) and event_time <= censoring[index])
        observed_time = event_time if events[index] else censoring[index]
        times[index] = max(0.0, observed_time - landmark)
    return eligible, times, events, initial_states


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
