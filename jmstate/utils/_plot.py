from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import atleast_1d
from sklearn.utils._param_validation import validate_params  # type: ignore

if TYPE_CHECKING:
    from ..model._base import MultiStateJointModel


@validate_params(
    {
        "figsize": [tuple],
    },
    prefer_skip_nested_validation=True,
)
def plot_params_history(
    model: MultiStateJointModel,
    *,
    figsize: tuple[int, int] = (10, 8),
) -> tuple[plt.Figure, np.ndarray]:  # type: ignore
    r"""Visualize the evolution of model parameters during fitting.

    This function generates a grid of subplots showing the trajectory of each model
    parameter across iterations, allowing assessment of convergence and exploration
    of the optimization process.

    Args:
        model (MultiStateJointModel): The fitted model whose parameter history is to
            be plotted.
        figsize (tuple[int, int], optional): Figure dimensions `(width, height)`.
            Defaults to `(10, 8)`.

    Raises:
        ValueError: If the model contains fewer than two recorded parameter states,
            preventing visualization.

    Returns:
        tuple[plt.Figure, np.ndarray]: A tuple containing the matplotlib `Figure`
            object and a flattened array of `Axes` objects corresponding to the
            subplots.
    """
    from ..model._base import MultiStateJointModel  # noqa: PLC0415

    validate_params(
        {
            "model": [MultiStateJointModel],
        },
        prefer_skip_nested_validation=True,
    )

    if len(model.params_history_) <= 1:
        raise ValueError("More than one parameter history is required to plot")

    # Get the names
    named_parameters_dict = dict(model.params.named_parameters())
    nsubplots = len(named_parameters_dict)
    ncols = math.ceil(math.sqrt(nsubplots))
    nrows = math.ceil(nsubplots / ncols)

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)  # type: ignore
    axes = atleast_1d(axes).ravel()

    Y = torch.stack(model.params_history_)
    i = 0
    for ax, (name, val) in zip(axes, named_parameters_dict.items(), strict=False):
        history = Y[:, i : (i := i + val.numel())]
        ax.plot(history, label=[f"{name}[{j}]" for j in range(val.numel())])
        ax.set(title=name, xlabel="Iteration", ylabel="Value")
        ax.legend()

    # Remove unused subplots
    for j in range(nsubplots, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Stochastic optimization of the parameters")  # type: ignore
    plt.tight_layout()

    return fig, axes


@validate_params(
    {
        "figsize": [tuple],
    },
    prefer_skip_nested_validation=True,
)
def plot_sampler_diagnostics(
    model: MultiStateJointModel,
    *,
    figsize: tuple[int, int] = (8, 4),
) -> tuple[plt.Figure, np.ndarray]:  # type: ignore
    r"""Visualize the evolution of sampler diagnostics during fitting.

    This function generates two subplots showing the mean acceptance rate and mean step
    size across iterations.

    Args:
        model (MultiStateJointModel): The fitted model whose sampler diagnostics are to
            be plotted.
        figsize (tuple[int, int], optional): Figure dimensions `(width, height)`.
            Defaults to `(8, 4)`.

    Raises:
        ValueError: If the model sampler is `None` or the diagnostics have fewer than
            two recorded values.

    Returns:
        tuple[plt.Figure, np.ndarray]: A tuple containing the matplotlib `Figure`
            object and a flattened array of `Axes` objects corresponding to the
            subplots.
    """
    from ..model._base import MultiStateJointModel  # noqa: PLC0415

    validate_params(
        {
            "model": [MultiStateJointModel],
        },
        prefer_skip_nested_validation=True,
    )

    if model.sampler is None:
        raise ValueError("Model sampler is None")

    if len(model.sampler.diagnostics_["mean_accept_rate"]) <= 1:
        raise ValueError(
            "Model sampler diagnostics have fewer than two recorded values."
        )

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)  # type: ignore
    axes = atleast_1d(axes).ravel()

    axes[0].plot(model.sampler.diagnostics_["mean_accept_rate"])
    axes[0].set(
        title="Mean Acceptance Rate",
        xlabel="Iteration",
        ylabel="Mean Acceptance Rate",
    )
    axes[0].axhline(
        y=model.sampler.target_accept_rate,
        color="r",
        linestyle="--",
        label="Target Acceptance Rate",
    )

    axes[1].plot(
        model.sampler.diagnostics_["mean_step_size"],
        label=[f"b[{j}]" for j in range(model.sampler.b.size(-1))],
    )
    axes[1].set(
        title="Mean Step Size (component-wise)",
        xlabel="Iteration",
        ylabel="Mean Step Size",
    )

    for ax in axes:
        ax.axvline(
            x=model.n_warmup, color="gray", linestyle="--", label="Warmup end"
        )
        ax.legend()

    plt.suptitle("Sampler diagnostics")  # type: ignore
    plt.tight_layout()

    return fig, axes
