from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

import torch
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils._param_validation import validate_params  # type: ignore
from sklearn.utils.validation import (  # type: ignore
    assert_all_finite,  # type: ignore
    check_consistent_length,  # type: ignore
)

from ..utils._checks import check_trajectories
from ..utils._surv import build_all_buckets
from ._defs import IndividualParametersFn, LinkFn, RegressionFn, Trajectory

if TYPE_CHECKING:
    from ..model._fit import FitMixin
    from ..model._predict import PredictMixin


# Dataclasses
@dataclass
class ModelDesign(BaseEstimator):
    r"""Dataclass encapsulating the design of a multistate joint model.

    This class defines the parametric and functional design of the model, including
    individual-specific parameters, regression structures, and link functions for state
    transitions. All functions should be implemented to allow maximum broadcasting to
    ensure efficient vectorized computation. If broadcasting is not possible, `vmap` may
    be used for safe parallelization.

    Functions provided to the MCMC sampler will automatically be wrapped with
    `torch.no_grad()`. If gradient computation is required regardless of the sampling
    context, wrap the function explicitly with `torch.enable_grad()`.

    All functions must be well-defined on closed intervals of their input domain and
    differentiable almost everywhere to ensure compatibility with gradient-based
    procedures.

    Individual Parameters:
        - `indiv_params_fn` is a function that computes individual parameters. Given
          `fixed_effects` (fixed population-level parameters), `x` (covariates matrix of
          shape :math:`(n, p)`), and `b` (random effects, either 2D or 3D), it yields
          tensors of corresponding dimensions, either 2D or 3D depending on the model
          design. This function defines the mapping from population-level parameters and
          covariates to individual-specific parameters.

    Regression:
        - `regression_fn` is a function that maps time points and individual parameters
          to the expected observations. It must accept 1D or 2D time inputs and 2D or 3D
          individual parameters. The output tensor must have at least three dimensions:
          the last dimension corresponds to the response variable, the second-last to
          repeated measurements, the third-last to individuals, and an optional
          fourth-last dimension for parallelization across MCMC chains.

    Link Functions:
        - `link_fns` is a mapping from transition keys (tuples of
          `(state_from, state_to)`) to link functions. Each link function shares the
          same requirements as `regression_fn` and defines the transformation from
          regression outputs to transition-specific parameters.

    Attributes:
        indiv_params_fn (IndividualParametersFn): Function that computes individual
            parameters. Given `fixed_effects` (fixed population-level parameters), `x`
            (covariates matrix of shape :math:`(n, p)`), and `b` (random effects, either
            2D or 3D), it yields tensors of corresponding dimensions, either 2D or 3D
            depending on the model design. This function defines the mapping from
            population-level parameters and covariates to individual-specific
            parameters.
        regression_fn (RegressionFn): Regression function mapping time points and
            individual parameters to the expected observations. It must accept 1D or 2D
            time inputs and 2D or 3D individual parameters. The output tensor must have
            at least three dimensions: the last dimension corresponds to the response
            variable, the second-last to repeated measurements, the third-last to
            individuals, and an optional fourth-last dimension for parallelization
            across MCMC chains.
        link_fns (Mapping[tuple[Any, Any], LinkFn]): Mapping from transition keys
            (tuples of `(state_from, state_to)`) to link functions. Each link function
            shares the same requirements as `regression_fn` and defines the
            transformation from regression outputs to transition-specific parameters.

    Examples:
        >>> def sigmoid(t: torch.Tensor, indiv_params: torch.Tensor):
        ...     scale, offset, slope = indiv_params.chunk(3, dim=-1)
        ...     # Fully broadcasted
        ...     return (scale * torch.sigmoid((t - offset) / slope)).unsqueeze(-1)
        >>> fixed_plus_b = lambda fixed, x, b: fixed + b
        >>> link_fns = {("alive", "dead"): sigmoid}
        >>> design = ModelDesign(fixed_plus_b, sigmoid, link_fns)
    """

    indiv_params_fn: IndividualParametersFn
    regression_fn: RegressionFn
    link_fns: Mapping[tuple[Any, Any], LinkFn]


@dataclass
class ModelData(BaseEstimator):
    r"""Dataclass containing learnable multistate joint model data.

    Covariates:
        - `x` is a matrix of covariates of shape :math:`(n, p)`, where :math:`n` denotes
          the number of individuals and :math:`p` the number of covariates.

    Measurement Times:
        - `t` represents the measurement times. It can be either:
            - a 2D tensor of shape :math:`(n, m)` when each individual has
              individual-specific time points,
            - a 1D tensor of shape :math:`(m,)` when all individuals share the same
              measurement times.

    Padding with NaNs is optional. However, `t` must not contain NaN values at positions
    where `y` is observed (i.e., where `y` is not NaN).

    Observations:
        - `y` is expected to be a 3D tensor of shape :math:`(n, m, d)`, where :math:`n`
          is the number of individuals, :math:`m` is the maximum number of measurements
          per individual, and :math:`d` is the dimension of the observation space
          :math:`\mathbb{R}^d`. Padding is performed with NaNs.

    Trajectories:
        - `trajectories` contains the individual-level multistate trajectories. They
          correspond to a `list[list[tuple[float, Any]]]` where each inner list is a
          trajectory and each tuple is a `(time, state)` pair.

    Censoring Times:
        - `c` represents the right censoring times, and are expected to be a column
          vector of shape :math:`(n, 1)`. Each value must be greater than or equal to
          the corresponding maximum transition time for each individual.

    The data can be completed using the `prepare` method, which formats it for manual
    likelihood evaluation and MCMC procedures. This usage is intended for advanced users
    familiar with the codebase. The method is called automatically by the `fit` and
    `predict` routines and does not require explicit user intervention in standard
    workflows.

    Raises:
        ValueError: If some trajectory is empty.
        ValueError: If some trajectory is not sorted.
        ValueError: If some trajectory is not compatible with the censoring times.
        ValueError: If any of the inputs contain inf or NaN values except `y`.
        ValueError: If the size is not consistent between inputs.

    Attributes:
        x (torch.Tensor): Fixed covariate matrix of shape `(n, p)`, where `n` is the
            number of individuals and `p` the number of covariates.
        t (torch.Tensor): Measurement times. Either a 1D tensor of shape `(m,)` when
            times are shared across individuals, or a 2D tensor of shape `(n, m)`
            when individuals have distinct time grids. Padding with NaNs may be
            used when required.
        y (torch.Tensor): Longitudinal measurements of shape `(n, m, d)`, where `n` is
            the number of individuals, `m` the maximum number of measurements per
            individual, and `d` the observation dimension. Padding is performed
            with NaNs when necessary.
        trajectories (list[Trajectory]): List of individual trajectories. Each
            `Trajectory` consists of a sequence of `(time, state)` tuples.
        c (torch.Tensor): Censoring times provided as a column vector. Each value must
            be greater than or equal to the corresponding maximum trajectory time.
        valid_mask (torch.Tensor): Boolean mask indicating valid (non-padded)
            measurements.
        n_valid (torch.Tensor): Number of valid measurements per individual.
        valid_t (torch.Tensor): Filtered tensor containing only valid measurement
            times.
        valid_y (torch.Tensor): Filtered tensor containing only valid measurements.
        buckets (dict[tuple[Any, Any], tuple[torch.Tensor, ...]]): Grouped trajectory
            data structures used for likelihood computation.
    """

    x: torch.Tensor
    t: torch.Tensor
    y: torch.Tensor
    trajectories: list[Trajectory]
    c: torch.Tensor

    def __len__(self) -> int:
        """Gets the number of individuals.

        Returns:
            int: The number of individuals.
        """
        return len(self.trajectories)

    def __post_init__(self):
        """Runs the post init conversions.

        Raises:
            ValueError: If some trajectory is empty.
            ValueError: If some trajectory is not sorted.
            ValueError: If some trajectory is not compatible with the censoring times.
            ValueError: If any of the inputs contain NaN or infinite values except `t`
                and `y` for which NaN values are allowed.
            ValueError: If the size is not consistent between inputs.
        """
        validate_params(
            {
                "x": [torch.Tensor],
                "t": [torch.Tensor],
                "y": [torch.Tensor],
                "trajectories": [list],
                "c": [torch.Tensor],
            },
            prefer_skip_nested_validation=True,
        )

        check_trajectories(self.trajectories, self.c)

        assert_all_finite(self.x, input_name="x")
        assert_all_finite(self.t, input_name="t", allow_nan=True)
        assert_all_finite(self.y, input_name="y", allow_nan=True)
        assert_all_finite(self.c, input_name="c")

        check_consistent_length(self.x, self.y, self.c, self.trajectories)
        torch.broadcast_to(self.t, self.y.shape[:-1])

        # Check NaNs between t and y
        if ((~self.y.isnan()).any(dim=-1) & self.t.isnan()).any():
            raise ValueError("NaN time values on non NaN y values are not allowed")


@dataclass
class ModelDataUnchecked(ModelData):
    """Unchecked model data class."""

    valid_mask: torch.Tensor = field(init=False)
    n_valid: torch.Tensor = field(init=False)
    valid_t: torch.Tensor = field(init=False)
    valid_y: torch.Tensor = field(init=False)
    buckets: dict[tuple[Any, Any], tuple[torch.Tensor, ...]] = field(init=False)

    def __post_init__(self):
        """Overrides to skip checks."""
        pass

    def prepare(self, model: FitMixin | PredictMixin) -> Self:
        """Sets the representation for likelihood computations according to model.

        Args:
            model (MCMCMixin): The multistate joint model.

        Returns:
            Self: The prepared (completed) data.
        """

        def quad_fn(
            t0: torch.Tensor, t1: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            half = 0.5 * (t1 - t0)
            return half, torch.cat(
                [t1, (t0 + t1).addmm(half, model._std_nodes, beta=0.5)],  # type: ignore
                dim=-1,
            )

        self.valid_mask = ~self.y.isnan()
        self.n_valid = self.valid_mask.sum(dim=-2).to(self.y.dtype)
        self.valid_t = self.t.nan_to_num(self.t.nanmean().item())
        self.valid_y = self.y.nan_to_num()
        self.buckets = build_all_buckets(
            self.trajectories, self.c, tuple(model.design.link_fns.keys())
        )
        self.quads = {
            key: quad_fn(t0, t1) for key, (_, t0, t1, _) in self.buckets.items()
        }

        return self


@dataclass
class SampleData(BaseEstimator):
    r"""Dataclass containing individual-level data for sampling procedures.

    Covariates:
        - `x` is a matrix of covariates of shape :math:`(n, p)`, where :math:`n` denotes
          the number of individuals and :math:`p` the number of covariates.

    Trajectories:
        - `trajectories` contains the individual-level multistate trajectories. They
          correspond to a `list[list[tuple[float, Any]]]` where each inner list is a
          trajectory and each tuple is a `(time, state)` pair.

    Individual Parameters:
        - `indiv_params` represents the individual-specific parameters. It is expected
          to have the same number of rows as there are trajectories. Use a 3D tensor
          only if you fully understand the codebase and mechanisms. Trajectory sampling
          may only be used with matrices.

    Truncation Times:
        - `t_trunc` corresponds to truncation or conditioning times for each individual.
          This attribute is optional and, if not provided, is set to the maximum
          observation time per individual.

    The data class is used for simulation and for prediction of quantities related to
    survival functions or trajectories. Unlike `ModelData`, it assumes exact knowledge
    of the individual parameters.

    Raises:
        ValueError: If some trajectory is empty.
        ValueError: If some trajectory is not sorted.
        ValueError: If some trajectory is not compatible with the censoring times.
        ValueError: If any of the inputs contain NaN or infinite values.
        ValueError: If the size is not consistent between inputs.

    Attributes:
        x (torch.Tensor): Fixed covariate matrix of shape `(n, p)`, where `n` is the
            number of individuals and `p` the number of covariates.
        trajectories (list[Trajectory]): List of individual trajectories. Each
            `Trajectory` consists of a sequence of `(time, state)` tuples.
        indiv_params (torch.Tensor): Individual parameters with the same number of
            rows as there are trajectories. Use a matrix by default.
        t_trunc (torch.Tensor | None): Optional truncation times per individual. If
            None, the maximum observation time is used.
    """

    x: torch.Tensor
    trajectories: list[Trajectory]
    indiv_params: torch.Tensor
    t_trunc: torch.Tensor | None = None

    def __len__(self) -> int:
        """Gets the number of individuals.

        Returns:
            int: The number of individuals.
        """
        return len(self.trajectories)

    def __post_init__(self):
        """Runs the post init conversions and checks.

        Raises:
            ValueError: If some trajectory is empty.
            ValueError: If some trajectory is not sorted.
            ValueError: If some trajectory is not compatible with the truncation times.
            ValueError: If any of the inputs contain inf or NaN values.
            ValueError: If the size is not consistent between inputs.
        """
        validate_params(
            {
                "x": [torch.Tensor],
                "trajectories": [list],
                "indiv_params": [torch.Tensor],
                "t_trunc": [torch.Tensor, None],
            },
            prefer_skip_nested_validation=True,
        )

        check_trajectories(self.trajectories, self.t_trunc)

        assert_all_finite(self.x, input_name="x")
        assert_all_finite(self.indiv_params, input_name="indiv_params")
        assert_all_finite(self.t_trunc, input_name="t_trunc")

        check_consistent_length(
            self.x,
            self.indiv_params.transpose(0, -2),
            self.t_trunc,
            self.trajectories,
        )


@dataclass
class SampleDataUnchecked(SampleData):
    """Unchecked sample data class."""

    def __post_init__(self):
        """Overrides to skip checks."""
        pass
