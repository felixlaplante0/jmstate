from abc import ABC, abstractmethod
from typing import Final, NamedTuple, Protocol, TypeAlias

import torch
from torch import nn

# Type Aliases
Trajectory: TypeAlias = list[tuple[float, str]]


# Constants
LOG_TWO_PI: Final[torch.Tensor] = torch.log(torch.tensor(2.0 * torch.pi))


# User definitions
class LogBaseHazardFn(ABC, nn.Module):
    r"""Abstract base class for log base hazard functions.

    This class represents a log-transformed baseline hazard function in a multistate
    model. The log base hazard is parameterized as a `torch.nn.Module`, allowing its
    parameters to be optimized during model fitting. For default base hazards, a
    `frozen` attribute can be set to prevent optimization of the module parameters.

    Tensor input conventions:

    - `t0`: a column vector of previous transition times of shape :math:`(n, 1)`.
    - `t1`: a matrix of future time points at which the log base hazard is evaluated,
        of shape :math:`(n, m)` matching the number of rows in `t0`.

    Notes:
        The outputs are in log scale and can be directly used in likelihood
        computations for multistate models.
    """

    @abstractmethod
    def forward(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor: ...


class IndividualParametersFn(Protocol):
    r"""Protocol defining the individual parameters function.

    This function maps population-level parameters, covariates, and random effects
    to individual-specific parameters used in a multistate model.

    Tensor input conventions:

    - `fixed_effects`: fixed population-level parameters.
    - `x`: covariates matrix of shape :math:`(n, p)`.
    - `b`: random effects of shape either :math:`(n, q)` (2D) or :math:`(n, m, q)` (3D).

    Tensor output conventions:

    - Last dimension corresponds to the number of parameters :math:`l`.
    - Second-last dimension corresponds to the number of individuals :math:`n`.
    - Optional third-last dimension may be used for parallelization across MCMC
      chains (= batched processing).

    Args:
        fixed_effects (torch.Tensor): Fixed population-level parameters.
        x (torch.Tensor): Fixed covariates matrix.
        b (torch.Tensor): Random effects tensor.

    Returns:
        torch.Tensor: Individual parameters tensor of shape consistent with
            :math:`(n, l)` or :math:`(n_chains, n, l)` for parallelized
            computations.

    Examples:
        >>> indiv_params_fn = lambda fixed, x, b: fixed + b
    """

    def __call__(
        self, fixed_effects: torch.Tensor, x: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor: ...


class RegressionFn(Protocol):
    r"""Protocol defining a regression function for multistate models.

    This function maps evaluation times and individual-specific parameters to predicted
    response values. It must support both 1D and 2D time inputs and individual
    parameters of order 2 or 3, returning either 3D or 4D tensors depending on the
    model design.

    Tensor input conventions:

    - `t` represents the measurement times. It can be either:
        - a 2D tensor of shape :math:`(n, m)` when each individual has
          individual-specific time points,
        - a 1D tensor of shape :math:`(m,)` when all individuals share the same
          measurement times.
    - `indiv_params` represents the individual-specific parameters. It is expected
      to have shape :math:`(n, l)` or :math:`(n_chains, n, l)` where :math:`n`
      is the number of individuals and :math:`l` is the number of parameters.

    Tensor output conventions:

    - Last dimension corresponds to the response variable dimension :math:`d`.
    - Second-last dimension corresponds to repeated measurements :math:`m`.
    - Third-last dimension corresponds to individual index :math:`n`.
    - Optional fourth-last dimension may be used for parallelization across MCMC
      chains (= batched processing).

    This protocol is conceptually identical to `LinkFn`.

    Args:
        t (torch.Tensor): Evaluation times of shape :math:`(n, m)` or :math:`(m,)`.
        indiv_params (torch.Tensor): Individual parameters of shape 2D :math:`(n, l)`
            or 3D :math:`(n_chains, n, l)`.

    Returns:
        torch.Tensor: Predicted response values of shape consistent with `(n, m, d)` or
            `(n_chains, n, m, d)` for parallelized computations.

    Examples:
        >>> def sigmoid(t: torch.Tensor, indiv_params: torch.Tensor):
        ...     scale, offset, slope = indiv_params.chunk(3, dim=-1)
        ...     # Fully broadcasted computation
        ...     return (scale * torch.sigmoid((t - offset) / slope)).unsqueeze(-1)
    """

    def __call__(self, t: torch.Tensor, indiv_params: torch.Tensor) -> torch.Tensor: ...


class LinkFn(Protocol):
    r"""Protocol defining a link function for multistate models.

    A link function maps evaluation times and individual-specific parameters to
    transformed outputs, such as transition-specific parameters. Requirements are
    identical to those of `RegressionFn`.

    Tensor input conventions:

    - `t` represents the measurement times. It can be either:
        - a 2D tensor of shape :math:`(n, m)` when each individual has
          individual-specific time points,
        - a 1D tensor of shape :math:`(m,)` when all individuals share the same
          measurement times.
    - `indiv_params` represents the individual-specific parameters. It is expected
      to have shape :math:`(n, l)` or :math:`(n_chains, n, l)` where :math:`n`
      is the number of individuals and :math:`l` is the number of parameters.

    Tensor output conventions:

    - Last dimension corresponds to the response variable dimension :math:`d`.
    - Second-last dimension corresponds to repeated measurements :math:`m`.
    - Third-last dimension corresponds to individual index :math:`n`.
    - Optional fourth-last dimension may be used for parallelization across MCMC
      chains (= batched processing).

    This protocol is conceptually identical to `RegressionFn`.

    Args:
        t (torch.Tensor): Evaluation times of shape :math:`(n, m)` or :math:`(m,)`.
        indiv_params (torch.Tensor): Individual parameters of shape 2D :math:`(n, l)`
            or 3D :math:`(n_chains, n, l)`.

    Returns:
        torch.Tensor: Transformed outputs consistent with shapes `(n, m, d)` or
            `(n_chains, n, m, d)` for parallelized computations.

    Examples:
        >>> def sigmoid(t: torch.Tensor, indiv_params: torch.Tensor):
        ...     scale, offset, slope = indiv_params.chunk(3, dim=-1)
        ...     # Fully broadcasted computation
        ...     return (scale * torch.sigmoid((t - offset) / slope)).unsqueeze(-1)
    """

    def __call__(self, t: torch.Tensor, indiv_params: torch.Tensor) -> torch.Tensor: ...


# Named tuples
class BucketData(NamedTuple):
    r"""NamedTuple representing a set of transitions for visualization purposes.

    This structure stores the transition times of individuals grouped together,
    typically used to visualize the trajectories per transition type in multistate
    models. Each entry corresponds to a single transition for a specific individual.

    Attributes:
        idxs (torch.Tensor): Indices of individuals corresponding to the transitions in
            this bucket, shape :math:`(k,)`, where :math:`k` is the number of
            transitions in the bucket.
        t0 (torch.Tensor): Column vector of previous transition times, shape
            :math:`(k, 1)`. Represents the start time of each transition.
        t1 (torch.Tensor): Column vector of next transition times, shape
            :math:`(k, 1)`. Represents the end time of each transition.

    Notes:
        Each tensor is aligned such that `t0[i]` and `t1[i]` correspond to the
        transition of individual `idxs[i]`. This alignment facilitates plotting or
        analyzing transitions per type across individuals.
    """

    idxs: torch.Tensor
    t0: torch.Tensor
    t1: torch.Tensor
