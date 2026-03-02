from numbers import Integral
from typing import Any, Self, cast

import torch
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils._param_validation import (  # type: ignore
    Interval,  # type: ignore
    StrOptions,  # type: ignore
    validate_params,  # type: ignore
)
from sklearn.utils.validation import assert_all_finite  # type: ignore
from torch import nn

from ..types._defs import LogBaseHazardFn
from ..utils._checks import check_matrix_dim
from ..utils._linalg import flat_from_log_cholesky, log_cholesky_from_flat


class PrecisionParameters(BaseEstimator, nn.Module):
    r"""`nn.Module` encapsulating precision matrix parameters.

    This class provides three types of precision matrix parametrization: full,
    diagonal, and spherical (scalar). The default is full matrix parametrization.
    Precision matrices are internally represented using the **log-Cholesky
    parametrization** of the inverse covariance (precision) matrix. Formally, let
    :math:`P = \Sigma^{-1}` be the precision matrix and :math:`L` its Cholesky factor
    with positive diagonal elements. The log-Cholesky representation :math:`\tilde{L}`
    is defined by:

    .. math::
        \tilde{L}_{ij} = L_{ij}, \quad i > j

    .. math::
        \tilde{L}_{ii} = \log L_{ii}.

    This representation ensures numerical stability and avoids explicit inversion when
    computing quadratic forms. The log determinant of the precision matrix is then

    .. math::
        \log \det P = 2 \operatorname{Tr}(\tilde{L}).

    Instances can be created from a precision matrix using the `from_precision` or from
    a covariance matrix using the `from_covariance` classmethod with `precision_type`
    set to `'full'`, `'diag'`, or `'spherical'`.

    Attributes:
        flat (torch.Tensor): Flat representation of the precision matrix suitable
            for optimization.
        dim (int): Dimension of the precision matrix.
        precision_type (str): Type of parametrization, one of `'full'`, `'diag'`,
            or `'spherical'`.

    Examples:
        >>> random_prec = PrecisionParameters.from_covariance(torch.eye(3), "diag")
        >>> noise_prec = PrecisionParameters.from_covariance(torch.eye(2), "spherical")
    """

    @classmethod
    @validate_params(
        {
            "P": [torch.Tensor],
            "precision_type": [StrOptions({"full", "diag", "spherical"})],
        },
        prefer_skip_nested_validation=True,
    )
    def from_precision(cls, P: torch.Tensor, precision_type: str = "full") -> Self:
        r"""Gets instance from precision matrix according to choice of precision type.

        Args:
            P (torch.Tensor): The square precision matrix.
            precision_type (str, optional): The method, `'full'`, `'diag'`, or
                `'spherical'`. Defaults to `'full'`.

        Returns:
            Self: The usable representation.
        """
        L = cast(torch.Tensor, torch.linalg.cholesky(P))  # type: ignore
        L.diagonal().log_()
        return cls(flat_from_log_cholesky(L, precision_type), L.size(0), precision_type)

    @classmethod
    @validate_params(
        {
            "V": [torch.Tensor],
            "precision_type": [StrOptions({"full", "diag", "spherical"})],
        },
        prefer_skip_nested_validation=True,
    )
    def from_covariance(cls, V: torch.Tensor, precision_type: str = "full") -> Self:
        r"""Gets instance from covariance matrix according to choice of precision type.

        Args:
            V (torch.Tensor): The square covariance matrix.
            precision_type (str, optional): The method, `'full'`, `'diag'`, or
                `'spherical'`. Defaults to `'full'`.

        Returns:
            Self: The usable representation.
        """
        return cls.from_precision(V.inverse(), precision_type)

    @validate_params(
        {
            "flat": [torch.Tensor],
            "dim": [Interval(Integral, 1, None, closed="left")],
            "precision_type": [StrOptions({"full", "diag", "spherical"})],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(self, flat: torch.Tensor, dim: int, precision_type: str):
        """Initializes the `PrecisionParameters` object.

        Args:
            flat (torch.Tensor): The flat representation of the precision matrix.
            dim (int): The dimension of the precision matrix.
            precision_type (str): The method used to parametrize the precision matrix.

        Raises:
            ValueError: If the representation is invalid.
        """
        super().__init__()  # type: ignore

        check_matrix_dim(flat, dim, precision_type)

        self.flat = nn.Parameter(flat)
        self.dim = dim
        self.precision_type = precision_type

    @property
    def precision(self) -> torch.Tensor:
        """Gets the precision matrix.

        Returns:
            torch.Tensor: The precision matrix.
        """
        L = log_cholesky_from_flat(self.flat, self.dim, self.precision_type)
        L.diagonal().exp_()
        return L @ L.T

    @property
    def covariance(self) -> torch.Tensor:
        """Gets the covariance matrix.

        Returns:
            torch.Tensor: The covariance matrix.
        """
        L = log_cholesky_from_flat(self.flat, self.dim, self.precision_type)
        L.diagonal().exp_()
        return torch.cholesky_inverse(L)

    @property
    def _prec_cholesky_and_log_eigvals(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets Cholesky factor of precision matrix and its log eigvals.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Precision matrix and log eigvals.
        """
        L = log_cholesky_from_flat(self.flat, self.dim, self.precision_type)
        log_eigvals = 2 * L.diag()
        L.diagonal().exp_()

        return L, log_eigvals


class ModelParameters(BaseEstimator, nn.Module):
    r"""`nn.Module` encapsulating all model parameters for a multistate joint model.

    This module contains fixed population-level parameters, covariate effects, link
    coefficients, random effects, noise precision, and log base hazard functions.
    Parameters can be shared by assigning the same `nn.Parameter` object to multiple
    fields. Reusing tensors directly is not supported and requires wrapping in
    `nn.Parameter` for correct computations, as `nn.ParameterDict` would break the
    object ids otherwise.

    Fixed population-level parameters:
        - `fixed_effects` are fixed population-level parameters.

    Random effects and noise precision matrices:
        - `random_prec` and `noise_prec` are `PrecisionParameters` objects
          representing the random effects and residual noise precision matrices
          respectively.

    Base hazard functions:
        - `base_hazards` is a dictionary of `LogBaseHazardFn` modules keyed by
          `(from_state, to_state)` tuples for each transition; optimization can be
          disabled per hazard via its `frozen` attribute.

    Link and covariate coefficients:
        - `link_coefs` and `x_coefs` are dictionaries keyed by `(from_state,
          to_state)` tuples, representing linear coefficients for links and
          covariates, respectively.

    Attributes:
        fixed_effects (torch.Tensor): Fixed population-level parameters.
        random_prec (PrecisionParameters): Precision parameters for random effects.
        noise_prec (PrecisionParameters): Precision parameters for residual noise.
        base_hazards (nn.ModuleDict): Log base hazard functions per transition.
        link_coefs (nn.ParameterDict): Linear parameters for the link functions.
        x_coefs (nn.ParameterDict): Covariate parameters for each transition.

    Examples:
        >>> fixed_effects = torch.zeros(3)
        >>> random_prec = PrecisionParameters.from_covariance(torch.eye(3), "diag")
        >>> noise_prec = PrecisionParameters.from_covariance(torch.eye(2), "spherical")
        >>> link_coefs = {(0, 1): torch.zeros(3), (1, 0): torch.zeros(3)}
        >>> x_coefs = {(0, 1): torch.zeros(2), (1, 0): torch.zeros(2)}
        >>> params = ModelParameters(
        ...     fixed_effects,
        ...     random_prec,
        ...     noise_prec,
        ...     link_coefs,
        ...     x_coefs,
        ... )
        >>> # Shared parameters
        >>> shared_coef = nn.Parameter(torch.zeros(3))  # Mandatory nn.Parameter
        >>> shared_link_coefs = {(0, 1): shared_coef, (1, 0): shared_coef}
        >>> shared_params = ModelParameters(
        ...     fixed_effects,
        ...     random_prec,
        ...     noise_prec,
        ...     shared_link_coefs,
        ...     x_coefs,
        ... )
    """

    fixed_effects: torch.Tensor
    random_prec: PrecisionParameters
    noise_prec: PrecisionParameters
    base_hazards: nn.ModuleDict
    link_coefs: nn.ParameterDict
    x_coefs: nn.ParameterDict

    @validate_params(
        {
            "fixed_effects": [torch.Tensor],
            "random_prec": [PrecisionParameters],
            "noise_prec": [PrecisionParameters],
            "base_hazards": [dict],
            "link_coefs": [dict],
            "x_coefs": [dict],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        fixed_effects: torch.Tensor,
        random_prec: PrecisionParameters,
        noise_prec: PrecisionParameters,
        base_hazards: dict[tuple[Any, Any], LogBaseHazardFn],
        link_coefs: dict[tuple[Any, Any], torch.Tensor],
        x_coefs: dict[tuple[Any, Any], torch.Tensor],
    ):
        """Initializes the `ModelParams` object.

        Args:
            fixed_effects (torch.Tensor): Fixed population-level parameters.
            random_prec (PrecisionParameters): Precision parameters for random effects.
            noise_prec (PrecisionParameters): Precision parameters for residual noise.
            base_hazards (dict[tuple[Any, Any], LogBaseHazardFn]): Log base hazard
                functions.
            link_coefs (dict[tuple[Any, Any], torch.Tensor]): Linear parameters for the
                link functions.
            x_coefs (dict[tuple[Any, Any], torch.Tensor]): Covariate parameters for each
                transition.

        Raises:
            ValueError: If any of the tensors contains NaN or infinite values.
        """
        super().__init__()  # type: ignore

        self.fixed_effects = nn.Parameter(fixed_effects)
        self.random_prec = random_prec
        self.noise_prec = noise_prec
        self.base_hazards = nn.ModuleDict({str(k): v for k, v in base_hazards.items()})
        self.link_coefs = nn.ParameterDict({str(k): v for k, v in link_coefs.items()})
        self.x_coefs = nn.ParameterDict({str(k): v for k, v in x_coefs.items()})

        for key, val in self.named_parameters():
            assert_all_finite(val.detach(), input_name=key)

    def numel(self) -> int:
        """Return the number of unique parameters.

        Returns:
            int: The number of the (unique) parameters.
        """
        return sum(p.numel() for p in self.parameters())
