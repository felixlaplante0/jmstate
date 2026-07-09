"""Base hazard functions."""

__all__ = ["Exponential", "Gompertz", "LogNormal", "Weibull"]

from numbers import Real
from typing import cast

import torch
from sklearn.utils._param_validation import (  # type: ignore
    Interval,
    StrOptions,
    validate_params,  # type: ignore
)
from torch import nn

from ..types._defs import LOG_TWO_PI, LogBaseHazardFn


class Exponential(LogBaseHazardFn):
    r"""Implements the Exponential base hazard.

    Exponential base hazard is time independent.

    It is given by the formula:

    .. math::
        \lambda_0(t) = \lambda.

    This method expects:
        - `t0`: a column vector of previous transition times, shape `(n, 1)`.
        - `t1`: a matrix of future evaluation times, shape `(n, m)`, with the same
          number of rows as `t0`.

    The output is the log base hazard evaluated at each `t1` relative to `t0`.

    Optimization of the parameters can be disabled by checking the `forzen` flag.

    Attributes:
        log_lmda (nn.Parameter | torch.Tensor): The log rate factor.
        frozen (bool): Whether the parameters are frozen.
    """

    log_lmda: nn.Parameter | torch.Tensor

    @validate_params(
        {
            "lmda": [Interval(Real, 0, None, closed="neither")],
            "frozen": [bool],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(self, lmda: float, *, frozen: bool = False):
        """Initializes the Exponential hazard.

        Args:
            lmda (float): The rate factor.
            frozen (bool, optional): Whether to freeze the parameters. Defaults to
                `False`.
        """
        super().__init__()  # type: ignore

        log_lmda_tensor = torch.log(torch.tensor(lmda))
        self.log_lmda = nn.Parameter(log_lmda_tensor) if not frozen else log_lmda_tensor
        self.frozen = frozen

    def forward(
        self,
        t0: torch.Tensor,  # noqa: ARG002
        t1: torch.Tensor,  # noqa: ARG002
    ) -> torch.Tensor:
        """Calls the Exponential base hazard.

        Args:
            t0 (torch.Tensor): Previous transition times, shape :math:`(n, 1)`.
            t1 (torch.Tensor): Future evaluation times, shape :math:`(n, m)`.

        Returns:
            torch.Tensor: The computed base hazard in log scale.
        """
        return self.log_lmda

    @property
    def lmda(self) -> torch.Tensor:
        """Gets the rate factor.

        Returns:
            torch.Tensor: The rate factor.
        """
        return self.log_lmda.exp()


class Weibull(LogBaseHazardFn):
    r"""Implements the Weibull base hazard.

    Weibull base hazard is time dependent.

    It is given by the formula:

    .. math::
        \lambda_0(t) = k \lambda^k t^{k - 1}.

    This method expects:
        - `t0`: a column vector of previous transition times, shape `(n, 1)`.
        - `t1`: a matrix of future evaluation times, shape `(n, m)`, with the same
            number of rows as `t0`.

    The output is the log base hazard evaluated at each `t1` relative to `t0`.

    If `clock_type` is set to `sojourn`, given `t0` and `t1`, the transformation will be
    computed at `t1 - t0` (sojourn time), and simply `t1` if set to `absolute`.

    Optimization of the parameters can be disabled by checking the `forzen` flag.

    Attributes:
        log_lmda (nn.Parameter | torch.Tensor): The log of the scale parameter.
        log_k (nn.Parameter | torch.Tensor): The log of the shape parameter.
        clock_type (str): The type of clock to use.
        frozen (bool): Whether the parameters are frozen.
    """

    log_lmda: nn.Parameter | torch.Tensor
    log_k: nn.Parameter | torch.Tensor
    clock_type: str
    frozen: bool

    @validate_params(
        {
            "lmda": [Interval(Real, 0, None, closed="neither")],
            "k": [Interval(Real, 0, None, closed="neither")],
            "clock_type": [StrOptions({"sojourn", "absolute"})],
            "frozen": [bool],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        lmda: float,
        k: float,
        *,
        clock_type: str = "sojourn",
        frozen: bool = False,
    ):
        """Initializes the Weibull base hazard.

        Args:
            lmda (float): The scale parameter.
            k (float): The shape parameter.
            clock_type (str, optional): The type of clock to use. Defaults to "sojourn".
            frozen (bool, optional): Whether to freeze the parameters. Defaults to
                `False`.
        """
        super().__init__()  # type: ignore

        log_lmda_tensor = torch.log(torch.tensor(lmda))
        self.log_lmda = nn.Parameter(log_lmda_tensor) if not frozen else log_lmda_tensor
        log_k_tensor = torch.log(torch.tensor(k))
        self.log_k = nn.Parameter(log_k_tensor) if not frozen else log_k_tensor
        self.clock_type = clock_type
        self.frozen = frozen

    def forward(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        """Calls the Weibull base hazard.

        Args:
            t0 (torch.Tensor): Previous transition times, shape `(n, 1)`.
            t1 (torch.Tensor): Future evaluation times, shape `(n, m)`.

        Returns:
            torch.Tensor: The computed base hazard in log scale.
        """
        t = t1 - t0 if self.clock_type == "sojourn" else t1
        log_t = torch.log(t).clamp(min=-50)
        return self.log_k + self.k * self.log_lmda + (self.k - 1) * log_t

    @property
    def k(self) -> torch.Tensor:
        """Gets the shape parameter.

        Returns:
            torch.Tensor: The shape parameter.
        """
        return self.log_k.exp()

    @property
    def lmda(self) -> torch.Tensor:
        """Gets the scale parameter.

        Returns:
            torch.Tensor: The scale parameter.
        """
        return self.log_lmda.exp()


class Gompertz(LogBaseHazardFn):
    r"""Implements the Gompertz base hazard.

    Gompertz base hazard is time dependent. It is given by the formula:

    .. math::
        \lambda_0(t) = a \exp{bt}.

    This method expects:
        - `t0`: a column vector of previous transition times, shape `(n, 1)`.
        - `t1`: a matrix of future evaluation times, shape `(n, m)`, with the same
          number of rows as `t0`.

    The output is the log base hazard evaluated at each `t1` relative to `t0`.

    If `clock_type` is set to `sojourn`, given `t0` and `t1`, the transformation will be
    computed at `t1 - t0` (sojourn time), and simply `t1` if `clock_type` is set to
    `absolute`.

    Optimization of the parameters can be disabled by checking the `forzen` flag.

    Attributes:
        log_a (nn.Parameter | torch.Tensor): The baseline hazard parameter.
        b (nn.Parameter | torch.Tensor): The shape parameter.
        clock_type (str): The type of clock to use.
        frozen (bool): Whether the parameters are frozen.
    """

    log_a: nn.Parameter | torch.Tensor
    b: nn.Parameter | torch.Tensor
    clock_type: str
    frozen: bool

    @validate_params(
        {
            "a": [Interval(Real, 0, None, closed="neither")],
            "b": [Interval(Real, None, None, closed="neither")],
            "clock_type": [StrOptions({"sojourn", "absolute"})],
            "frozen": [bool],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        a: float,
        b: float,
        *,
        clock_type: str = "sojourn",
        frozen: bool = False,
    ):
        """Initializes the Gompertz base hazard.

        Args:
            a (float): The baseline hazard.
            b (float): The shape parameter.
            clock_type (str, optional): The type of clock to use. Defaults to "sojourn".
            frozen (bool, optional): Whether to freeze the parameters. Defaults to
                `False`.
        """
        super().__init__()  # type: ignore

        log_a_tensor = torch.log(torch.tensor(a))
        self.log_a = nn.Parameter(log_a_tensor) if not frozen else log_a_tensor
        b_tensor = torch.tensor(b)
        self.b = nn.Parameter(b_tensor) if not frozen else b_tensor
        self.clock_type = clock_type
        self.frozen = frozen

    def forward(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        """Calls the Gompertz base hazard.

        Args:
            t0 (torch.Tensor): Previous transition times, shape `(n, 1)`.
            t1 (torch.Tensor): Future evaluation times, shape `(n, m)`.

        Returns:
            torch.Tensor: The computed base hazard in log scale.
        """
        t = t1 - t0 if self.clock_type == "sojourn" else t1
        return self.log_a + self.b * t

    @property
    def a(self) -> torch.Tensor:
        """Gets the baseline hazard.

        Returns:
            torch.Tensor: The baseline hazard.
        """
        return self.log_a.exp()


class LogNormal(LogBaseHazardFn):
    r"""Implements the log normal base hazard.

    Log normal base hazard is time dependent. It is given by the formula:

    .. math::
        \lambda_0(t) = \frac{\phi\left( \frac{\log t - \mu}{\sigma} \right)}{t \sigma
        \, \Phi\left( -\frac{\log t - \mu}{\sigma} \right)},
        \quad t > 0,

    where:

    .. math::
        \phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}, \quad
        \Phi(z) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^z e^{-t^2/2} \, dt.

    This method expects:
        - `t0`: a column vector of previous transition times, shape `(n, 1)`.
        - `t1`: a matrix of future evaluation times, shape `(n, m)`, with the same
          number of rows as `t0`.

    The output is the log base hazard evaluated at each `t1` relative to `t0`.

    If `clock_type` is set to `sojourn`, given `t0` and `t1`, the transformation will be
    computed at `t1 - t0` (sojourn time), and simply `t1` if `clock_type` is set to
    `absolute`.

    Optimization of the parameters can be disabled by checking the `forzen` flag.

    Attributes:
        mu (nn.Parameter | torch.Tensor): The log time mean.
        log_scale (nn.Parameter | torch.Tensor): The log of scale.
        clock_type (str): The type of clock to use.
        frozen (bool): Whether the parameters are frozen.
    """

    mu: nn.Parameter | torch.Tensor
    log_scale: nn.Parameter | torch.Tensor
    clock_type: str
    frozen: bool

    @validate_params(
        {
            "mu": [Interval(Real, None, None, closed="neither")],
            "scale": [Interval(Real, 0, None, closed="neither")],
            "clock_type": [StrOptions({"sojourn", "absolute"})],
            "frozen": [bool],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        mu: float,
        scale: float,
        *,
        clock_type: str = "sojourn",
        frozen: bool = False,
    ):
        """Initializes the log normal base hazard.

        Args:
            mu (float): The log time mean.
            scale (float): The log time scale.
            clock_type (str, optional): The type of clock to use. Defaults to "sojourn".
            frozen (bool, optional): Whether to freeze the parameters. Defaults to
                `False`.
        """
        super().__init__()  # type: ignore

        mu_tensor = torch.tensor(mu)
        self.mu = nn.Parameter(mu_tensor) if not frozen else mu_tensor
        log_scale_tensor = torch.log(torch.tensor(scale))
        self.log_scale = (
            nn.Parameter(log_scale_tensor) if not frozen else log_scale_tensor
        )
        self.clock_type = clock_type
        self.frozen = frozen

    def forward(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        """Calls the log normal base hazard.

        Args:
            t0 (torch.Tensor): Previous transition times, shape `(n, 1)`.
            t1 (torch.Tensor): Future evaluation times, shape `(n, m)`.

        Returns:
            torch.Tensor: The computed base hazard in log scale.
        """
        t = t1 - t0 if self.clock_type == "sojourn" else t1
        log_t = torch.log(t).clamp(min=-50)
        z = (log_t - self.mu) / self.scale
        log_pdf = -log_t - self.log_scale - 0.5 * LOG_TWO_PI - 0.5 * z**2
        log_sf = cast(torch.Tensor, torch.special.log_ndtr(-z))  # type: ignore
        return log_pdf - log_sf

    @property
    def scale(self) -> torch.Tensor:
        """Gets the scale.

        Returns:
            torch.Tensor: The scale.
        """
        return self.log_scale.exp()
