"""Base hazard functions."""

__all__ = ["Exponential", "Gompertz", "LogNormal", "Neural", "Weibull"]

from numbers import Real
from typing import cast

import torch
from sklearn.utils._param_validation import (  # type: ignore
    Interval,
    StrOptions,
    validate_params,  # type: ignore
)
from torch import nn

from ..types._defs import LOG_CLAMP, LOG_TWO_PI, LogBaseHazardFn


def _register(module: nn.Module, frozen: bool, **tensors: torch.Tensor) -> None:
    """Registers tensors as buffers if frozen, else as parameters, in order.

    Args:
        module (nn.Module): The module owning the tensors.
        frozen (bool): Whether to freeze the tensors.
        **tensors (torch.Tensor): The tensors to register, by name.
    """
    for name, tensor in tensors.items():
        if frozen:
            module.register_buffer(name, tensor)
        else:
            setattr(module, name, nn.Parameter(tensor))
    module.frozen = frozen


class Neural(LogBaseHazardFn):
    r"""Implements a neural-network log base hazard.

    The supplied network maps each scalar time to a scalar log hazard. It receives a
    two-dimensional tensor of shape ``(n_times, 1)`` and its output is reshaped to the
    input time shape.

    Args:
        nn (nn.Module): Network mapping scalar times to scalar log hazards.
        clock_type (str, optional): The clock used to construct the input times.
            Defaults to ``"sojourn"``.
    """

    @validate_params(
        {
            "nn": [nn.Module],
            "clock_type": [StrOptions({"sojourn", "absolute"})],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(self, nn: nn.Module, *, clock_type: str = "sojourn"):
        """Initializes the neural log base hazard.

        Args:
            nn (nn.Module): Network mapping scalar times to scalar log hazards.
            clock_type (str, optional): The type of clock to use. Defaults to
                "sojourn".
        """
        super().__init__()  # type: ignore
        self.nn = nn
        self.clock_type = clock_type

    def forward(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        """Calls the neural log base hazard.

        Args:
            t0 (torch.Tensor): Previous transition times, shape `(n, 1)`.
            t1 (torch.Tensor): Future evaluation times, shape `(n, m)`.

        Returns:
            torch.Tensor: The computed base hazard in log scale, shaped as `t1`.
        """
        t = t1 - t0 if self.clock_type == "sojourn" else t1
        first = next(self.nn.parameters(), None)
        if first is not None:
            t = t.to(first.dtype)
        return self.nn(t.reshape(-1, 1)).reshape(t.shape)


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

    Optimization of the parameters can be disabled by checking the `frozen` flag.

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

        _register(self, frozen, log_lmda=torch.log(torch.tensor(lmda)))

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

    Optimization of the parameters can be disabled by checking the `frozen` flag.

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

        _register(
            self,
            frozen,
            log_lmda=torch.log(torch.tensor(lmda)),
            log_k=torch.log(torch.tensor(k)),
        )
        self.clock_type = clock_type

    def forward(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        """Calls the Weibull base hazard.

        Args:
            t0 (torch.Tensor): Previous transition times, shape `(n, 1)`.
            t1 (torch.Tensor): Future evaluation times, shape `(n, m)`.

        Returns:
            torch.Tensor: The computed base hazard in log scale.
        """
        t = t1 - t0 if self.clock_type == "sojourn" else t1
        log_t = torch.log(t).clamp(min=-LOG_CLAMP)
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

    Optimization of the parameters can be disabled by checking the `frozen` flag.

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

        _register(self, frozen, log_a=torch.log(torch.tensor(a)), b=torch.tensor(b))
        self.clock_type = clock_type

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

    Optimization of the parameters can be disabled by checking the `frozen` flag.

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

        _register(
            self,
            frozen,
            mu=torch.tensor(mu),
            log_scale=torch.log(torch.tensor(scale)),
        )
        self.clock_type = clock_type

    def forward(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        """Calls the log normal base hazard.

        Args:
            t0 (torch.Tensor): Previous transition times, shape `(n, 1)`.
            t1 (torch.Tensor): Future evaluation times, shape `(n, m)`.

        Returns:
            torch.Tensor: The computed base hazard in log scale.
        """
        t = t1 - t0 if self.clock_type == "sojourn" else t1
        log_t = torch.log(t).clamp(min=-LOG_CLAMP)
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
