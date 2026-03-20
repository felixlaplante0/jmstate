from bisect import bisect_left
from numbers import Integral, Real
from typing import Final

import torch
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils._param_validation import Interval, validate_params  # type: ignore
from torch.distributions import Normal
from torch.nn.utils import parameters_to_vector

from ..types._data import ModelDesign
from ..types._parameters import ModelParameters
from ._fit import FitMixin
from ._predict import PredictMixin
from ._sampler import MetropolisWithinGibbsSampler

# Constants
SIGNIFICANCE_LEVELS: Final[tuple[float, ...]] = (
    0.001,
    0.01,
    0.05,
    0.1,
    float("inf"),
)
SIGNIFICANCE_CODES: Final[tuple[str, ...]] = (
    "[red3]***[/]",
    "[orange3]**[/]",
    "[yellow3]*[/]",
    ".",
    "",
)


class MultiStateJointModel(BaseEstimator, FitMixin, PredictMixin):
    r"""Nonlinear multistate joint model for longitudinal and survival data.

    This class generalizes both linear and standard joint models to accommodate
    multiple states under a semi-Markov assumption. The model is fully specified
    by a `ModelDesign` object, which defines longitudinal and hazard functions, and
    a `ModelParameters` object, which contains the associated (modifiable) parameters.
    Parameters may be shared across functions as specified in the model design. The
    `ModelDesign` is fixed at initialization, while `ModelParameters` are updated
    in place during fitting.

    Model fitting is performed using a stochastic gradient ascent algorithm, and
    parameter sampling is handled by a Metropolis-Within-Gibbs MCMC procedure. The
    default settings are based on commonly accepted values, and the step sizes are
    adapted component-wise dynamically based on the acceptance rate.

    Dynamic prediction is supported via the prediction methods in `PredictMixin`,
    which allow both single and double Monte Carlo integration.

    Design settings:
        - `design`: The model specification defining the individual parameters,
          regression, and link functions.
        - `params`: Initial values for the model parameters; modifiable during fitting.

    Numerical integration settings:
        - `n_quad`: Number of nodes for Gauss-Legendre quadrature in hazard integration.
        - `n_bisect`: Number of bisection steps for transition time sampling.

    MCMC settings:
        - `sampler`: MCMC sampler instance used during model fitting.
        - `n_chains`: Number of parallel MCMC chains.
        - `init_step_size`: Initial kernel standard deviation in
           Metropolis-Within-Gibbs.
        - `adapt_rate`: Adaptation rate for the step size.
        - `target_accept_rate`: Target mean acceptance probability.
        - `n_warmup`: Number of warmup iterations per chain.
        - `n_subsample`: Number of subsamples between predictions; higher values
          reduce autocorrelation but increase computation time. A value of one means
          no subsampling. This value may be very sensitive.

    Fitting settings:
        - `optimizer`: Optimizer for stochastic gradient ascent. If `None`, fitting
          is disabled. Recommended: `torch.optim.Adam` with learning rate 0.01 to 0.5.
        - `max_iter_fit`: Maximum iterations for gradient ascent.
        - `n_samples_summary`: Number of samples used to compute the Fisher
          Information Matrix and model selection criteria; higher values improve
          accuracy.
        - `tol`: Tolerance for the :math:`R^2` convergence criterion.
        - `window_size`: Window size for :math:`R^2` convergence; default 100.
          This criterion is scale-agnostic and provides a local stationarity test.

    Printing and visualization:
        - `verbose`: Whether to print progress during fitting and prediction.
        - After fitting, `summary` and `plot_params_history` (from `jmstate.utils`)
          can be used to display p-values, log-likelihood, AIC, BIC, and the
          evolution of parameters over iterations.

    Attributes:
        design (ModelDesign): The model specification defining the individual
            parameters, regression, and link functions.
        params (ModelParameters): The modifiable model parameters.
        optimizer (torch.optim.Optimizer | None): Optimizer used for fitting.
        n_quad (int): Number of Gauss-Legendre quadrature nodes for hazard integration.
        n_bisect (int): Number of bisection steps for transition time sampling.
        sampler (MetropolisWithinGibbsSampler | None): MCMC sampler instance used during
            model fitting.
        n_chains (int): Number of parallel MCMC chains.
        init_step_size (float): Initial kernel standard deviation in
            Metropolis-Within-Gibbs.
        adapt_rate (float): Adaptation rate for the MCMC step size.
        target_accept_rate (float): Target acceptance probability.
        n_warmup (int): Number of warmup iterations per MCMC chain.
        n_subsample (int): Number of subsamples for MCMC iterations.
        max_iter_fit (int): Maximum number of iterations for stochastic gradient ascent.
        tol (float): Tolerance for :math:`R^2` convergence criterion.
        window_size (int): Window size for :math:`R^2` convergence evaluation.
        n_samples_summary (int): Number of posterior samples for computing Fisher
            Information and selection criteria.
        verbose (bool): Flag to print fitting and prediction progress.
        params_history_ (list[torch.Tensor]): History of parameter values as flattened
            tensors.
        fim_ (torch.Tensor | None): Fisher Information Matrix.
        loglik_ (float | None): Log-likelihood of the fitted model.
        aic_ (float | None): Akaike Information Criterion.
        bic_ (float | None): Bayesian Information Criterion.

    Examples:
        >>> from jmstate import MultiStateJointModel
        >>> optimizer = torch.optim.Adam(params.parameters(), lr=0.1)
        >>> model = MultiStateJointModel(design, params, optimizer)
        >>> model.fit(data)
        >>> model.summary()
    """

    design: ModelDesign
    params: ModelParameters
    optimizer: torch.optim.Optimizer | None
    n_quad: int
    n_bisect: int
    sampler: MetropolisWithinGibbsSampler | None
    n_chains: int
    init_step_size: float
    adapt_rate: float
    target_accept_rate: float
    n_warmup: int
    n_subsample: int
    max_iter: int
    tol: float
    window_size: int
    verbose: bool | int
    params_history_: list[torch.Tensor]
    fim_: torch.Tensor | None
    loglik_: float | None
    aic_: float | None
    bic_: float | None

    @validate_params(
        {
            "design": [ModelDesign],
            "params": [ModelParameters],
            "optimizer": [torch.optim.Optimizer, None],
            "n_quad": [Interval(Integral, 1, None, closed="left")],
            "n_bisect": [Interval(Integral, 1, None, closed="left")],
            "n_chains": [Interval(Integral, 1, None, closed="left")],
            "init_step_size": [Interval(Real, 0, None, closed="neither")],
            "adapt_rate": [Interval(Real, 0, None, closed="left")],
            "target_accept_rate": [Interval(Real, 0, 1, closed="neither")],
            "n_warmup": [Interval(Integral, 0, None, closed="left")],
            "n_subsample": [Interval(Integral, 0, None, closed="left")],
            "max_iter": [Interval(Integral, 0, None, closed="left")],
            "tol": [Interval(Real, 0, 1, closed="both")],
            "window_size": [Interval(Integral, 2, None, closed="left")],
            "verbose": ["verbose"],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        design: ModelDesign,
        params: ModelParameters,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        n_quad: int = 32,
        n_bisect: int = 32,
        n_chains: int = 5,
        init_step_size: float = 0.1,
        adapt_rate: float = 0.1,
        target_accept_rate: float = 0.44,
        n_warmup: int = 100,
        n_subsample: int = 10,
        max_iter: int = 1000,
        tol: float = 0.1,
        window_size: int = 100,
        verbose: bool | int = True,
    ):
        r"""Initialize the multistate joint model with specified design and parameters.

        Constructs a joint model based on a user-defined `ModelDesign` and initial
        `ModelParameters`. Provides default settings for numerical integration, MCMC
        sampling, stochastic gradient fitting, and printing options.

        Args:
            design (ModelDesign): The model specification defining the individual
                parameters, regression, and link functions.
            params (ModelParameters): Initial values for the model parameters;
                modifiable during fitting.
            optimizer (torch.optim.Optimizer | None, optional): Optimizer used for
                fitting. If `None`, fitting is disabled. Defaults to None.
            n_quad (int, optional): Number of nodes for Gauss-Legendre quadrature in
                hazard integration. Defaults to 32.
            n_bisect (int, optional): Number of bisection steps for transition time
                sampling. Defaults to 32.
            n_chains (int, optional): Number of parallel MCMC chains. Defaults to 5.
            init_step_size (float, optional): Initial step size for the MCMC sampler.
                Defaults to 0.1.
            adapt_rate (float, optional): Adaptation rate for the MCMC step size.
                Defaults to 0.1.
            target_accept_rate (float, optional): Target mean acceptance probability for
                Metropolis-Within-Gibbs. Defaults to 0.44.
            n_warmup (int, optional): Number of warmup iterations per MCMC chain.
                Defaults to 100.
            n_subsample (int, optional): Number of subsamples between MCMC updates.
                Defaults to 10.
            max_iter (int, optional): Maximum number of iterations for stochastic
                gradient ascent. Defaults to 1000.
            tol (float, optional): Tolerance for :math:`R^2` convergence criterion.
                Defaults to 0.1.
            window_size (int, optional): Window size for :math:`R^2` convergence
                evaluation. Defaults to 100.
            verbose (bool | int, optional): Flag to print progress during fitting and
                prediction. Defaults to True.
        """
        # Info of the Mixin Classes
        super().__init__(
            optimizer=optimizer,
            n_quad=n_quad,
            n_bisect=n_bisect,
            n_chains=n_chains,
            init_step_size=init_step_size,
            adapt_rate=adapt_rate,
            target_accept_rate=target_accept_rate,
            max_iter=max_iter,
            tol=tol,
            window_size=window_size,
        )

        # Store model components
        self.design = design
        self.params = params
        self.n_warmup = n_warmup
        self.n_subsample = n_subsample
        self.verbose = verbose
        self.params_history_ = [parameters_to_vector(self.params.parameters()).detach()]
        self.fim_ = None
        self.loglik_ = None
        self.aic_ = None
        self.bic_ = None

    @property
    def stderr(self) -> torch.Tensor:
        r"""Computes the estimated standard errors of the model parameters.

        The standard errors are derived from the diagonal of the inverse of the
        estimated Fisher Information Matrix evaluated at the Maximum Likelihood Estimate
        (MLE). They provide a measure of uncertainty for each parameter and can be used
        to construct confidence intervals.

        .. math::
            \mathrm{stderr} = \sqrt{\operatorname{diag}\left( \hat{\mathcal{I}}_n
            (\hat{\theta})^{-1} \right)}

        Raises:
            ValueError: If the model has not been fitted and the Fisher Information
                Matrix is unavailable.

        Returns:
            torch.Tensor: Vector of standard errors corresponding to each parameter.
        """
        if self.fim_ is None:
            raise ValueError(
                "Fisher Information Matrix is not available. Call "
                "compute_summary() first."
            )

        return self.fim_.inverse().diag().sqrt()

    def summary(self):
        r"""Print a statistical summary of the fitted multistate joint model.

        This function displays the estimated parameter values, their standard errors,
        and the (nullity) p-values testing the hypothesis that each coefficient is zero.
        Note that for certain parameters—such as precision components—these p-values
        may not be meaningful and should be interpreted with caution.

        Additionally, the function prints model selection metrics including the log
        likelihood, Akaike Information Criterion (AIC), and Bayesian Information
        Criterion (BIC), with color-enhanced formatting for readability.

        Raises:
            ValueError: If the model has not been fitted.
        """
        vector = parameters_to_vector(self.params.parameters())
        stderr = self.stderr
        zvalues = torch.abs(vector / stderr)
        pvalues = 2 * (1 - Normal(0, 1).cdf(zvalues))

        table = Table()
        table.add_column("Parameter name", justify="left")
        table.add_column("Value", justify="center")
        table.add_column("Standard Error", justify="center")
        table.add_column("z-value", justify="center")
        table.add_column("p-value", justify="center")
        table.add_column("Significance level", justify="center")

        i = 0
        for name, val in self.params.named_parameters():
            for j in range(val.numel()):
                code = SIGNIFICANCE_CODES[
                    bisect_left(SIGNIFICANCE_LEVELS, pvalues[i].item())
                ]

                table.add_row(
                    f"{name}[{j}]",
                    f"{vector[i].item():.3f}",
                    f"{stderr[i].item():.3f}",
                    f"{zvalues[i].item():.3f}",
                    f"{pvalues[i].item():.3f}",
                    code,
                )
                i += 1

        criteria = Text(
            f"Log-likelihood: {self.loglik_:.3f}\n"
            f"AIC: {self.aic_:.3f}\n"
            f"BIC: {self.bic_:.3f}",
            style="bold cyan",
        )

        content = Group(table, Rule(style="dim"), criteria, Rule(style="dim"))
        panel = Panel(
            content, title="Model Summary", border_style="green", expand=False
        )

        Console().print(panel)
