from math import ceil
from typing import Any, Self
from warnings import warn

import torch
from sklearn.utils._param_validation import validate_params  # type: ignore
from torch import nn
from torch.func import jacfwd  # type: ignore
from torch.nn.utils import parameters_to_vector
from torch.nn.utils.stateless import _reparametrize_module  # type: ignore
from tqdm import trange

from ..types._data import ModelData, ModelDataUnchecked, ModelDesign
from ..types._parameters import ModelParameters
from ._hazard import HazardMixin
from ._longitudinal import LongitudinalMixin
from ._prior import PriorMixin
from ._sampler import MCMCMixin, MetropolisWithinGibbsSampler


class FitMixin(PriorMixin, LongitudinalMixin, HazardMixin, MCMCMixin, nn.Module):
    """Mixin for fitting the model."""

    design: ModelDesign
    params: ModelParameters
    optimizer: torch.optim.Optimizer | None
    n_warmup: int
    n_subsample: int
    max_iter_fit: int
    tol: float
    window_size: int
    n_samples_summary: int
    verbose: bool | int
    params_history_: list[torch.Tensor]
    fim_: torch.Tensor | None
    loglik_: float | None
    aic_: float | None
    bic_: float | None

    def __init__(
        self,
        optimizer: torch.optim.Optimizer | None,
        max_iter_fit: int,
        tol: float,
        window_size: int,
        n_samples_summary: int,
        *args: Any,
        **kwargs: Any,
    ):
        """Initializes the fit parameters.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer.
            max_iter_fit (int): The maximum number of iterations for fitting.
            tol (float): The tolerance for the convergence.
            window_size (int): The window size for the convergence.
            n_samples_summary (int): The number of samples used to compute Fisher
                Information Matrix and model selection criteria.
        """
        super().__init__(*args, **kwargs)

        self.optimizer = optimizer
        self.max_iter_fit = max_iter_fit
        self.tol = tol
        self.window_size = window_size
        self.n_samples_summary = n_samples_summary

    def _logpdfs_fn(
        self,
        data: ModelDataUnchecked,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Gets the log pdfs.

        Args:
            data (ModelData): Dataset on which likelihood is computed.
            b (torch.Tensor): The random effects.

        Returns:
           torch.Tensor: The log pdfs.
        """
        indiv_params = self.design.indiv_params_fn(self.params.fixed_effects, data.x, b)
        return (
            self._longitudinal_logliks(data, indiv_params)
            + self._hazard_logliks(data, indiv_params)
            + self._prior_logliks(b)
        )

    def _is_converged(self) -> bool:
        """Checks if the optimizer has converged.

        This is based on a linear regression of the parameters with the current
        number of iterations. If the mean of :math:`R^2` is below a threshold,
        the optimizer is considered to have converged.

        Returns:
            bool: True if the optimizer has converged, False otherwise.
        """

        def r2(Y: torch.Tensor) -> torch.Tensor:
            n = Y.size(0)
            i = torch.arange(n, dtype=torch.get_default_dtype())
            i_centered = i - (n - 1) / 2
            y_centered = Y - Y.mean(dim=0)
            num = (i_centered @ y_centered) ** 2
            den = i_centered.pow(2).sum() * y_centered.pow(2).sum(dim=0)
            return (num / den).nan_to_num()

        if len(self.params_history_) < self.window_size:
            return False

        Y = torch.stack(self.params_history_[-self.window_size :])
        return r2(Y).mean().item() < self.tol

    def _fit(self, data: ModelDataUnchecked, sampler: MetropolisWithinGibbsSampler):
        """Fits the model using the optimizer and the sampler.

        Args:
            data (ModelData): The data.
            sampler (MetropolisWithinGibbsSampler): The sampler.

        Raises:
            ValueError: If the optimizer is not initialized.
        """
        if self.max_iter_fit <= 0:
            return

        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized.")

        def closure():
            self.optimizer.zero_grad()  # type: ignore
            loss = -self._logpdfs_fn(data, sampler.b).mean()
            loss.backward()  # type: ignore
            return loss.item()

        for i in trange(  # noqa: B007
            self.max_iter_fit,
            desc="Fitting joint model",
            disable=not bool(self.verbose),
        ):
            self.optimizer.step(closure)
            self.params_history_.append(
                parameters_to_vector(self.params.parameters()).detach()
            )

            # Restore logpdfs and indiv_params, because parameters changed
            sampler.reset().run(self.n_subsample)

            if self._is_converged():
                break

        if i == self.max_iter_fit - 1:  # type: ignore
            warn(
                "Model may not have converged in the specified number of iterations. "
                "Try to increase `max_iter_fit`, `tol`, or `window_size`. Also try "
                "to increase `n_subsample` or `n_warmup` for better MCMC mixing.",
                stacklevel=2,
            )

    def _compute_fim_and_criteria(
        self, data: ModelDataUnchecked, sampler: MetropolisWithinGibbsSampler
    ):
        """Computes the Fisher Information Matrix and model selection criteria.

        Args:
            data (ModelData): The data.
            sampler (MetropolisWithinGibbsSampler): The sampler.
        """
        if self.n_samples_summary <= 0:
            return

        n, q = len(data), sampler.b.size(-1)

        # Jac forward since output dimension > input dimension
        @jacfwd  # type: ignore
        def _dict_jac_fn(
            named_parameters_dict: dict[str, torch.Tensor], b: torch.Tensor
        ) -> torch.Tensor:
            with _reparametrize_module(self, named_parameters_dict):
                return self._logpdfs_fn(data, b).mean(dim=0)

        def _jac_fn(b: torch.Tensor) -> torch.Tensor:
            out = _dict_jac_fn(dict(self.named_parameters()), b)  # type: ignore
            return torch.cat([p.reshape(n, -1) for p in out.values()], dim=-1)  # type: ignore

        # Initialize accumulators
        mjac = torch.zeros(n, self.params.numel())
        logpdf = 0.0
        mb = torch.zeros(n, q)
        mb2 = torch.zeros(n, q, q)

        n_iter = ceil(self.n_samples_summary / self.n_chains)
        for _ in trange(
            n_iter,
            desc="Computing FIM and Model Selection Criteria",
            disable=not bool(self.verbose),
        ):
            # Mean jacobian across chains
            mjac += _jac_fn(sampler.b).detach()  # type: ignore

            # Mean logpdf across chains
            logpdf += sampler.logpdfs.sum().item() / self.n_chains

            # Mean and outer product of b across chains
            mb += sampler.b.mean(dim=0)
            mb2 += torch.einsum("ijk,ijl->jkl", sampler.b, sampler.b) / self.n_chains

            sampler.run(self.n_subsample)

        mjac /= n_iter
        logpdf /= n_iter
        mb /= n_iter
        mb2 /= n_iter

        # Compute FIM as variance of the score
        self.fim_ = mjac.T @ mjac

        # Compute entropy Laplace approximation
        covs = mb2 - torch.einsum("ij,ik->ijk", mb, mb)
        entropy = 0.5 * (torch.logdet(covs) + self.params.random_prec.dim).sum().item()

        self.loglik_ = logpdf + entropy
        self.aic_ = -2 * self.loglik_ + 2 * self.params.numel()
        self.bic_ = -2 * self.loglik_ + torch.logdet(self.fim_).item()

    @validate_params(
        {
            "data": [ModelData],
        },
        prefer_skip_nested_validation=True,
    )
    def fit(self, data: ModelData) -> Self:
        r"""Fit the model to observed data using maximum likelihood estimation.

        Computes the Maximum Likelihood Estimate (MLE) :math:`\hat{\theta}` of the model
        parameters. Optimization is performed using the configured `optimizer` for up to
        `max_iter_fit` iterations. Convergence is assessed via a linearity-based
        stationarity test on the last `window_size` iterates: the :math:`R^2` statistic
        measures whether the trajectory of each parameter component is better explained
        by a linear trend than by a constant. Convergence is declared when all
        :math:`R^2` values are below `tol`, indicating negligible linear drift.

        The fitting procedure leverages the Fisher identity coupled with a stochastic
        gradient algorithm and a Metropolis-Hastings MCMC sampler. The Fisher identity
        states:

        .. math::
            \nabla_\theta \log \mathcal{L}(\theta ; x) = \mathbb{E}_{b \sim p(\cdot
            \mid x, \theta)} \left( \nabla_\theta \log \mathcal{L}(\theta ; x, b)
            \right).

        The expected Fisher Information Matrix is estimated as:

        .. math::
            \mathcal{I}_n(\theta) = \sum_{i=1}^n \mathbb{E}_{b \sim p(\cdot \mid x_i,
            \hat{\theta})} \left(\nabla \log \mathcal{L}(\hat{\theta} ; x_i, b) \nabla
            \log \mathcal{L}(\hat{\theta} ; x_i, b)^T \right).

        Model selection criteria are computed using a Laplace approximation of the
        posterior distribution, providing closed-form estimates of entropy. The
        parameter `n_samples_summary` controls the number of posterior samples used for
        computing the Fisher Information Matrix and selection metrics; increasing this
        number improves accuracy at the cost of computational time.

        For additional details, see ISSN 2824-7795.

        Args:
            data (ModelData): Dataset containing covariates, longitudinal measurements,
                trajectories, and censoring times used for fitting.

        Raises:
            ValueError: If the optimizer has not been initialized prior to fitting.

        Returns:
            Self: The fitted model instance with estimated parameters.
        """
        data = ModelDataUnchecked(
            data.x, data.t, data.y, data.trajectories, data.c
        ).prepare(self)

        # Initialize MCMC
        sampler = self._init_sampler(data).run(self.n_warmup)

        self._fit(data, sampler)
        self._compute_fim_and_criteria(data, sampler)

        return self
