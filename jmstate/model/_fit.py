from math import ceil, log
from numbers import Integral
from typing import Any, Self
from warnings import warn

import torch
from sklearn.base import check_is_fitted  # type: ignore
from sklearn.exceptions import ConvergenceWarning  # type: ignore
from sklearn.utils._param_validation import Interval, validate_params  # type: ignore
from torch import nn
from torch.distributions import MultivariateNormal
from torch.func import jacfwd  # type: ignore
from torch.nn.utils import parameters_to_vector
from torch.nn.utils.stateless import _reparametrize_module  # type: ignore
from tqdm import trange

from ..types._data import (
    ModelData,
    ModelDataUnchecked,
    ModelDesign,
    prepare_model_data,
)
from ..types._parameters import ModelParameters
from ..utils._dtype import dtype_device
from ._hazard import HazardMixin
from ._longitudinal import LongitudinalMixin
from ._prior import PriorMixin
from ._sampler import MCMCMixin, MetropolisWithinGibbsSampler


class FitMixin(PriorMixin, LongitudinalMixin, HazardMixin, MCMCMixin, nn.Module):
    """Mixin for fitting the model."""

    design: ModelDesign
    params: ModelParameters
    optimizer: torch.optim.Optimizer | None
    sampler: MetropolisWithinGibbsSampler | None
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

    def __init__(
        self,
        optimizer: torch.optim.Optimizer | None,
        max_iter: int,
        tol: float,
        window_size: int,
        *args: Any,
        **kwargs: Any,
    ):
        """Initializes the fit parameters.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer.
            max_iter (int): The maximum number of iterations for fitting.
            tol (float): The tolerance for the convergence.
            window_size (int): The window size for the convergence.
        """
        super().__init__(*args, **kwargs)

        self.optimizer = optimizer
        self.sampler = None
        self.max_iter = max_iter
        self.tol = tol
        self.window_size = window_size

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

        This is based on a linear regression of the parameters with the current number
        of iterations. If the mean of :math:`R^2` is below a threshold, the optimizer is
        considered to have converged.

        Returns:
            bool: True if the optimizer has converged, False otherwise.
        """

        def r2(Y: torch.Tensor) -> torch.Tensor:
            n = Y.size(0)
            i = torch.arange(n, dtype=Y.dtype, device=Y.device)
            i_centered = i - (n - 1) / 2
            y_centered = Y - Y.mean(dim=0)
            num = (i_centered @ y_centered) ** 2
            den = i_centered.pow(2).sum() * y_centered.pow(2).sum(dim=0)
            return (num / den).nan_to_num()

        if len(self.params_history_) < self.window_size:
            return False

        last = self.params_history_[-1]
        Y = torch.stack(
            [
                h.to(dtype=last.dtype, device=last.device)
                for h in self.params_history_[-self.window_size :]
            ]
        )
        return r2(Y).mean().item() < self.tol

    def _warn_not_converged(self, *, stacklevel: int) -> None:
        """Emits a convergence warning for an exhausted optimization budget.

        Args:
            stacklevel (int): Warning stack level passed to `warn`.
        """
        warn(
            "Model may not have converged in the specified number of iterations. "
            "Try to increase `max_iter`, `tol`, or `window_size`. Also try "
            "to increase `n_subsample` or `n_warmup` for better MCMC mixing.",
            category=ConvergenceWarning,
            stacklevel=stacklevel,
        )

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
        `max_iter` iterations. Convergence is assessed via a linearity-based
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

        Args:
            data (ModelData): Dataset containing covariates, longitudinal measurements,
                trajectories, and censoring times used for fitting.

        Raises:
            ValueError: If the optimizer has not been initialized prior to fitting.

        Returns:
            Self: The fitted model instance with estimated parameters.
        """
        data = prepare_model_data(data, self)

        # Initialize MCMC
        self.sampler = self._init_sampler(data).run(self.n_warmup)

        def closure():
            self.optimizer.zero_grad()  # type: ignore
            loss = -self.sampler.logpdfs_fn(self.sampler.b).mean()  # type: ignore
            loss.backward()  # type: ignore
            return loss.item()

        converged = False
        for _ in trange(
            self.max_iter,
            desc="Fitting joint model",
            disable=not bool(self.verbose),
        ):
            self.optimizer.step(closure)
            self.params_history_.append(
                parameters_to_vector(self.params.parameters()).detach()
            )

            # Restore logpdfs and indiv_params, because parameters changed
            self.sampler.reset().run(self.n_subsample)

            if self._is_converged():
                converged = True
                break

        if self.max_iter > 0 and not converged:
            self._warn_not_converged(stacklevel=3)

        return self

    @validate_params(
        {
            "n_posterior_samples": [Interval(Integral, 1, None, closed="left")],
            "n_importance_samples": [Interval(Integral, 1, None, closed="left")],
            "importance_batch_size": [Interval(Integral, 1, None, closed="left")],
        },
        prefer_skip_nested_validation=True,
    )
    def compute_summary(
        self,
        *,
        n_posterior_samples: int = 500,
        n_importance_samples: int = 1000,
        importance_batch_size: int = 128,
    ) -> Self:
        r"""Computes summary statistics for the fitted model.

        The expected Fisher Information Matrix is estimated as:

        .. math::
            \mathcal{I}_n(\theta) = \sum_{i=1}^n \mathbb{E}_{b \sim p(\cdot \mid x_i,
            \hat{\theta})} \left(\nabla \log \mathcal{L}(\hat{\theta} ; x_i, b) \nabla
            \log \mathcal{L}(\hat{\theta} ; x_i, b)^T \right).

        Model selection criteria use importance sampling with subject-specific Gaussian
        proposals fitted to posterior draws. The first sampling stage estimates each
        proposal's mean and covariance, and the second draws independent samples from
        those proposals to estimate the marginal likelihood.

        For additional details, see ISSN 2824-7795.

        Args:
            n_posterior_samples (int, optional): Number of posterior samples used to
                estimate the Fisher Information Matrix and Gaussian proposal moments.
                Defaults to 500.
            n_importance_samples (int, optional): Number of independent Gaussian
                proposal draws used to estimate the marginal likelihood. Defaults to
                1000.
            importance_batch_size (int, optional): Number of importance draws evaluated
                together. This controls memory use, not statistical accuracy. Defaults
                to 128.

        Returns:
            Self: The fitted model with summary statistics computed.
        """
        check_is_fitted(self, "sampler")

        n, q = self.sampler.b.shape[1:]  # type: ignore

        # Jac forward since output dimension > input dimension
        @jacfwd  # type: ignore
        def _dict_jac_fn(
            named_parameters_dict: dict[str, torch.Tensor],
        ) -> torch.Tensor:
            with _reparametrize_module(self, named_parameters_dict):
                logpdfs = self.sampler.logpdfs_fn(self.sampler.b)  # type: ignore
                return logpdfs.mean(dim=0)

        def _jac_fn() -> torch.Tensor:
            out = _dict_jac_fn(dict(self.named_parameters()))  # type: ignore
            parts = [p.reshape(n, -1) for p in out.values()]  # type: ignore
            return torch.cat(parts, dim=-1)

        # Initialize accumulators on the model device in kernel precision
        dtype, device = dtype_device(self.params)
        working = dtype
        mjac = torch.zeros(n, self.params.numel(), dtype=working, device=device)
        mb = torch.zeros(n, q, dtype=working, device=device)
        mb2 = torch.zeros(n, q, q, dtype=working, device=device)

        n_iter = ceil(n_posterior_samples / self.n_chains)
        for _ in trange(
            n_iter,
            desc="Estimating FIM and Gaussian proposal",
            disable=not bool(self.verbose),
        ):
            # Mean jacobian across chains
            mjac += _jac_fn().detach()  # type: ignore

            # Mean and outer product of b across chains
            b = self.sampler.b  # type: ignore
            mb += b.mean(dim=0)
            mb2 += torch.einsum("ijk,ijl->jkl", b, b) / self.n_chains

            self.sampler.run(self.n_subsample)  # type: ignore

        mjac /= n_iter
        mb /= n_iter
        mb2 /= n_iter

        # Compute FIM as variance of the score
        self.fim_ = mjac.T @ mjac

        # Fit Gaussian proposals to the posterior moments
        covs = mb2 - torch.einsum("ij,ik->ijk", mb, mb)
        covs = 0.5 * (covs + covs.mT)
        # A divergent fit can leave the posterior moments non-finite. Sanitize
        # them so the proposal stays valid; the failure then surfaces as a NaN
        # likelihood instead of a crash.
        mb = torch.nan_to_num(mb, nan=0.0, posinf=1e6, neginf=-1e6)
        covs = torch.nan_to_num(covs, nan=0.0, posinf=1e6, neginf=0.0)
        covs = 0.5 * (covs + covs.mT)
        try:
            proposal = MultivariateNormal(mb, covariance_matrix=covs)
        except (ValueError, RuntimeError):
            # Empirical covariance may be singular; add a small relative jitter
            jitter = 1e-6 * covs.diagonal(dim1=-2, dim2=-1).mean().clamp(min=1e-6)
            eye = torch.eye(q, dtype=covs.dtype, device=covs.device)
            try:
                proposal = MultivariateNormal(
                    mb, covariance_matrix=covs + jitter * eye
                )
            except (ValueError, RuntimeError):
                # Last resort: diagonal proposal with floored variances
                var = covs.diagonal(dim1=-2, dim2=-1).clamp(min=1e-6)
                proposal = MultivariateNormal(
                    mb, covariance_matrix=torch.diag_embed(var)
                )

        # Estimate each subject's marginal likelihood in bounded-memory batches
        log_weight_sum = torch.full((n,), -torch.inf, dtype=working, device=device)
        with torch.no_grad():
            for start in trange(
                0,
                n_importance_samples,
                importance_batch_size,
                desc="Computing importance-sampling likelihood",
                disable=not bool(self.verbose),
            ):
                batch_size = min(importance_batch_size, n_importance_samples - start)
                samples = proposal.sample((batch_size,))
                log_weights = self.sampler.logpdfs_fn(samples) - proposal.log_prob(
                    samples
                )
                log_weight_sum = torch.logaddexp(
                    log_weight_sum, torch.logsumexp(log_weights, dim=0)
                )

        self.loglik_ = (log_weight_sum - log(n_importance_samples)).sum().item()
        self.aic_ = -2 * self.loglik_ + 2 * self.params.numel()
        fim_sign, fim_logdet = torch.linalg.slogdet(self.fim_)
        self.bic_ = (
            -2 * self.loglik_ + fim_logdet.item()
            if fim_sign > 0 and torch.isfinite(fim_logdet)
            else None
        )

        return self
