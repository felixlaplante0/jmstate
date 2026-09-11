from collections.abc import Iterator
from math import ceil
from numbers import Integral
from typing import Any, cast

import torch
from sklearn.utils._param_validation import Interval, validate_params  # type: ignore
from sklearn.utils.validation import (  # type: ignore
    check_consistent_length,  # type: ignore
)
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import trange

from ..types._data import (
    ModelData,
    ModelDataUnchecked,
    ModelDesign,
    SampleDataUnchecked,
)
from ..types._defs import Trajectory
from ..types._parameters import ModelParameters
from ..utils._checks import check_finite
from ._hazard import HazardMixin
from ._sampler import MCMCMixin


class PredictMixin(HazardMixin, MCMCMixin):
    """Mixin class for prediction."""

    design: ModelDesign
    params: ModelParameters
    n_chains: int
    n_warmup: int
    n_subsample: int
    verbose: bool | int
    fim_: torch.Tensor | None

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the prediction mixin."""
        super().__init__(*args, **kwargs)

    def _sample_params(self, sample_size: int) -> torch.Tensor:
        """Sample model parameters based on asymptotic behavior of the MLE.

        This uses Bernstein-von Mises theorem to approximate the posterior distribution
        of the parameters as a multivariate normal distribution with mean equal to the
        MLE and covariance matrix equal to the inverse of the Fisher Information Matrix.

        Args:
            sample_size (int): The desired sample size.

        Returns:
            torch.Tensor: A tensor of sampled model parameters as vectors.
        """
        loc = parameters_to_vector(self.params.parameters()).detach()
        try:
            dist = torch.distributions.MultivariateNormal(
                loc=loc, precision_matrix=cast(torch.Tensor, self.fim_)
            )
        except ValueError:
            # Fisher information may be singular; add a small relative jitter
            fim = cast(torch.Tensor, self.fim_)
            jitter = 1e-6 * fim.diagonal().mean().clamp(min=1e-6)
            dist = torch.distributions.MultivariateNormal(
                loc=loc,
                precision_matrix=fim
                + jitter * torch.eye(fim.size(0), dtype=fim.dtype, device=fim.device),
            )
        return dist.sample((sample_size,))

    def _posterior_draws(
        self,
        data: ModelDataUnchecked,
        *,
        n_samples: int,
        double_monte_carlo: bool,
        desc: str,
    ) -> Iterator[torch.Tensor]:
        """Yields posterior draws of individual parameters in chain batches.

        Owns the sampling loop shared by all prediction methods: parameter
        sampling for the double Monte Carlo procedure, MCMC warmup and
        subsampling, and restoration of the fitted parameters afterwards.

        Args:
            data (ModelDataUnchecked): Prepared dataset.
            n_samples (int): Number of posterior samples to draw.
            double_monte_carlo (bool): Whether to sample model parameters.
            desc (str): Progress bar description.

        Raises:
            ValueError: If `double_monte_carlo` is True and summary statistics
                have not been computed.

        Yields:
            torch.Tensor: Individual parameters of shape `(n_chains, n, l)`.
        """
        n_iter = ceil(n_samples / self.n_chains)

        init_params = None
        sampled_params = None
        if double_monte_carlo:
            if self.fim_ is None:
                raise ValueError(
                    "Double Monte Carlo requires summary statistics. "
                    "Call compute_summary() first."
                )
            init_params = (
                parameters_to_vector(self.params.parameters()).detach().clone()
            )
            sampled_params = self._sample_params(n_iter)

        sampler = self._init_sampler(data)
        if not double_monte_carlo:
            sampler.run(self.n_warmup)

        try:
            for i in trange(n_iter, desc=desc, disable=not bool(self.verbose)):
                if double_monte_carlo:
                    vector_to_parameters(sampled_params[i], self.params.parameters())  # type: ignore
                    sampler = self._init_sampler(data).run(self.n_warmup)

                yield self.design.indiv_params_fn(
                    self.params.fixed_effects, data.x, sampler.b
                )

                if not double_monte_carlo:
                    sampler.run(self.n_subsample)
        finally:
            if init_params is not None:
                vector_to_parameters(init_params, self.params.parameters())  # type: ignore

    @torch.no_grad()  # type: ignore
    @validate_params(
        {
            "data": [ModelData],
            "u": [torch.Tensor],
            "n_samples": [Interval(Integral, 1, None, closed="left")],
            "double_monte_carlo": [bool],
        },
        prefer_skip_nested_validation=True,
    )
    def predict_y(
        self,
        data: ModelData,
        u: torch.Tensor,
        *,
        n_samples: int = 1000,
        double_monte_carlo: bool = False,
    ) -> torch.Tensor:
        r"""Predict longitudinal measurements at specified times.

        Computes the predicted longitudinal responses for each individual at the
        specified prediction times :math:`u` by evaluating the regression function
        conditional on posterior draws of the random effects :math:`b`. The prediction
        may optionally use a double Monte Carlo procedure to account for parameter
        uncertainty following Rizopoulos (2011).

        The input `u` must be a matrix of shape :math:`(n, m)` where :math:`n` is the
        number of individuals and :math:`m` is the number of prediction time points.

        Args:
            data (ModelData): The dataset containing covariates, observed outcomes,
                trajectories, and censoring information.
            u (torch.Tensor): Matrix of prediction times of shape `(n, m)`.
            n_samples (int, optional): Number of posterior samples to draw. Defaults to
                1000.
            double_monte_carlo (bool, optional): If True, predictions incorporate a
                double Monte Carlo procedure to sample parameters. Defaults to False.

        Raises:
            ValueError: If `double_monte_carlo` is True and the model has not been
                fitted.
            ValueError: If `u` contains NaN or infinite values.
            ValueError: If `u` has a shape inconsistent with the number of individuals.

        Returns:
            torch.Tensor: Predicted longitudinal outcomes of shape `(n_samples, n, m)`,
                where predictions are stacked along the first dimension.
        """
        check_finite(u, "u")
        check_consistent_length(u, data)

        # Load and complete data
        data = ModelDataUnchecked(
            data.x, data.t, data.y, data.trajectories, data.c
        ).prepare(self)
        u = u.to(dtype=data.t.dtype, device=data.t.device)

        y_pred: list[torch.Tensor] = []
        draws = self._posterior_draws(
            data,
            n_samples=n_samples,
            double_monte_carlo=double_monte_carlo,
            desc="Predicting longitudinal values",
        )
        for indiv_params in draws:
            y = self.design.regression_fn(u, indiv_params)
            y_pred.extend(y[j] for j in range(y.size(0)))

        return torch.stack(y_pred[:n_samples])

    @torch.no_grad()  # type: ignore
    @validate_params(
        {
            "data": [ModelData],
            "u": [torch.Tensor],
            "n_samples": [Interval(Integral, 1, None, closed="left")],
            "double_monte_carlo": [bool],
        },
        prefer_skip_nested_validation=True,
    )
    def predict_surv_logps(
        self,
        data: ModelData,
        u: torch.Tensor,
        *,
        n_samples: int = 1000,
        double_monte_carlo: bool = False,
    ) -> torch.Tensor:
        r"""Predict survival log-probabilities at specified times.

        Computes the predicted log survival probabilities for each individual at the
        specified prediction times :math:`u` by evaluating the survival function
        conditional on posterior draws of the random effects :math:`b`. The computation
        may optionally use a double Monte Carlo procedure to incorporate parameter
        uncertainty following Rizopoulos (2011).

        The predicted quantity is given by:

        .. math::
            \log \mathbb{P}(T^* \geq u \mid T^* > c) = -\int_c^u \lambda(t) \, dt,

        where :math:`c` denotes the individual censoring time. In the presence of
        multiple transitions, :math:`\lambda(t)` is the sum over all possible
        transition-specific hazards:

        .. math::
            -\int_c^u \sum_{k'} \lambda^{k' \mid k}(t \mid t_0) \, dt,

        using the Chasles property to simplify computation and improve numerical
        precision.

        The input `u` must be a matrix of shape :math:`(n, m)` where :math:`n` is the
        number of individuals and :math:`m` is the number of prediction time points.

        Args:
            data (ModelData): The dataset containing covariates, observed outcomes,
                trajectories, and censoring information.
            u (torch.Tensor): Matrix of prediction times of shape `(n, m)`.
            n_samples (int, optional): Number of posterior samples to draw. Defaults to
                1000.
            double_monte_carlo (bool, optional): If True, predictions incorporate a
                double Monte Carlo procedure to sample parameters. Defaults to False.

        Raises:
            ValueError: If `double_monte_carlo` is True and the model has not been
                fitted.
            ValueError: If `u` contains NaN or infinite values.
            ValueError: If `u` has a shape inconsistent with the number of individuals.

        Returns:
            torch.Tensor: Predicted survival log-probabilities of shape `(n_samples, n,
            m)`, stacked along the first dimension.
        """
        check_finite(u, "u")
        u = torch.broadcast_to(u, (len(data), -1))

        # Load and complete data
        data = ModelDataUnchecked(
            data.x, data.t, data.y, data.trajectories, data.c
        ).prepare(self)
        u = u.to(dtype=data.t.dtype, device=data.t.device)

        surv_logps_pred: list[torch.Tensor] = []
        draws = self._posterior_draws(
            data,
            n_samples=n_samples,
            double_monte_carlo=double_monte_carlo,
            desc="Predicting survival log probabilities",
        )
        for indiv_params in draws:
            sample_data = SampleDataUnchecked(
                data.x, data.trajectories, indiv_params, data.c
            )
            surv_logps = self.compute_surv_logps(sample_data, u)
            surv_logps_pred.extend(surv_logps[j] for j in range(surv_logps.size(0)))

        return torch.stack(surv_logps_pred[:n_samples])

    @torch.no_grad()  # type: ignore
    @validate_params(
        {
            "data": [ModelData],
            "c": [torch.Tensor],
            "max_length": [Interval(Integral, 1, None, closed="left")],
            "n_samples": [Interval(Integral, 1, None, closed="left")],
            "double_monte_carlo": [bool],
        },
        prefer_skip_nested_validation=True,
    )
    def predict_trajectories(
        self,
        data: ModelData,
        c: torch.Tensor,
        *,
        max_length: int = 10,
        n_samples: int = 1000,
        double_monte_carlo: bool = False,
    ) -> list[list[Trajectory]]:
        r"""Predict individual-level trajectories up to specified censoring times.

        Simulates the evolution of individual trajectories conditional on posterior
        draws of the random effects :math:`b`, up to the censoring times `c`.
        Trajectories are truncated to a maximum length of `max_length` to avoid infinite
        loops. The simulation algorithm is a variant of Gillespie's method adapted for
        individual parameters. If `double_monte_carlo` is True, then the prediction is
        computed using the double Monte Carlo procedure described in Rizopoulos (2011).

        The input `c` must be a column vector of shape :math:`(n, 1)` where :math:`n` is
        the number of individuals.

        Args:
            data (ModelData): The dataset containing covariates, observed outcomes,
                trajectories, and censoring information.
            c (torch.Tensor): Column vector of censoring times for each individual.
            max_length (int, optional): Maximum length of generated trajectories.
                Defaults to 10.
            n_samples (int, optional): Number of posterior samples to draw. Defaults to
                1000.
            double_monte_carlo (bool, optional): If True, predictions incorporate a
                double Monte Carlo procedure to sample parameters. Defaults to False.

        Raises:
            ValueError: If `double_monte_carlo` is True and the model has not been
                fitted.
            ValueError: If `c` contains NaN or infinite values.
            ValueError: If `c` has a shape inconsistent with the number of individuals.

        Returns:
            list[list[Trajectory]]: Predicted trajectories for each individual,
            organized as a list of lists, with the outer list indexing posterior draws
            and the inner list indexing individuals.
        """
        check_finite(c, "c")
        check_consistent_length(c, data)

        # Load and complete data
        data = ModelDataUnchecked(
            data.x, data.t, data.y, data.trajectories, data.c
        ).prepare(self)
        c = c.to(dtype=data.c.dtype, device=data.c.device)

        trajectories_pred: list[list[Trajectory]] = []
        draws = self._posterior_draws(
            data,
            n_samples=n_samples,
            double_monte_carlo=double_monte_carlo,
            desc="Predicting trajectories",
        )
        for indiv_params in draws:
            sample_data = SampleDataUnchecked(
                data.x, data.trajectories, indiv_params, data.c
            )
            trajectories_pred.extend(
                self.sample_trajectories(sample_data, c, max_length=max_length)
            )

        return trajectories_pred[:n_samples]
