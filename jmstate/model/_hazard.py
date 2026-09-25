from math import isfinite
from numbers import Integral
from typing import Any

import torch
from sklearn.utils._param_validation import Interval, validate_params  # type: ignore
from sklearn.utils.validation import (  #  type: ignore
    check_consistent_length,  #  type: ignore
)

from ..types._data import (
    ModelDataUnchecked,
    ModelDesign,
    SampleData,
    SampleDataUnchecked,
)
from ..types._defs import LOG_CLAMP, Trajectory
from ..types._parameters import ModelParameters
from ..utils._checks import check_finite
from ..utils._surv import _host_times, _quad_tensors, build_remaining_buckets
from ..utils.dtype import dtype_device


class HazardMixin:
    """Mixin class for hazard model computations."""

    design: ModelDesign
    params: ModelParameters
    n_quad: int
    n_bisect: int

    def __init__(
        self,
        n_quad: int,
        n_bisect: int,
        *args: Any,
        **kwargs: Any,
    ):
        """Initializes the hazard mixin.

        Args:
            n_quad (int): Number of quadrature nodes.
            n_bisect (int): The number of bisection steps.
            *args (Any): Positional arguments forwarded to the next mixin.
            **kwargs (Any): Keyword arguments forwarded to the next mixin.
        """
        super().__init__(*args, **kwargs)

        self.n_quad = n_quad
        self.n_bisect = n_bisect

    def _align_sample_data(
        self, sample_data: SampleData
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Moves sample tensors to the model dtype/device in one call each.

        Args:
            sample_data (SampleData): The sampling data.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: Aligned
                covariates, individual parameters and conditioning times.
        """
        dtype, device = dtype_device(self.params)
        x, indiv_params, t_cond = (
            None if tensor is None else tensor.to(dtype=dtype, device=device)
            for tensor in (sample_data.x, sample_data.indiv_params, sample_data.t_cond)
        )
        return x, indiv_params, t_cond  # type: ignore

    def _log_hazard(
        self,
        key: tuple[Any, Any],
        t0: torch.Tensor,
        t1: torch.Tensor,
        x: torch.Tensor,
        indiv_params: torch.Tensor,
    ) -> torch.Tensor:
        """Computes log hazard.

        Args:
            key (tuple[Any, Any]): The transition key.
            t0 (torch.Tensor): A column vector of previous transition times.
            t1 (torch.Tensor): A matrix of next transition times.
            x (torch.Tensor): The fixed covariates.
            indiv_params (torch.Tensor): The individual parameters.

        Returns:
            torch.Tensor: The computed log hazard.
        """
        str_key = str(key)

        # Compute baseline hazard
        base = self.params.base_hazards[str_key](t0, t1)

        # Compute time-varying effects (weights follow activations)
        link_out = self.design.link_fns[key](t1, indiv_params)
        mod = link_out @ self.params.link_coefs[str_key].to(link_out.dtype)

        # Compute covariates effect (if any)
        var = 0
        if str_key in self.params.x_coefs:
            var = x @ self.params.x_coefs[str_key].to(x.dtype).unsqueeze(-1)

        return (base + mod + var).reshape((*indiv_params.shape[:-1], -1))

    def _cum_hazard(
        self,
        key: tuple[Any, Any],
        t0: torch.Tensor,
        t1: torch.Tensor,
        x: torch.Tensor,
        indiv_params: torch.Tensor,
    ) -> torch.Tensor:
        """Computes cumulative hazard.

        Args:
            key (tuple[Any, Any]): The transition key.
            t0 (torch.Tensor): A column vector of previous transition times.
            t1 (torch.Tensor): A matrix of next transition times.
            x (torch.Tensor): The fixed covariates.
            indiv_params (torch.Tensor): The individual parameters.

        Returns:
            torch.Tensor: The computed cumulative hazard.
        """
        # Careful with negative times
        t1 = torch.max(t0, t1)

        # Transform to quadrature interval
        nodes, weights = _quad_tensors(self.n_quad, t1.dtype, t1.device)
        half = 0.5 * (t1 - t0)
        quad = (0.5 * (t0 + t1).unsqueeze(-1) + half.unsqueeze(-1) * nodes).flatten(
            start_dim=-2
        )

        # Compute hazard at quadrature points
        vals = (
            self._log_hazard(key, t0, quad, x, indiv_params).clamp(max=LOG_CLAMP).exp()
        )

        return half * (vals.unflatten(-1, (-1, weights.size(-1))) @ weights)

    def _hazard_logliks(
        self, data: ModelDataUnchecked, indiv_params: torch.Tensor
    ) -> torch.Tensor:
        """Computes the hazard log likelihoods.

        Args:
            data (ModelDataUnchecked): Dataset on which likelihood is computed.
            indiv_params (torch.Tensor): A matrix of individual parameters.

        Returns:
            torch.Tensor: The computed log likelihoods.
        """
        logliks = torch.zeros(
            indiv_params.shape[:-1],
            dtype=indiv_params.dtype,
            device=indiv_params.device,
        )

        _nodes, weights = _quad_tensors(
            self.n_quad, indiv_params.dtype, indiv_params.device
        )
        for key, (idxs, t0, obs, half, quad) in data.quad_buckets.items():
            vals = self._log_hazard(
                key, t0, quad, data.x[idxs], indiv_params[..., idxs, :]
            )
            vals[..., 1:].clamp_(max=LOG_CLAMP).exp_()

            # Compute log likelihoods and scatter add
            obs_logliks = vals[..., 0]
            alts_logliks = half.flatten() * (vals[..., 1:] @ weights)
            logliks.index_add_(-1, idxs, (-alts_logliks).addcmul(obs, obs_logliks))

        return logliks

    @torch.no_grad()  # type: ignore
    @validate_params(
        {
            "sample_data": [SampleData],
            "u": [torch.Tensor],
        },
        prefer_skip_nested_validation=True,
    )
    def compute_surv_logps(
        self, sample_data: SampleData, u: torch.Tensor
    ) -> torch.Tensor:
        r"""Compute log survival probabilities at specified times.

        Evaluates the log-probability of remaining event-free up to the prediction times
        :math:`u` conditional on individual-level parameters and censoring times. The
        computation uses the hazard function :math:`\lambda(t)` to obtain:

        .. math::
            \log \mathbb{P}(T^* \geq u \mid T^* > c) = -\int_c^u \lambda(t) \, dt,

        where :math:`c` denotes the censoring time for each individual. In cases with
        multiple possible transitions, :math:`\lambda(t)` sums over all
        transition-specific hazards:

        .. math::
            -\int_c^u \sum_{k'} \lambda^{k' \mid k}(t \mid t_0) \, dt,

        exploiting the Chasles property to simplify computation and improve numerical
        precision.

        The input `u` must be a matrix of shape :math:`(n, m)` where :math:`n` is the
        number of individuals and :math:`m` is the number of prediction time points.

        Args:
            sample_data (SampleData): The dataset containing covariates,
                individual-level parameters, trajectories, and censoring information.
            u (torch.Tensor): Matrix of evaluation times of shape `(n, m)`.

        Raises:
            ValueError: If `u` contains NaN or infinite values.
            ValueError: If `u` has a shape inconsistent with the number of individuals.

        Returns:
            torch.Tensor: Computed survival log-probabilities of shape `(n, m)`, with
            rows corresponding to individuals and columns to prediction times.
        """
        check_finite(u, "u")
        u = torch.broadcast_to(u, (len(sample_data), -1))

        x, indiv_params, t_cond = self._align_sample_data(sample_data)
        u = u.to(dtype=x.dtype, device=x.device)

        # Get buckets from last states
        buckets = build_remaining_buckets(
            self, sample_data.trajectories, u.max(dim=1).values
        )
        return self._surv_logps(buckets, x, indiv_params, t_cond, u)

    def _surv_logps(
        self,
        buckets: dict[tuple[Any, Any], tuple[torch.Tensor, ...]],
        x: torch.Tensor,
        indiv_params: torch.Tensor,
        t_cond: torch.Tensor | None,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Computes log survival probabilities from prebuilt buckets.

        Args:
            buckets (dict[tuple[Any, Any], tuple[torch.Tensor, ...]]): Remaining
                buckets built from the trajectories.
            x (torch.Tensor): The aligned covariates.
            indiv_params (torch.Tensor): The aligned individual parameters.
            t_cond (torch.Tensor | None): The aligned conditioning times.
            u (torch.Tensor): The aligned evaluation times of shape `(n, m)`.

        Returns:
            torch.Tensor: Computed survival log-probabilities.
        """
        # Compute the log probabilities summing over transitions
        nlogps = torch.zeros(
            *indiv_params.shape[:-1],
            u.size(1),
            dtype=x.dtype,
            device=x.device,
        )
        for key, (idxs, t0, _t1) in buckets.items():
            # Compute negative log survival and scatter add
            t0 = t0 if t_cond is None else t_cond[idxs]  # noqa: PLW2901
            alts_logliks = self._cum_hazard(
                key, t0, u[idxs], x[idxs], indiv_params[..., idxs, :]
            )
            nlogps.index_add_(-2, idxs, alts_logliks)

        return -nlogps.clamp(min=0.0)

    def _sample_transition(
        self,
        key: tuple[Any, Any],
        t0: torch.Tensor,
        t1: torch.Tensor,
        x: torch.Tensor,
        indiv_params: torch.Tensor,
    ) -> torch.Tensor:
        """Sample survival times using inverse transform sampling.

        Args:
            key (tuple[Any, Any]): The transition key.
            t0 (torch.Tensor): The start times.
            t1 (torch.Tensor): The end times.
            x (torch.Tensor): The covariates.
            indiv_params (torch.Tensor): The individual parameters.

        Returns:
            torch.Tensor: The computed pre transition times.
        """
        # Initialize for bisection search
        t_left, t_right = t0.clone(), torch.nextafter(t1, t1.new_tensor(float("inf")))

        # Generate exponential random variables
        target = -torch.log(torch.rand_like(t_left))

        # Bisection search for survival times
        for _ in range(self.n_bisect):
            t_mid = 0.5 * (t_left + t_right)
            surv_nlogps = self._cum_hazard(key, t0, t_mid, x, indiv_params)

            # Update search bounds
            accept_mask = surv_nlogps.reshape(target.shape) < target
            torch.where(accept_mask, t_mid, t_left, out=t_left)
            torch.where(accept_mask, t_right, t_mid, out=t_right)

        return t_right

    def _sample_trajectory_step(
        self,
        sample_data: SampleData,
        c: torch.Tensor,
        *,
        censoring: list[float] | None = None,
    ) -> bool:
        """Appends the next simulated transition.

        Args:
            sample_data (SampleData): Sampling data aligned to the model.
            c (torch.Tensor): Sampling censoring time aligned to the model.
            censoring (list[float] | None, optional): Host censoring times to
                reuse. Defaults to None.

        Returns:
            bool: True if the sampling is done.
        """
        x, indiv_params, t_cond = (
            sample_data.x,
            sample_data.indiv_params,
            sample_data.t_cond,
        )

        # Get buckets from last states
        current_buckets = build_remaining_buckets(
            self, sample_data.trajectories, c, censoring=censoring
        )

        if not current_buckets:
            return True

        # Initialize candidate transition times
        n_transitions = len(current_buckets)
        t_candidates = torch.full(
            (len(sample_data), n_transitions),
            float("inf"),
            dtype=x.dtype,
            device=x.device,
        )

        for j, (key, (idxs, t0, t1)) in enumerate(current_buckets.items()):
            # Sample transition times, and condition with c
            t0 = t0 if t_cond is None else t_cond[idxs]  # noqa: PLW2901
            t_sample = self._sample_transition(
                key, t0, t1, x[idxs], indiv_params[..., idxs, :]
            )
            t_candidates[idxs, j] = t_sample.flatten()

        # Find earliest transition (single host transfer instead of one per row)
        min_times, argmin_idxs = torch.min(t_candidates, dim=1)
        dest_states = [key[1] for key in current_buckets]
        for trajectory, time, arg_idx in zip(
            sample_data.trajectories,
            min_times.tolist(),
            argmin_idxs.tolist(),
            strict=True,
        ):
            if isfinite(time):
                trajectory.append((time, dest_states[arg_idx]))

        return False

    @torch.no_grad()  # type: ignore
    @validate_params(
        {
            "sample_data": [SampleData],
            "c": [torch.Tensor],
            "max_length": [Interval(Integral, 1, None, closed="left")],
        },
        prefer_skip_nested_validation=True,
    )
    def sample_trajectories(
        self,
        sample_data: SampleData,
        c: torch.Tensor,
        *,
        max_length: int = 10,
    ) -> list[Trajectory] | list[list[Trajectory]]:
        r"""Simulate individual trajectories from the multistate joint model.

        Generates sample trajectories for each individual up to the censoring times `c`,
        truncating to a maximum of `max_length` transitions to prevent infinite loops.
        The simulation employs a variant of Gillespie's algorithm adapted for individual
        parameter draws. These sampled trajectories form the basis for posterior
        predictive checks or downstream predictions in the joint model framework.

        The input `c` must be a column vector of shape :math:`(n, 1)` where :math:`n` is
        the number of individuals.

        If ``indiv_params`` has a leading chain dimension ``(C, n, l)``, sampling runs
        vectorized over the ``C * n`` chain-major rows and returns one trajectory list
        per chain.

        Args:
            sample_data (SampleData): The dataset containing covariates, trajectories,
                and individual-level parameter used for sampling.
            c (torch.Tensor): Column vector of censoring times for each individual.
            max_length (int, optional): Maximum number of iterations or transitions
                sampled per trajectory. Defaults to 10.

        Raises:
            ValueError: If `c` contains NaN or infinite values.
            ValueError: If `c` has a shape inconsistent with the number of individuals.

        Returns:
            list[Trajectory] | list[list[Trajectory]]: Sampled trajectories, one per
                individual, or one list per chain for 3D individual parameters. Each
                trajectory is truncated at the censoring time.
        """
        check_finite(c, "c")
        check_consistent_length(c, sample_data)
        return self._sample_trajectories(sample_data, c, max_length)

    def _sample_trajectories(
        self, sample_data: SampleData, c: torch.Tensor, max_length: int
    ) -> list[Trajectory] | list[list[Trajectory]]:
        """Simulates trajectories from already validated inputs.

        Args:
            sample_data (SampleData): The sampling data.
            c (torch.Tensor): Column vector of censoring times.
            max_length (int): Maximum number of transitions sampled.

        Returns:
            list[Trajectory] | list[list[Trajectory]]: Sampled trajectories, one
                list per chain for 3D individual parameters.
        """
        n = len(sample_data)
        leading = sample_data.indiv_params.shape[:-2]
        n_chains = leading[0] if leading else 1

        def _rep(tensor: torch.Tensor) -> torch.Tensor:
            return (
                tensor.unsqueeze(0)
                .expand(n_chains, *tensor.shape)
                .reshape(n_chains * n, *tensor.shape[1:])
            )

        # Chain-major flattened copies to vectorize over chains and individuals
        trajectories_flat = [
            trajectory.copy()
            for _ in range(n_chains)
            for trajectory in sample_data.trajectories
        ]
        flat = SampleDataUnchecked(
            _rep(sample_data.x),
            trajectories_flat,
            sample_data.indiv_params.reshape(n_chains * n, -1),
            None if sample_data.t_cond is None else _rep(sample_data.t_cond),
        )
        flat.x, flat.indiv_params, flat.t_cond = self._align_sample_data(flat)
        c_flat = _rep(c)
        c_model = c_flat.to(dtype=flat.x.dtype, device=flat.x.device)
        censoring = _host_times(c_flat)

        # Sample future transitions iteratively
        for _ in range(max_length):
            if self._sample_trajectory_step(flat, c_model, censoring=censoring):
                break
            flat.t_cond = None

        # Compare in the dtype of c with a single host transfer
        last_times = torch.tensor(
            [trajectory[-1][0] for trajectory in trajectories_flat],
            dtype=c.dtype if c.is_floating_point() else torch.get_default_dtype(),
        )
        keep = (last_times <= c_flat.reshape(-1).cpu()).tolist()
        simulated = [
            trajectory if kept else trajectory[:-1]
            for trajectory, kept in zip(trajectories_flat, keep, strict=True)
        ]
        if not leading:
            return simulated
        return [simulated[k * n : (k + 1) * n] for k in range(n_chains)]
