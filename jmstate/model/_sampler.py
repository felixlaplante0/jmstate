from collections.abc import Callable
from typing import Any, Self, cast

import torch

from ..types._data import ModelDataUnchecked
from ..types._parameters import ModelParameters


class MCMCMixin:
    """Mixin for MCMC sampling."""

    params: ModelParameters
    n_chains: int
    init_step_size: float
    adapt_rate: float
    target_accept_rate: float

    def _logpdfs_fn(
        self,
        data: ModelDataUnchecked,
        b: torch.Tensor,
    ) -> torch.Tensor: ...

    def __init__(
        self,
        n_chains: int,
        init_step_size: float,
        adapt_rate: float,
        target_accept_rate: float,
        *args: Any,
        **kwargs: Any,
    ):
        """Initializes the MCMC mixin.

        Args:
            n_chains (int): The number of parallel chains to spawn.
            init_step_size (float): Kernel step in Metropolis-Hastings.
            adapt_rate (float): Adaptation rate for the step_size.
            target_accept_rate (float): Mean acceptance target.
        """
        super().__init__(*args, **kwargs)

        self.n_chains = n_chains
        self.init_step_size = init_step_size
        self.adapt_rate = adapt_rate
        self.target_accept_rate = target_accept_rate

    def _init_sampler(self, data: ModelData):
        """Initializes the MCMC sampler.

        Args:
            data (ModelDataUnchecked): The model data.

        Returns:
            MetropolisWithinGibbsSampler: The initialized MCMC sampler.
        """
        return MetropolisWithinGibbsSampler(
            lambda b: self._logpdfs_fn(data, b),
            torch.zeros(self.n_chains, len(data), self.params.random_prec.dim),
            self.n_chains,
            self.init_step_size,
            self.adapt_rate,
            self.target_accept_rate,
        )


class MetropolisWithinGibbsSampler:
    """A robust Metropolis-within-Gibbs sampler with adaptive step size."""

    logpdfs_fn: Callable[[torch.Tensor], torch.Tensor]
    n_chains: int
    adapt_rate: float
    target_accept_rate: float
    b: torch.Tensor
    logpdfs: torch.Tensor
    j: int
    step_sizes: torch.Tensor
    _noise: torch.Tensor

    def __init__(
        self,
        logpdfs_fn: Callable[[torch.Tensor], torch.Tensor],
        init_b: torch.Tensor,
        n_chains: int,
        init_step_size: float,
        adapt_rate: float,
        target_accept_rate: float,
    ):
        """Initializes the Metropolis-within-Gibbs sampler kernel.

        Args:
            logpdfs_fn (Callable[[torch.Tensor], torch.Tensor]): The log pdfs function.
            init_b (torch.Tensor): Starting b for the chain.
            n_chains (int): The number of parallel chains to spawn.
            init_step_size (float): Kernel step in Metropolis-within-Gibbs.
            adapt_rate (float): Adaptation rate for the step_size.
            target_accept_rate (float): Mean acceptance target.
        """
        self.logpdfs_fn = cast(  # type: ignore
            Callable[[torch.Tensor], torch.Tensor],
            torch.no_grad()(logpdfs_fn),
        )
        self.n_chains = n_chains
        self.adapt_rate = adapt_rate
        self.target_accept_rate = target_accept_rate

        # Set state, log pdfs, and individual parameters
        self.b = init_b
        self.reset()

        # Initialize step sizes and noise
        self.j = 0
        self.step_sizes = torch.full((1, *self.b.shape[1:]), init_step_size)
        self._noise = torch.empty(*self.b.shape[:-1])

    def reset(self) -> Self:
        """Resets the log pdfs and individual parameters.

        Returns:
            Self: The sampler instance.
        """
        self.logpdfs = self.logpdfs_fn(self.b)
        return self

    def step(self) -> Self:
        """Performs a single kernel step.

        Returns:
            Self: The sampler instance.
        """
        # Generate proposal noise
        self._noise.uniform_(-1, 1)

        # Get the proposal
        proposed_state = self.b.clone()
        proposed_state[..., self.j] += self._noise * self.step_sizes[..., self.j]
        proposed_logpdfs = self.logpdfs_fn(proposed_state)
        logpdf_diff = proposed_logpdfs - self.logpdfs

        # Vectorized acceptance decision
        log_uniform = torch.log(torch.rand_like(logpdf_diff))
        accept_mask = log_uniform < logpdf_diff

        torch.where(accept_mask.unsqueeze(-1), proposed_state, self.b, out=self.b)
        torch.where(accept_mask, proposed_logpdfs, self.logpdfs, out=self.logpdfs)

        # Update step sizes
        mean_accept_mask = accept_mask.to(torch.get_default_dtype()).mean(dim=0)
        adaptation = (mean_accept_mask - self.target_accept_rate) * self.adapt_rate
        self.step_sizes[..., self.j] *= torch.exp(adaptation)
        self.j = (self.j + 1) % self.b.size(-1)

        return self

    def run(self, n_steps: int) -> Self:
        """Runs the sampler for a given number of steps.

        Args:
            n_steps (int): The number of steps to run the sampler for.

        Returns:
            Self: The sampler instance.
        """
        for _ in range(n_steps):
            self.step()
        return self
