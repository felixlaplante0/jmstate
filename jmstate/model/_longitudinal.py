from typing import Any

import torch

from ..types._data import ModelDataUnchecked, ModelDesign
from ..types._defs import LOG_TWO_PI
from ..types._parameters import ModelParameters


class LongitudinalMixin:
    """Mixin class for longitudinal model computations."""

    design: ModelDesign
    params: ModelParameters

    def __init__(self, *args: Any, **kwargs: Any):
        """Initializes the longitudinal mixin."""
        super().__init__(*args, **kwargs)

    def _longitudinal_logliks(
        self, data: ModelDataUnchecked, indiv_params: torch.Tensor
    ) -> torch.Tensor:
        """Computes the longitudinal log likelihoods.

        Args:
            data (ModelDataUnchecked): Dataset on which likelihood is computed.
            indiv_params (torch.Tensor): A 3D tensor of individual parameters.

        Returns:
            torch.Tensor: The computed log likelihoods.
        """
        # Careful with NaNs
        predicted = self.design.regression_fn(data.valid_t, indiv_params)
        diff = data.valid_y.addcmul(predicted, data.valid_mask, value=-1.0)

        noise_prec_cholesky, noise_prec_log_eigvals = (
            self.params.noise_prec._prec_cholesky_log_eigvals  # type: ignore
        )
        noise_quad_form = (diff @ noise_prec_cholesky).pow(2).sum(dim=(-2, -1))
        noise_norm_factor = data.n_valid @ (noise_prec_log_eigvals - LOG_TWO_PI)

        return 0.5 * (noise_norm_factor - noise_quad_form)
