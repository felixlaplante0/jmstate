"""Dtype and device resolution utilities.

The model parameters own the canonical dtype and device. Data tensors are
aligned to them once in ``prepare`` so that likelihood code performs no
per-call transfers or casts. Time-like tensors are kept in at least
``float32`` (mixed precision): ``bfloat16``/``float16`` parameters are
supported, but quadrature, Cholesky factors and survival integrals run in
``float32`` or higher for numerical stability.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = [
    "canonical_dtype_device",
    "resolve_dtype",
]

#: Minimum floating dtype used for time-like tensors and critical kernels.
MIN_FLOAT_DTYPE: torch.dtype = torch.float32


def canonical_dtype_device(module: nn.Module) -> tuple[torch.dtype, torch.device]:
    """Gets the canonical dtype and device from a module's tensors.

    Args:
        module (nn.Module): Module holding the reference tensors (typically
            ``ModelParameters``).

    Returns:
        tuple[torch.dtype, torch.device]: First floating-point parameter's
            dtype and device, else the default dtype on CPU.
    """
    for param in module.parameters(recurse=True):
        if param.is_floating_point():
            return param.dtype, param.device
    return torch.get_default_dtype(), torch.device("cpu")


def resolve_dtype(
    *dtypes: torch.dtype, floor: torch.dtype = MIN_FLOAT_DTYPE
) -> torch.dtype:
    """Promotes dtypes without ever losing precision.

    With a single argument this is also the kernel dtype: ``torch.float64``
    stays, anything else runs in at least ``floor``.

    Args:
        *dtypes (torch.dtype): Candidate dtypes (e.g. parameter and data
            dtypes).
        floor (torch.dtype, optional): Minimum floating dtype kept. Defaults
            to ``torch.float32`` so low-precision setups still store times
            and run critical kernels accurately.

    Returns:
        torch.dtype: The promoted dtype, raised to ``floor`` if floating.
    """
    out = dtypes[0]
    for dtype in dtypes[1:]:
        out = torch.promote_types(out, dtype)
    if out.is_floating_point and torch.finfo(out).bits < torch.finfo(floor).bits:
        return floor
    return out
