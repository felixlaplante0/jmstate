"""Dtype and device resolution utilities.

The model parameters own the canonical dtype and device. Data tensors are
aligned to them once in ``prepare`` so that likelihood code performs no
per-call transfers or casts.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = [
    "dtype_device",
]


def dtype_device(module: nn.Module) -> tuple[torch.dtype, torch.device]:
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
