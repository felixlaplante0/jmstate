"""Dtype and device resolution utilities."""

import torch
from torch import nn

__all__ = [
    "dtype_device",
    "model_dtype",
]


def model_dtype(dtype: torch.dtype | None = None) -> torch.dtype:
    """Resolves a supported model dtype.

    When ``dtype`` is provided it is validated against the supported set. When
    omitted, ``torch``'s default dtype is restricted to a supported one:
    ``float64`` is preserved and any other dtype selects ``float32``.

    Args:
        dtype (torch.dtype | None, optional): The dtype to validate. Defaults to
            None, in which case ``torch.get_default_dtype()`` is restricted.

    Raises:
        ValueError: If ``dtype`` is provided and is neither ``torch.float32`` nor
            ``torch.float64``.

    Returns:
        torch.dtype: ``torch.float32`` or ``torch.float64``.
    """
    if dtype is None:
        default = torch.get_default_dtype()
        return torch.float64 if default == torch.float64 else torch.float32
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"Only torch.float32 and torch.float64 are supported, got {dtype!r}"
        )
    return dtype


def dtype_device(module: nn.Module) -> tuple[torch.dtype, torch.device]:
    """Gets the canonical dtype and device from a module's tensors.

    Args:
        module (nn.Module): Module holding the reference tensors (typically
            ``ModelParameters``).

    Raises:
        ValueError: If the first floating-point parameter has a dtype other than
            ``torch.float32`` or ``torch.float64``.

    Returns:
        tuple[torch.dtype, torch.device]: The first floating-point parameter's
            validated dtype and device, else the default dtype on CPU.
    """
    for param in module.parameters(recurse=True):
        if param.is_floating_point():
            return model_dtype(param.dtype), param.device
    return model_dtype(), torch.device("cpu")
