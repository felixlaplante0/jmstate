from collections.abc import Callable
from functools import cache
from typing import Final

import torch

from ..types._defs import PrecisionType


@cache
def _tril_indices(dim: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Caches the lower triangular indices of a square matrix.

    Args:
        dim (int): Dimension of the matrix.
        device (torch.device): Device of the indices.

    Returns:
        tuple[torch.Tensor, ...]: The row and column indices.
    """
    return tuple(torch.tril_indices(dim, dim, device=device))


def add_jitter(mat: torch.Tensor) -> torch.Tensor:
    """Adds a small relative jitter to the diagonal of a (batched) square matrix.

    Args:
        mat (torch.Tensor): The matrix of shape `(..., dim, dim)`.

    Returns:
        torch.Tensor: The jittered matrix.
    """
    jitter = 1e-6 * mat.diagonal(dim1=-2, dim2=-1).mean().clamp(min=1e-6)
    eye = torch.eye(mat.size(-1), dtype=mat.dtype, device=mat.device)
    return mat + jitter * eye


def _tril_from_flat(flat: torch.Tensor, dim: int) -> torch.Tensor:
    """Generates the lower triangular matrix associated with flat tensor.

    Args:
        flat (torch.Tensor): Flat tensor
        dim (int): Dimension of the matrix.

    Returns:
        torch.Tensor: The lower triangular matrix.
    """
    out = torch.zeros(dim, dim, dtype=flat.dtype, device=flat.device)
    return out.index_put_(_tril_indices(dim, flat.device), flat)


def _flat_from_tril(L: torch.Tensor) -> torch.Tensor:
    """Flattens the lower triangular part (including the diagonal) of a square matrix.

    Into a 1D tensor, in row-wise order.

    Args:
        L (torch.Tensor): Square lower-triangular matrix of shape (dim, dim).

    Returns:
        torch.Tensor: Flattened 1D tensor containing the lower triangular entries.
    """
    dim = L.size(0)
    return L[_tril_indices(dim, L.device)]


_N_ELEMENTS: Final[dict[str, Callable[[int], int]]] = {
    "full": lambda dim: dim * (dim + 1) // 2,
    "diag": lambda dim: dim,
    "spherical": lambda *_: 1,
}

_FROM_FLAT: Final[dict[str, Callable[[torch.Tensor, int], torch.Tensor]]] = {
    "full": _tril_from_flat,
    "diag": lambda flat, *_: torch.diag(flat),
    "spherical": lambda flat, dim: (
        flat * torch.eye(dim, dtype=flat.dtype, device=flat.device)
    ),
}

_TO_FLAT: Final[dict[str, Callable[[torch.Tensor], torch.Tensor]]] = {
    "full": _flat_from_tril,
    "diag": lambda L: L.diag(),
    "spherical": lambda L: L[0, 0].flatten(),
}


def check_precision_type(precision_type: str) -> None:
    """Checks that a precision type is supported.

    Args:
        precision_type (str): The precision type.

    Raises:
        ValueError: If the precision type is not valid.
    """
    if precision_type not in _FROM_FLAT:
        allowed = ", ".join(repr(key) for key in _FROM_FLAT)
        raise ValueError(
            f"Precision type must be one of {allowed}, got {precision_type!r}"
        )


def check_matrix_dim(flat: torch.Tensor, dim: int, precision_type: str) -> None:
    """Checks dimensions for matrix according to precision type.

    Args:
        flat (torch.Tensor): The flat tensor.
        dim (int): The dimension of the matrix.
        precision_type (str): The precision type.

    Raises:
        ValueError: If the precision type is not valid.
        ValueError: If the number of elements is incompatible with the precision type.
    """
    check_precision_type(precision_type)
    expected = _N_ELEMENTS[precision_type](dim)
    if flat.numel() != expected:
        raise ValueError(
            f"{flat.numel()} elements are incompatible with precision type "
            f"{precision_type!r} of dimension {dim} (expected {expected})"
        )


def log_cholesky_from_flat(
    flat: torch.Tensor, dim: int, precision_type: PrecisionType
) -> torch.Tensor:
    """Computes the log-Cholesky factor from the flat tensor.

    Args:
        flat (torch.Tensor): The flat tensor parameter.
        dim (int): The dimension of the matrix.
        precision_type (PrecisionType): The precision type, ``"full"``, ``"diag"`` or
            ``"spherical"``.

    Raises:
        ValueError: If the precision type is not valid.

    Returns:
        torch.Tensor: The log-Cholesky representation.
    """
    check_precision_type(precision_type)
    return _FROM_FLAT[precision_type](flat, dim)


def flat_from_log_cholesky(
    L: torch.Tensor, precision_type: PrecisionType
) -> torch.Tensor:
    """Computes the flat tensor from the log-Cholesky factor.

    Args:
        L (torch.Tensor): The square lower-triangular matrix parameter.
        precision_type (PrecisionType): The precision type, ``"full"``, ``"diag"`` or
            ``"spherical"``.

    Raises:
        ValueError: If the precision type is not valid.

    Returns:
        torch.Tensor: The flat representation.
    """
    check_precision_type(precision_type)
    return _TO_FLAT[precision_type](L)
