import torch


def _tril_from_flat(flat: torch.Tensor, dim: int) -> torch.Tensor:
    """Generates the lower triangular matrix associated with flat tensor.

    Args:
        flat (torch.Tensor): Flat tensor
        dim (int): Dimension of the matrix.

    Returns:
        torch.Tensor: The lower triangular matrix.
    """
    return torch.zeros(dim, dim).index_put_(tuple(torch.tril_indices(dim, dim)), flat)


def _flat_from_tril(L: torch.Tensor) -> torch.Tensor:
    """Flattens the lower triangular part (including the diagonal) of a square matrix.

    Into a 1D tensor, in row-wise order.

    Args:
        L (torch.Tensor): Square lower-triangular matrix of shape (dim, dim).

    Returns:
        torch.Tensor: Flattened 1D tensor containing the lower triangular entries.
    """
    dim = L.size(0)
    return L[tuple(torch.tril_indices(dim, dim))]


def log_cholesky_from_flat(
    flat: torch.Tensor, dim: int, precision_type: str
) -> torch.Tensor:
    """Computes log cholesky from flat tensor according to choice of precision type.

    Args:
        flat (torch.Tensor): The flat tensor parameter.
        dim (int): The dimension of the matrix.
        precision_type (str): The precision type, `'full'`, `'diag'`, or `'spherical'`.

    Raises:
        ValueError: If the precision type is not valid.

    Returns:
        torch.Tensor: The log cholesky representation.
    """
    match precision_type:
        case "full":
            return _tril_from_flat(flat, dim)
        case "diag":
            return torch.diag(flat)
        case "spherical":
            return flat * torch.eye(dim)
        case _:
            raise ValueError(
                "Precision type must be be either 'full', 'diag' or 'spherical', got "
                f"{precision_type}"
            )


def flat_from_log_cholesky(L: torch.Tensor, precision_type: str) -> torch.Tensor:
    """Computes flat tensor from log cholesky according to choice of precision type.

    Args:
        L (torch.Tensor): The square lower triangular matrix parameter.
        precision_type (str): The precision type, `'full'`, `'diag'`, or `'spherical'`.

    Raises:
        ValueError: If the precision type is not valid.

    Returns:
        torch.Tensor: The flat representation.
    """
    match precision_type:
        case "full":
            return _flat_from_tril(L)
        case "diag":
            return L.diag()
        case "spherical":
            return L[0, 0].flatten()
        case _:
            raise ValueError(
                "Precision type must be be either 'full', 'diag' or 'spherical', got "
                f"{precision_type}"
            )
