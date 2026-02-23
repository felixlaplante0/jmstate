import itertools

import torch

from ..types._defs import Trajectory


def check_trajectories(trajectories: list[Trajectory], c: torch.Tensor | None):
    """Check if trajectories are not empty, well sorted and compatible with censoring.

    Args:
        trajectories (list[Trajectory]): The trajectories.
        c (torch.Tensor | None): The censoring times.

    Raises:
        ValueError: If some trajectory is empty.
        ValueError: If some trajectory is not sorted.
        ValueError: If some trajectory is not compatible with the censoring times.
    """
    if any(len(trajectory) == 0 for trajectory in trajectories):
        raise ValueError("Trajectories must not be empty")
    if any(
        not all(t0 <= t1 for t0, t1 in itertools.pairwise(t for t, _ in trajectory))
        for trajectory in trajectories
    ):
        raise ValueError(
            "Trajectories must be sorted by time, in ascending order. Also ensure "
            "there are no NaN values as this will trigger the check"
        )
    if c is not None and any(
        trajectory[-1][0] > c for trajectory, c in zip(trajectories, c, strict=True)
    ):
        raise ValueError("Transitions times may not be greater than censoring times")


def check_matrix_dim(flat: torch.Tensor, dim: int, precision_type: str):
    """Checks dimensions for matrix according to precision type.

    Args:
        flat (torch.Tensor): The flat tensor.
        dim (int): The dimension of the matrix.
        precision_type (str): The precision type.

    Raises:
        ValueError: If the number of elements is incompatible with precision type
            `'full'`.
        ValueError: If the number of elements is incompatible with precision type
            `'diag'`.
        ValueError: If the number of elements is not one and the precision type is
            `'spherical'`.
        ValueError: If the precision type is not valid.
    """
    match precision_type:
        case "full":
            if flat.numel() != (dim * (dim + 1)) // 2:
                raise ValueError(
                    f"{flat.numel()} is incompatible with full matrix of dimension "
                    f"{dim}"
                )
        case "diag":
            if flat.numel() != dim:
                raise ValueError(
                    f"{flat.numel()} is incompatible with diagonal matrix of dimension "
                    f"{dim}"
                )
        case "spherical":
            if flat.numel() != 1:
                raise ValueError(
                    f"Expected 1 element for flat, got {flat.numel()} for scalar "
                    f"marix of dimension {dim}"
                )
        case _:
            raise ValueError(
                f"Precision type must be be either 'full', 'diag' or 'spherical', got "
                f"{precision_type}"
            )
