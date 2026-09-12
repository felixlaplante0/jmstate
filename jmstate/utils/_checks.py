import itertools

import torch

from ..types._defs import CENSORING_TOLERANCE, Trajectory


def check_finite(
    value: torch.Tensor | None, input_name: str, *, allow_nan: bool = False
):
    """Checks that a tensor contains no NaN or infinite values.

    Unlike scikit-learn's ``assert_all_finite``, this works on any device
    (CPU, CUDA, XPU) without host transfers.

    Args:
        value (torch.Tensor | None): The tensor to check, or None to skip.
        input_name (str): Name used in the error message.
        allow_nan (bool, optional): Whether NaN values are allowed (infinite
            values are still rejected). Defaults to False.

    Raises:
        ValueError: If forbidden values are found.
    """
    if value is None:
        return
    bad = value.isinf().any() if allow_nan else (~torch.isfinite(value)).any()
    if bool(bad):
        raise ValueError(f"{input_name} contains NaN or infinite values")


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
    if c is not None:
        # Single host transfer so device tensors cost no per-row syncs.
        limits = c.reshape(-1).tolist() if torch.is_tensor(c) else list(c)
        if any(
            trajectory[-1][0] > limit + CENSORING_TOLERANCE * max(1.0, abs(limit))
            for trajectory, limit in zip(trajectories, limits, strict=True)
        ):
            raise ValueError(
                "Transitions times may not be greater than censoring times"
            )
