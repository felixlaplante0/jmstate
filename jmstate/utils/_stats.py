from statistics import NormalDist

import torch


def confidence_interval(
    estimate: torch.Tensor, stderr: torch.Tensor, level: float = 0.95
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes Wald confidence intervals from standard errors.

    The bounds are ``estimate +/- z * stderr``, where ``z`` is the normal
    quantile matching the requested confidence ``level``.

    Args:
        estimate (torch.Tensor): Point estimates.
        stderr (torch.Tensor): Standard errors, broadcastable to ``estimate``.
        level (float, optional): Confidence level in ``(0, 1)``. Defaults to 0.95.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Lower and upper confidence bounds.
    """
    quantile = NormalDist().inv_cdf(0.5 + level / 2.0)
    return estimate - quantile * stderr, estimate + quantile * stderr
