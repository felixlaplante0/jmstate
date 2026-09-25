from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sklearn.utils._param_validation import validate_params  # type: ignore

from ..types._defs import BucketData, Trajectory
from ._surv_ext import (
    _build_buckets,
    _build_quad_buckets,
    _build_remaining_buckets,
)
from .dtype import dtype_device, model_dtype

if TYPE_CHECKING:
    from ..model._hazard import HazardMixin


@cache
def _quad_tensors(
    n_quad: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gets cached Gauss-Legendre nodes and weights.

    Args:
        n_quad (int): Number of quadrature nodes.
        dtype (torch.dtype): Target dtype.
        device (torch.device): Target device.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Nodes of shape ``(1, n_quad)`` and
            weights of shape ``(n_quad,)``.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)  # type: ignore
    return (
        torch.tensor(nodes, dtype=dtype, device=device).unsqueeze(0),
        torch.tensor(weights, dtype=dtype, device=device),
    )


def _from_numpy(values: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Wraps a NumPy array, degrading gracefully without NumPy interop.

    Some PyTorch builds (notably macOS x86-64 with NumPy 2) can import NumPy but
    cannot convert from it, so fall back to a list conversion. The fast path
    stays in place everywhere else.

    Args:
        values (np.ndarray): One-dimensional source array.
        dtype (torch.dtype): Target tensor dtype.

    Raises:
        RuntimeError: If the conversion fails for another reason than missing
            NumPy interop.

    Returns:
        torch.Tensor: The wrapped tensor, sharing memory with ``values`` when the
            bridge is available and no cast is needed.
    """
    try:
        return torch.from_numpy(values).to(dtype=dtype)
    except RuntimeError as exc:
        if "Numpy is not available" not in str(exc):
            raise
        return torch.tensor(values.tolist(), dtype=dtype)


def _tensor(
    values: np.ndarray, dtype: torch.dtype, device: torch.device | None = None
) -> torch.Tensor:
    """Wraps a 1D array into a tensor, as a ``(k, 1)`` column for floating dtypes.

    Shares memory with ``values`` when no dtype conversion or device transfer is
    needed, so the returned tensor must not outlive it (it holds a reference).

    Args:
        values (np.ndarray): One-dimensional source array of shape ``(k,)``.
        dtype (torch.dtype): Output dtype.
        device (torch.device | None, optional): Target device, or None to keep the
            tensor where ``values`` is wrapped. Defaults to None.

    Returns:
        torch.Tensor: Column of shape ``(k, 1)`` for floating dtypes, else ``(k,)``.
    """
    out = _from_numpy(values, dtype).to(device=device)
    return out.reshape(-1, 1) if dtype.is_floating_point else out


@validate_params(
    {"trajectories": [list]},
    prefer_skip_nested_validation=True,
)
def build_buckets(
    trajectories: list[Trajectory],
) -> dict[tuple[Any, Any], BucketData]:
    """Builds buckets from trajectories for user convenience.

    The return structure stores the transition times of individuals grouped together,
    typically used to visualize the trajectories per transition type in multistate
    models. Each entry corresponds to a single transition for a specific individual.

    As no model is involved, times use ``torch``'s default dtype restricted to
    ``torch.float32`` or ``torch.float64``.

    Args:
        trajectories (list[Trajectory]): The list of individual trajectories.

    Returns:
        dict[tuple[Any, Any], BucketData]: Transition keys with values ``BucketData``.
    """
    dtype = model_dtype()
    result = {
        key: BucketData(
            _tensor(idxs, torch.int64), _tensor(t0s, dtype), _tensor(t1s, dtype)
        )
        for key, (idxs, t0s, t1s) in _build_buckets(
            trajectories, dtype == torch.float64
        ).items()
    }

    return dict(sorted(result.items(), key=lambda item: str(item[0])))


def _bucket_inputs(
    model: HazardMixin,
    trajectories: list[Trajectory],
    c: torch.Tensor,
    *,
    censoring: list[float] | None = None,
) -> tuple[torch.dtype, torch.device, list[tuple[Any, Any]], list[float]]:
    """Resolves working dtype/device, link keys and host censoring times.

    The host ``censoring`` list may be supplied to avoid a device sync when the
    same censoring times are reused (e.g. across trajectory sampling steps).

    Args:
        model (HazardMixin): The model instance providing the dtype and device.
        trajectories (list[Trajectory]): The trajectories.
        c (torch.Tensor): Censoring times.
        censoring (list[float] | None, optional): Host censoring times to reuse
            instead of converting ``c`` again. Defaults to None.

    Raises:
        ValueError: If the number of censoring times and trajectories differ.

    Returns:
        tuple[torch.dtype, torch.device, list[tuple[Any, Any]], list[float]]:
            The model dtype, device, transition keys and host censoring times.
    """
    dtype, device = dtype_device(model.params)
    if censoring is None:
        censoring = _host_times(c)
    if len(censoring) != len(trajectories):
        raise ValueError(
            f"Got {len(censoring)} censoring times for {len(trajectories)} trajectories"
        )
    return dtype, device, list(model.design.link_fns.keys()), censoring


def _host_times(c: torch.Tensor) -> list[float]:
    """Converts times to a host list of float64 values.

    Args:
        c (torch.Tensor): The times.

    Returns:
        list[float]: The flattened host times.
    """
    return c.reshape(-1).to(dtype=torch.float64, device="cpu").tolist()


def build_quad_buckets(
    model: HazardMixin,
    trajectories: list[Trajectory],
    c: torch.Tensor,
) -> dict[tuple[Any, Any], tuple[torch.Tensor, ...]]:
    """Build vectorizable bucket representation.

    Time columns follow the model parameters' dtype and device; indices and
    quadrature outputs live on the model device so likelihood code performs no
    transfers.

    Args:
        model (HazardMixin): The model instance.
        trajectories (list[Trajectory]): The trajectories.
        c (torch.Tensor): Censoring times.

    Returns:
        dict[tuple[Any, Any], tuple[torch.Tensor, ...]]: The vectorizable buckets
            representation.
    """
    dtype, device, link_keys, censoring = _bucket_inputs(model, trajectories, c)
    raw = _build_quad_buckets(
        trajectories, link_keys, censoring, dtype == torch.float64
    )

    nodes, _weights = _quad_tensors(model.n_quad, dtype, device)
    out: dict[tuple[Any, Any], tuple[torch.Tensor, ...]] = {}
    for key, (idxs, t0s, t1s, obs) in raw.items():
        t0, t1 = _tensor(t0s, dtype, device), _tensor(t1s, dtype, device)
        half = 0.5 * (t1 - t0)
        quad = torch.cat([t1, 0.5 * (t0 + t1) + half * nodes], dim=-1)
        out[key] = (
            _tensor(idxs, torch.int64, device),
            t0,
            _tensor(obs, torch.bool, device),
            half,
            quad,
        )

    return out


def build_remaining_buckets(
    model: HazardMixin,
    trajectories: list[Trajectory],
    c: torch.Tensor,
    *,
    censoring: list[float] | None = None,
) -> dict[tuple[Any, Any], tuple[torch.Tensor, ...]]:
    """Build possible bucket representation.

    Time columns follow the model parameters' dtype and device so prediction
    code performs no transfers.

    Args:
        model (HazardMixin): The model instance.
        trajectories (list[Trajectory]): The trajectories.
        c (torch.Tensor): Censoring times.
        censoring (list[float] | None, optional): Host censoring times to reuse
            instead of converting ``c`` again. Defaults to None.

    Returns:
        dict[tuple[Any, Any], tuple[torch.Tensor, ...]]: The possible buckets
            representation.
    """
    dtype, device, link_keys, censoring = _bucket_inputs(
        model, trajectories, c, censoring=censoring
    )
    raw = _build_remaining_buckets(
        trajectories, link_keys, censoring, dtype == torch.float64
    )

    c_full = c.reshape(-1, 1).to(dtype=dtype, device=device)
    return {
        key: (
            idxs_tensor := _tensor(idxs, torch.int64, device),
            _tensor(t0s, dtype, device),
            c_full[idxs_tensor],
        )
        for key, (idxs, t0s) in raw.items()
    }
