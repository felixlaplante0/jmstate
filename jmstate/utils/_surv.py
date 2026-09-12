from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sklearn.utils._param_validation import validate_params  # type: ignore

from ..types._defs import BucketData, Trajectory
from ._dtype import dtype_device
from ._surv_ext import (
    _build_buckets,
    _build_quad_buckets,
    _build_remaining_buckets,
)

if TYPE_CHECKING:
    from ..model._hazard import HazardMixin


def _from_numpy(values: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Wraps a NumPy array, degrading gracefully without NumPy interop.

    Some PyTorch builds (notably macOS x86-64 with NumPy 2) can import NumPy but
    cannot convert from it, so fall back to a list conversion. The fast path
    stays in place everywhere else.

    Args:
        values (np.ndarray): One-dimensional source array.
        dtype (torch.dtype): Target tensor dtype.

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


def _column(
    values: np.ndarray, dtype: torch.dtype, device: torch.device | None
) -> torch.Tensor:
    """Wraps a 1D time array into a ``(k, 1)`` float column.

    Shares memory with ``values`` when no dtype conversion or device transfer is
    needed, so the returned tensor must not outlive it (it holds a reference).

    Args:
        values (np.ndarray): One-dimensional transition times of shape ``(k,)``.
        dtype (torch.dtype): Floating-point output dtype.
        device (torch.device | None): Target device, or None to keep the tensor
            where ``values`` is wrapped.

    Returns:
        torch.Tensor: Column vector of shape ``(k, 1)`` and dtype ``dtype``.
    """
    out = _from_numpy(values, dtype).reshape(-1, 1)
    return out.to(device) if device is not None else out


def _index(values: np.ndarray, device: torch.device | None) -> torch.Tensor:
    """Wraps a 1D index array into an ``int64`` tensor.

    Args:
        values (np.ndarray): One-dimensional indices of shape ``(k,)``.
        device (torch.device | None): Target device, or None to keep the tensor
            where ``values`` is wrapped.

    Returns:
        torch.Tensor: Index vector of shape ``(k,)`` and dtype ``torch.int64``.
    """
    out = _from_numpy(values, torch.int64)
    return out.to(device) if device is not None else out


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

    Args:
        trajectories (list[Trajectory]): The list of individual trajectories.

    Returns:
        dict[tuple[Any, Any], BucketData]: Transition keys with values ``BucketData``.
    """
    dtype = torch.get_default_dtype()
    result = {
        key: BucketData(
            _index(idxs, None),
            _column(t0s, dtype, None),
            _column(t1s, dtype, None),
        )
        for key, (idxs, t0s, t1s) in _build_buckets(trajectories).items()
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
    """
    dtype, device = dtype_device(model.params)
    if censoring is None:
        censoring = c.reshape(-1).to(dtype=torch.float64, device="cpu").tolist()
    if len(censoring) != len(trajectories):
        raise ValueError(
            f"Got {len(censoring)} censoring times for {len(trajectories)} trajectories"
        )
    return dtype, device, list(model.design.link_fns.keys()), censoring


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
    raw = _build_quad_buckets(trajectories, link_keys, censoring)

    nodes, _weights = model._quad_nodes_weights(dtype, device)
    out: dict[tuple[Any, Any], tuple[torch.Tensor, ...]] = {}
    for key, (idxs, t0s, t1s, obs) in raw.items():
        idxs_ = _index(idxs, device)
        t0_ = _column(t0s, dtype, device)
        t1_ = _column(t1s, dtype, device)
        obs_ = _from_numpy(obs, torch.bool).to(device=device)
        half = 0.5 * (t1_ - t0_)
        quad = torch.cat([t1_, 0.5 * (t0_ + t1_) + half * nodes], dim=-1)
        out[key] = (idxs_, t0_, obs_, half, quad)

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
    raw = _build_remaining_buckets(trajectories, link_keys, censoring)

    c_full = c.reshape(-1, 1).to(dtype=dtype, device=device)
    return {
        key: (
            idxs_tensor := _index(idxs, device),
            _column(t0s, dtype, device),
            c_full[idxs_tensor],
        )
        for key, (idxs, t0s) in raw.items()
    }
