from __future__ import annotations

import itertools
from array import array
from collections import defaultdict
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import torch
from sklearn.utils._param_validation import validate_params  # type: ignore

from ..types._defs import BucketData, Trajectory

if TYPE_CHECKING:
    from ..model._hazard import HazardMixin


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

    Raises:
        TypeError: If the default dtype is not `float32` or `float64`.

    Returns:
        dict[tuple[Any, Any], BucketData]: Transition keys with values `BucketData`.
    """
    dtype = torch.get_default_dtype()
    if dtype == torch.float32:
        typecode = "f"
    elif dtype == torch.float64:
        typecode = "d"
    else:
        raise TypeError(f"Unsupported default dtype: {dtype}")

    # Process each individual trajectory
    buckets: defaultdict[
        tuple[Any, Any], tuple[array[int], array[float], array[float]]
    ] = defaultdict(lambda: (array("q"), array(typecode), array(typecode)))

    for i, trajectory in enumerate(trajectories):
        for (t0, s0), (t1, s1) in itertools.pairwise(trajectory):
            idxs, t0s, t1s = buckets[(s0, s1)]
            idxs.append(i)
            t0s.append(t0)
            t1s.append(t1)

    result = {
        key: BucketData(
            torch.frombuffer(idxs, dtype=torch.int64),
            torch.frombuffer(t0s, dtype=dtype).reshape(-1, 1),
            torch.frombuffer(t1s, dtype=dtype).reshape(-1, 1),
        )
        for key, (idxs, t0s, t1s) in buckets.items()
    }

    return dict(sorted(result.items()))


@lru_cache
def _build_alt_map(
    surv_keys: tuple[tuple[Any, Any], ...],
) -> defaultdict[Any, tuple[tuple[Any, Any], ...]]:
    """Builds alternative state mapping as tuples in a defaultdict.

    Args:
        surv_keys (tuple[tuple[Any, Any], ...]): The survival keys.

    Returns:
        defaultdict[Any, tuple[tuple[Any, Any], ...]]: The alternative state map.
    """
    return defaultdict(
        lambda: (),
        {
            s0: tuple((k, v) for k, v in surv_keys if k == s0)
            for s0 in {s0 for s0, _ in surv_keys}
        },
    )


def build_quad_buckets(
    model: HazardMixin,
    trajectories: list[Trajectory],
    c: torch.Tensor,
) -> dict[tuple[Any, Any], tuple[torch.Tensor, ...]]:
    """Build vectorizable bucket representation.

    Args:
        model (HazardMixin): The model instance.
        trajectories (list[Trajectory]): The trajectories.
        c (torch.Tensor): Censoring times.

    Raises:
        TypeError: If the default dtype is not `float32` or `float64`.

    Returns:
        dict[tuple[Any, Any], tuple[torch.Tensor, ...]]: The vectorizable buckets
            representation.
    """
    alt_map = _build_alt_map(tuple(model.design.link_fns.keys()))
    dtype = torch.get_default_dtype()
    if dtype == torch.float32:
        typecode = "f"
    elif dtype == torch.float64:
        typecode = "d"
    else:
        raise TypeError(f"Unsupported default dtype: {dtype}")

    # Initialize buckets
    buckets: defaultdict[
        tuple[Any, Any], tuple[array[int], array[float], array[float], array[bool]]
    ] = defaultdict(lambda: (array("q"), array(typecode), array(typecode), array("b")))

    # Process each individual trajectory
    for i, trajectory in enumerate(trajectories):
        for (t0, s0), (t1, s1) in itertools.pairwise(trajectory):
            for key in alt_map[s0]:
                idxs, t0s, t1s, obs = buckets[key]
                idxs.append(i)
                t0s.append(t0)
                t1s.append(t1)
                obs.append(key[1] == s1)

        (last_t, last_s), c_i = trajectory[-1], c[i].item()

        if last_t >= c_i:
            continue

        for key in alt_map[last_s]:
            idxs, t0s, t1s, obs = buckets[key]
            idxs.append(i)
            t0s.append(last_t)
            t1s.append(c_i)
            obs.append(False)

    out: dict[tuple[Any, Any], tuple[torch.Tensor, ...]] = {}
    for key, (idxs, t0s, t1s, obs) in buckets.items():
        idxs_ = torch.frombuffer(idxs, dtype=torch.int64)
        t0_ = torch.frombuffer(t0s, dtype=dtype).reshape(-1, 1)
        t1_ = torch.frombuffer(t1s, dtype=dtype).reshape(-1, 1)
        obs_ = torch.frombuffer(obs, dtype=torch.bool)
        half = 0.5 * (t1_ - t0_)
        quad = torch.cat(
            [t1_, (t0_ + t1_).addmm(half, model._std_nodes, beta=0.5)],  # type: ignore
            dim=-1,
        )

        out[key] = (idxs_, t0_, obs_, half, quad)

    return out


def build_remaining_buckets(
    model: HazardMixin,
    trajectories: list[Trajectory],
    c: torch.Tensor,
) -> dict[tuple[Any, Any], tuple[torch.Tensor, ...]]:
    """Build possible bucket representation.

    Args:
        model (HazardMixin): The model instance.
        trajectories (list[Trajectory]): The trajectories.
        c (torch.Tensor): Censoring times.

    Raises:
        TypeError: If the default dtype is not `float32` or `float64`.

    Returns:
        dict[tuple[Any, Any], tuple[torch.Tensor, ...]]: The possible buckets
            representation.
    """
    alt_map = _build_alt_map(tuple(model.design.link_fns.keys()))
    dtype = torch.get_default_dtype()
    if dtype == torch.float32:
        typecode = "f"
    elif dtype == torch.float64:
        typecode = "d"
    else:
        raise TypeError(f"Unsupported default dtype: {dtype}")

    # Initialize buckets
    buckets: defaultdict[tuple[Any, Any], tuple[array[int], array[float]]] = (
        defaultdict(lambda: (array("q"), array(typecode)))
    )

    # Process each individual trajectory
    for i, trajectory in enumerate(trajectories):
        last_t, last_s = trajectory[-1]

        if last_t >= c[i].item():
            continue

        for key in alt_map[last_s]:
            idxs, t0s = buckets[key]
            idxs.append(i)
            t0s.append(last_t)

    return {
        key: (
            idxs_tensor := torch.frombuffer(idxs, dtype=torch.int64),
            torch.frombuffer(t0s, dtype=dtype).reshape(-1, 1),
            c[idxs_tensor],
        )
        for key, (idxs, t0s) in buckets.items()
    }
