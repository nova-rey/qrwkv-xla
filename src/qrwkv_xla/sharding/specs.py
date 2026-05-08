from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec


@dataclass(frozen=True)
class ShardingPolicy:
    name: str
    mesh_axis: str
    supported: bool
    param_partition: tuple[str, ...]
    batch_partition: tuple[str | None, ...]
    output_partition: tuple[()]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["param_partition"] = list(self.param_partition)
        payload["batch_partition"] = list(self.batch_partition)
        payload["output_partition"] = list(self.output_partition)
        payload["notes"] = list(self.notes)
        return payload


def get_sharding_policy(name: str, *, mesh_axis: str = "data") -> ShardingPolicy:
    if name == "data_parallel_single_axis":
        return ShardingPolicy(
            name=name,
            mesh_axis=mesh_axis,
            supported=True,
            param_partition=(),
            batch_partition=(mesh_axis, None),
            output_partition=(),
            notes=(
                "params replicated",
                "batch axis sharded on the named data axis",
            ),
        )
    if name in {"model_parallel_placeholder", "fsdp_placeholder"}:
        return ShardingPolicy(
            name=name,
            mesh_axis=mesh_axis,
            supported=False,
            param_partition=(),
            batch_partition=(),
            output_partition=(),
            notes=("placeholder policy is intentionally unsupported in P46",),
        )
    raise ValueError(f"unsupported sharding policy: {name}")


def replicated_param_shardings(params: Any, *, mesh: Mesh) -> Any:
    replicated = NamedSharding(mesh, PartitionSpec())
    return jax.tree_util.tree_map(lambda _: replicated, params)


def data_parallel_batch_shardings(
    batch: dict[str, Any],
    *,
    mesh: Mesh,
    mesh_axis: str,
) -> dict[str, NamedSharding]:
    sharding = NamedSharding(mesh, PartitionSpec(mesh_axis, None))
    return {key: sharding for key in batch}


def data_parallel_device_put(
    batch: dict[str, Any],
    *,
    mesh: Mesh,
    mesh_axis: str,
) -> dict[str, jax.Array]:
    sharding = NamedSharding(mesh, PartitionSpec(mesh_axis, None))
    return {key: jax.device_put(value, sharding) for key, value in batch.items()}
