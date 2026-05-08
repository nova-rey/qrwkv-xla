from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from qrwkv_xla.lm.losses import masked_next_token_cross_entropy
from qrwkv_xla.sharding.mesh import MeshInfo, create_named_mesh
from qrwkv_xla.sharding.specs import (
    ShardingPolicy,
    data_parallel_batch_shardings,
    data_parallel_device_put,
    get_sharding_policy,
    replicated_param_shardings,
)
from qrwkv_xla.students.tiny import TinyStudent, TinyStudentConfig

CompileApi = Literal["auto", "pjit", "jit"]


@dataclass(frozen=True)
class PjitShardingSmokeResult:
    status: str
    overall_status: str
    phase: str
    created_at_utc: str
    compile_api: str
    requested_compile_api: str
    policy: ShardingPolicy
    mesh: MeshInfo
    batch_size: int
    seq_len: int
    finite_loss: bool
    initial_loss: float
    final_loss: float
    update_ran: bool
    multi_device_execution: bool
    step_status: str
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        policy = self.policy.to_dict()
        mesh = self.mesh.to_dict()
        payload["policy"] = self.policy.name
        payload["policy_details"] = policy
        payload["mesh"] = mesh
        payload["backend"] = self.mesh.backend
        payload["platform"] = self.mesh.platform
        payload["device_count"] = self.mesh.device_count
        payload["local_device_count"] = self.mesh.local_device_count
        payload["device_kinds"] = mesh["device_kinds"]
        payload["mesh_shape"] = mesh["mesh_shape"]
        payload["mesh_axis_names"] = mesh["mesh_axis_names"]
        payload["multi_device"] = self.mesh.multi_device
        payload["fallback_reason"] = self.mesh.fallback_reason
        payload["sequence_length"] = self.seq_len
        payload["loss"] = self.final_loss
        payload["loss_is_finite"] = self.finite_loss
        return payload


def run_pjit_sharding_smoke(
    *,
    require_multi_device: bool = False,
    mesh_axis: str = "data",
    batch_size: int = 2,
    seq_len: int = 8,
    compile_api: CompileApi = "auto",
    policy_name: str = "data_parallel_single_axis",
    skip_update: bool = False,
) -> PjitShardingSmokeResult:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if seq_len <= 1:
        raise ValueError("seq_len must be > 1")
    mesh, mesh_info = create_named_mesh(
        mesh_axis=mesh_axis,
        require_multi_device=require_multi_device,
    )
    policy = get_sharding_policy(policy_name, mesh_axis=mesh_axis)
    if not policy.supported:
        raise ValueError(f"sharding policy is unsupported in P46: {policy_name}")

    student = TinyStudent(
        TinyStudentConfig(
            vocab_size=257,
            hidden_size=16,
            num_layers=1,
            emit_logits=True,
        )
    )
    params = student.init_params(jax.random.PRNGKey(46))
    batch = _make_tiny_batch(batch_size=batch_size, seq_len=seq_len)

    with mesh:
        params = jax.device_put(
            params,
            replicated_param_shardings(params, mesh=mesh),
        )
        batch = data_parallel_device_put(batch, mesh=mesh, mesh_axis=mesh_axis)
        selected_compile_api = _select_compile_api(compile_api)
        loss_step = _compile_loss_step(
            student,
            params=params,
            batch=batch,
            mesh=mesh,
            mesh_axis=mesh_axis,
            compile_api=selected_compile_api,
            skip_update=skip_update,
        )
        first_loss, params = loss_step(
            params,
            batch,
            jnp.asarray(0.05, dtype=jnp.float32),
        )
        second_loss, params = loss_step(
            params,
            batch,
            jnp.asarray(0.05, dtype=jnp.float32),
        )
        del params

    first_loss.block_until_ready()
    second_loss.block_until_ready()
    initial_loss = float(first_loss)
    final_loss = float(second_loss)
    finite_loss = bool(jnp.isfinite(second_loss))
    status = "passed" if finite_loss else "failed_non_finite_loss"
    overall_status = "pass" if finite_loss else "fail"
    step_status = (
        "forward_loss_only_pass" if skip_update else "forward_loss_update_pass"
    )
    limitations = (
        "P46 proves tiny sharding compile plumbing only.",
        "P46 does not prove large-model sharding.",
        "P46 does not prove real training throughput.",
        "P46 does not prove production sharded checkpointing.",
        "P46 does not prove 0.5B/1.5B/7B training feasibility.",
        "If only run locally, P46 does not prove multi-device TPU behavior.",
        "P46 prepares the sharding doorway for post-P47 real training work.",
    )
    return PjitShardingSmokeResult(
        status=status,
        overall_status=overall_status,
        phase="P46",
        created_at_utc=datetime.now(UTC).isoformat(),
        compile_api=selected_compile_api,
        requested_compile_api=compile_api,
        policy=policy,
        mesh=mesh_info,
        batch_size=batch_size,
        seq_len=seq_len,
        finite_loss=finite_loss,
        initial_loss=initial_loss,
        final_loss=final_loss,
        update_ran=not skip_update,
        multi_device_execution=mesh_info.multi_device,
        step_status=step_status,
        limitations=limitations,
    )


def _make_tiny_batch(*, batch_size: int, seq_len: int) -> dict[str, jax.Array]:
    input_ids = jnp.arange(batch_size * seq_len, dtype=jnp.int32).reshape(
        batch_size,
        seq_len,
    )
    input_ids = (input_ids % 31) + 1
    labels = jnp.roll(input_ids, shift=-1, axis=1)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    label_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
    label_mask = label_mask.at[:, -1].set(0.0)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "label_mask": label_mask,
    }


def _select_compile_api(requested: CompileApi) -> str:
    if requested == "auto":
        return "jit_with_shardings"
    if requested == "jit":
        return "jit_with_shardings"
    if requested == "pjit":
        return "pjit"
    raise ValueError(f"unsupported compile_api: {requested}")


def _compile_loss_step(
    student: TinyStudent,
    *,
    params: Any,
    batch: dict[str, jax.Array],
    mesh: Any,
    mesh_axis: str,
    compile_api: str,
    skip_update: bool,
) -> Any:
    param_shardings = replicated_param_shardings(params, mesh=mesh)
    batch_shardings = data_parallel_batch_shardings(
        batch,
        mesh=mesh,
        mesh_axis=mesh_axis,
    )
    scalar_sharding = NamedSharding(mesh, PartitionSpec())
    out_shardings = (scalar_sharding, param_shardings)

    def loss_step(
        step_params: Any,
        step_batch: dict[str, jax.Array],
        learning_rate: jax.Array,
    ) -> tuple[jax.Array, Any]:
        def loss_fn(loss_params: Any) -> jax.Array:
            output = student.apply(
                loss_params,
                step_batch["input_ids"],
                attention_mask=step_batch["attention_mask"],
            )
            if output.logits is None:
                raise ValueError("tiny P46 student must emit logits")
            return masked_next_token_cross_entropy(
                logits=output.logits,
                labels=step_batch["labels"],
                label_mask=step_batch["label_mask"],
            )

        if skip_update:
            loss = loss_fn(step_params)
            return loss, step_params
        loss, grads = jax.value_and_grad(loss_fn)(step_params)
        next_params = jax.tree_util.tree_map(
            lambda param, grad: param - learning_rate * grad,
            step_params,
            grads,
        )
        return loss, next_params

    if compile_api == "pjit":
        from jax.experimental.pjit import pjit

        return pjit(
            loss_step,
            in_shardings=(param_shardings, batch_shardings, scalar_sharding),
            out_shardings=out_shardings,
        )
    if compile_api == "jit_with_shardings":
        return jax.jit(
            loss_step,
            in_shardings=(param_shardings, batch_shardings, scalar_sharding),
            out_shardings=out_shardings,
        )
    raise ValueError(f"unsupported selected compile_api: {compile_api}")
