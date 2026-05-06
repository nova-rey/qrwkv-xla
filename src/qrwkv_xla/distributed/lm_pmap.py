from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from qrwkv_xla.checkpointing import save_checkpoint
from qrwkv_xla.distributed.devices import DeviceTopology, get_device_topology
from qrwkv_xla.distributed.reductions import metrics_pmean, tree_pmean
from qrwkv_xla.distributed.replication import (
    replicate_to_devices,
    unreplicate_from_devices,
)
from qrwkv_xla.distributed.sharding import shard_batch_for_devices
from qrwkv_xla.lm.config import LMStageConfig, validate_lm_stage_config
from qrwkv_xla.lm.data import build_lm_batches, load_lm_token_sequences
from qrwkv_xla.lm.losses import masked_next_token_cross_entropy
from qrwkv_xla.optimizers import OptimizerConfig, init_optimizer_state, optimizer_update
from qrwkv_xla.prompting import (
    build_prompt_corpus_manifest,
    filter_prompt_corpus,
    read_prompt_corpus,
)
from qrwkv_xla.schedules import learning_rate_at_step
from qrwkv_xla.students import create_student
from qrwkv_xla.training.gradients import clip_gradients_by_global_norm


@dataclass(frozen=True)
class PmapLMSmokeResult:
    device_count: int
    per_device_batch_size: int
    steps: int
    initial_loss: float
    final_loss: float
    checkpoint_out: Path | None = None


@dataclass(frozen=True)
class PmapLMSkip:
    reason: str
    topology: DeviceTopology


@dataclass(frozen=True)
class _PreparedLMSmoke:
    config: LMStageConfig
    topology: DeviceTopology
    active_device_count: int
    per_device_batch_size: int
    batch: dict[str, jax.Array]
    student: Any
    student_config: dict[str, Any]
    prompt_manifest: Any


def prepare_pmap_lm_smoke(
    config: LMStageConfig,
    *,
    device_count_cap: int | None = None,
) -> _PreparedLMSmoke | PmapLMSkip:
    validate_lm_stage_config(config)
    if config.checkpoint.resume_from is not None:
        raise ValueError("pmap LM smoke does not support resume_from yet")
    if not config.distributed.enabled:
        raise ValueError("pmap LM smoke requires distributed.enabled=true")
    if config.distributed.mode != "pmap_data_parallel":
        raise ValueError("pmap LM smoke requires distributed.mode=pmap_data_parallel")

    token_sequences = load_lm_token_sequences(config.data)
    batches = build_lm_batches(
        token_sequences,
        sequence_length=config.data.sequence_length,
        batch_size=config.data.batch_size,
    )
    if not batches:
        raise ValueError("No LM batches were built")
    first_batch = batches[0]
    batch = {
        "input_ids": jnp.asarray(first_batch.input_ids),
        "labels": jnp.asarray(first_batch.labels),
        "attention_mask": jnp.asarray(first_batch.attention_mask),
        "label_mask": jnp.asarray(first_batch.label_mask),
    }
    batch_size = int(batch["input_ids"].shape[0])
    topology = get_device_topology()
    active_device_count = _resolve_active_device_count(
        batch_size=batch_size,
        local_device_count=topology.local_device_count,
        min_device_count=config.distributed.min_device_count,
        device_count_cap=device_count_cap,
    )
    if active_device_count is None:
        cap_text = f", cap={device_count_cap}" if device_count_cap is not None else ""
        return PmapLMSkip(
            reason=(
                "not enough divisible devices for pmap LM smoke: "
                f"batch_size={batch_size}, "
                f"local_device_count={topology.local_device_count}, "
                f"min_device_count={config.distributed.min_device_count}{cap_text}"
            ),
            topology=topology,
        )

    student = create_student(
        config.student.architecture,
        vocab_size=config.student.vocab_size,
        hidden_size=config.student.hidden_size,
        num_layers=config.student.num_layers,
        num_heads=config.student.num_heads,
        emit_logits=config.student.emit_logits,
        tie_embeddings=config.student.tie_embeddings,
    )
    prompt_manifest = _prompt_manifest_for_config(config)
    sharded_batch = shard_batch_for_devices(batch, device_count=active_device_count)
    return _PreparedLMSmoke(
        config=config,
        topology=topology,
        active_device_count=active_device_count,
        per_device_batch_size=int(sharded_batch["input_ids"].shape[1]),
        batch=sharded_batch,
        student=student,
        student_config=asdict(config.student),
        prompt_manifest=prompt_manifest,
    )


def run_pmap_lm_smoke(
    config: LMStageConfig,
    *,
    device_count_cap: int | None = None,
) -> PmapLMSmokeResult:
    prepared = prepare_pmap_lm_smoke(config, device_count_cap=device_count_cap)
    if isinstance(prepared, PmapLMSkip):
        raise RuntimeError(prepared.reason)

    optimizer_config = config.optimizer.to_optimizer_config()
    params = prepared.student.init_params(jax.random.PRNGKey(config.training.seed))
    optimizer_state = init_optimizer_state(params, optimizer_config)
    replicated_params = replicate_to_devices(
        params, device_count=prepared.active_device_count
    )
    replicated_optimizer_state = replicate_to_devices(
        optimizer_state,
        device_count=prepared.active_device_count,
    )
    axis_name = config.distributed.axis_name

    def loss_fn(
        local_params: Any, batch: dict[str, jax.Array]
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        output = prepared.student.apply(
            local_params,
            batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )
        if output.logits is None:
            raise ValueError("Stage 3 CE training requires student logits")
        ce_loss = masked_next_token_cross_entropy(
            logits=output.logits,
            labels=batch["labels"],
            label_mask=batch["label_mask"],
        )
        return ce_loss, {"loss": ce_loss, "ce_loss": ce_loss}

    def train_step(
        local_params: Any,
        local_optimizer_state: Any,
        learning_rate: float,
        batch: dict[str, jax.Array],
    ):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            local_params,
            batch,
        )
        grads = tree_pmean(grads, axis_name=axis_name)
        reduced_metrics = metrics_pmean(metrics, axis_name=axis_name)
        reduced_loss = jax.lax.pmean(loss, axis_name)
        clip_result = clip_gradients_by_global_norm(
            grads,
            max_grad_norm=config.gradients.max_grad_norm,
            epsilon=config.gradients.clip_epsilon,
        )
        update_config = OptimizerConfig(
            type=optimizer_config.type,
            learning_rate=learning_rate,
            beta1=optimizer_config.beta1,
            beta2=optimizer_config.beta2,
            epsilon=optimizer_config.epsilon,
            weight_decay=optimizer_config.weight_decay,
        )
        new_params, new_optimizer_state, optimizer_metrics = optimizer_update(
            local_params,
            clip_result.gradients,
            local_optimizer_state,
            update_config,
        )
        out_metrics = dict(reduced_metrics)
        out_metrics["loss"] = reduced_loss
        out_metrics["grad_global_norm"] = clip_result.global_norm
        out_metrics["grad_clipped_global_norm"] = clip_result.clipped_global_norm
        out_metrics["grad_clip_scale"] = clip_result.clip_scale
        out_metrics["grad_was_clipped"] = clip_result.was_clipped
        out_metrics.update(optimizer_metrics)
        return new_params, new_optimizer_state, out_metrics

    pmapped_train_step = jax.pmap(
        train_step, axis_name=axis_name, in_axes=(0, 0, None, 0)
    )

    initial_loss: float | None = None
    final_loss: float | None = None
    for step_index in range(config.training.max_steps):
        scheduled_lr = learning_rate_at_step(
            step=step_index,
            base_learning_rate=config.optimizer.learning_rate,
            config=config.lr_schedule,
        )
        replicated_params, replicated_optimizer_state, metrics = pmapped_train_step(
            replicated_params,
            replicated_optimizer_state,
            scheduled_lr,
            prepared.batch,
        )
        host_metrics = unreplicate_from_devices(metrics)
        loss_value = float(jnp.asarray(host_metrics["loss"]))
        if initial_loss is None:
            initial_loss = loss_value
        final_loss = loss_value

    assert initial_loss is not None
    assert final_loss is not None

    checkpoint_out = config.checkpoint.checkpoint_out
    if checkpoint_out is not None:
        save_checkpoint(
            checkpoint_out,
            unreplicate_from_devices(replicated_params),
            student_architecture=config.student.architecture,
            student_config=prepared.student_config,
            step=config.training.max_steps,
            learning_rate=config.optimizer.learning_rate,
            loss_config={"next_token_ce": {"enabled": True, "weight": 1.0}},
            target_manifest={
                "type": "prompt_corpus",
                "stage": config.training.stage,
                "manifest": asdict(prepared.prompt_manifest),
                "data": asdict(config.data),
            },
            optimizer_config=asdict(config.optimizer),
            optimizer_state=unreplicate_from_devices(replicated_optimizer_state),
            lr_schedule={
                "type": config.lr_schedule.type,
                "warmup_steps": config.lr_schedule.warmup_steps,
                "total_steps": config.lr_schedule.total_steps,
                "min_learning_rate": config.lr_schedule.min_learning_rate,
                "base_learning_rate": config.optimizer.learning_rate,
            },
            gradients=asdict(config.gradients),
            notes=[
                "simple JSON + NPZ checkpoint",
                "student-only stage 3 CE",
                "pmap data-parallel smoke",
                "saved from unreplicated first-device state",
            ],
            overwrite=config.checkpoint.overwrite,
        )

    return PmapLMSmokeResult(
        device_count=prepared.active_device_count,
        per_device_batch_size=prepared.per_device_batch_size,
        steps=config.training.max_steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        checkpoint_out=checkpoint_out,
    )


def _prompt_manifest_for_config(config: LMStageConfig):
    corpus = read_prompt_corpus(config.data.prompt_corpus)
    filtered = filter_prompt_corpus(
        corpus,
        split=config.data.prompt_split,
        tags=config.data.prompt_tags,
        limit=config.data.prompt_limit,
    )
    return build_prompt_corpus_manifest(
        filtered,
        description="Stage 3 CE prompt corpus selection.",
        notes=["student-only stage 3 CE", "pmap data-parallel smoke"],
    )


def _resolve_active_device_count(
    *,
    batch_size: int,
    local_device_count: int,
    min_device_count: int,
    device_count_cap: int | None,
) -> int | None:
    max_devices = min(batch_size, local_device_count)
    if device_count_cap is not None:
        if device_count_cap < 1:
            raise ValueError("device_count cap must be >= 1")
        max_devices = min(max_devices, device_count_cap)
    for candidate in range(max_devices, 0, -1):
        if candidate < min_device_count:
            continue
        if batch_size % candidate == 0:
            return candidate
    return None
