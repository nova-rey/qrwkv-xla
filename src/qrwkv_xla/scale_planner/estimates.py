from __future__ import annotations

from dataclasses import dataclass

from qrwkv_xla.scale_planner.profiles import (
    HardwareProfile,
    ModelProfile,
    TrainingMode,
    dtype_bytes,
)

GIB = 1024**3
YES_THRESHOLD = 0.60
MAYBE_THRESHOLD = 0.85


@dataclass(frozen=True)
class ParameterEstimate:
    total_params: int
    embedding_params: int
    per_layer_params: int
    mlp_params: int
    attention_or_time_mix_params: int
    lm_head_params: int
    components: dict[str, int]
    assumptions: list[str]


@dataclass(frozen=True)
class MemoryEstimate:
    total_bytes: int
    components: dict[str, int]
    warnings: list[str]
    assumptions: list[str]
    microbatch_size: int
    grad_accum_steps: int


@dataclass(frozen=True)
class FitEstimate:
    fit: str
    limiting_factor: str
    available_memory_gb: float
    estimated_total_gb: float
    safety_margin_gb: float
    warnings: list[str]
    utilization: float


def estimate_qwen_reference_parameters(profile: ModelProfile) -> ParameterEstimate:
    h = profile.hidden_size
    layers = profile.num_layers
    kv = profile.kv_hidden_size
    mlp = profile.effective_mlp_hidden_size
    vocab = profile.vocab_size

    embedding_params = vocab * h
    per_layer_norm_params = layers * 2 * h
    final_norm_params = h
    attention_projection_params = layers * (6 * h * h + 2 * h * kv)
    time_mix_params = layers * 2 * h
    attention_or_time_mix_params = attention_projection_params + time_mix_params
    mlp_params = layers * 3 * h * mlp
    lm_head_params = 0
    if profile.emit_logits and not profile.tie_embeddings:
        lm_head_params = vocab * h + vocab
    elif profile.emit_logits:
        lm_head_params = vocab

    per_layer_params = per_layer_norm_params + attention_or_time_mix_params + mlp_params
    components = {
        "embedding_params": embedding_params,
        "layer_norm_params": per_layer_norm_params + final_norm_params,
        "attention_or_time_mix_params": attention_or_time_mix_params,
        "mlp_params": mlp_params,
        "lm_head_params": lm_head_params,
        "per_layer_params": per_layer_params,
    }
    assumptions = [
        "Counts the current qrwkv_xla rwkv7_qwen_reference parameter surface.",
        "Parameter estimate is approximate and intended for conservative planning.",
        "Estimate does not claim RADLADS/HF checkpoint parity or sharding effects.",
    ]
    return ParameterEstimate(
        total_params=sum(
            [
                embedding_params,
                per_layer_norm_params,
                final_norm_params,
                attention_or_time_mix_params,
                mlp_params,
                lm_head_params,
            ]
        ),
        embedding_params=embedding_params,
        per_layer_params=per_layer_params,
        mlp_params=mlp_params,
        attention_or_time_mix_params=attention_or_time_mix_params,
        lm_head_params=lm_head_params,
        components=components,
        assumptions=assumptions,
    )


def estimate_training_memory(
    model: ModelProfile,
    hardware: HardwareProfile,
    mode: TrainingMode,
    *,
    sequence_length: int,
    batch_size: int,
    microbatch_size: int | None = None,
    grad_accum_steps: int = 1,
    dtype: str | None = None,
) -> MemoryEstimate:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be > 0")
    resolved_microbatch = batch_size if microbatch_size is None else microbatch_size
    if resolved_microbatch <= 0 or resolved_microbatch > batch_size:
        raise ValueError("microbatch_size must be in [1, batch_size]")

    params = estimate_qwen_reference_parameters(model)
    resolved_dtype = dtype or mode.dtype
    model_b = dtype_bytes(resolved_dtype)
    act_b = dtype_bytes(resolved_dtype)
    target_b = dtype_bytes(resolved_dtype)

    weights = params.total_params * model_b
    gradients = params.total_params * model_b
    optimizer_first = (
        params.total_params * 4 if mode.optimizer in {"adam", "adamw"} else 0
    )
    optimizer_second = (
        params.total_params * 4 if mode.optimizer in {"adam", "adamw"} else 0
    )
    optimizer_master = params.total_params * 4 if mode.uses_fp32_master_weights else 0
    optimizer_state = optimizer_first + optimizer_second + optimizer_master

    tokens = batch_size * sequence_length
    micro_tokens = resolved_microbatch * sequence_length
    selected_layers = 1 if mode.target_layers == "final" else model.num_layers
    if mode.target_layers == "selected":
        selected_layers = max(1, model.num_layers // 4)

    activations_estimate = int(
        micro_tokens
        * model.hidden_size
        * model.num_layers
        * act_b
        * mode.activation_multiplier
    )
    wkv_recurrent_state = (
        model.num_layers
        * resolved_microbatch
        * model.num_heads
        * model.resolved_head_size
        * model.resolved_head_size
        * act_b
    )
    shift_state = model.num_layers * resolved_microbatch * model.hidden_size * act_b
    teacher_hidden_targets = (
        tokens * selected_layers * model.hidden_size * target_b
        if mode.include_hidden_targets
        else 0
    )
    teacher_logits_targets = (
        tokens * model.vocab_size * target_b if mode.include_logits_targets else 0
    )
    input_ids_and_masks = tokens * (dtype_bytes("int32") + dtype_bytes("int32"))
    checkpoint_size_estimate = weights + optimizer_master

    components = {
        "weights": weights,
        "gradients": gradients,
        "optimizer_state": optimizer_state,
        "optimizer_first_moment": optimizer_first,
        "optimizer_second_moment": optimizer_second,
        "optimizer_fp32_master_weights": optimizer_master,
        "activations_estimate": activations_estimate,
        "wkv_recurrent_state": wkv_recurrent_state,
        "shift_state": shift_state,
        "teacher_hidden_targets_per_batch": teacher_hidden_targets,
        "teacher_logits_targets_per_batch": teacher_logits_targets,
        "input_ids_and_masks": input_ids_and_masks,
        "checkpoint_size_estimate": checkpoint_size_estimate,
    }

    reserve_fraction = 0.30 if hardware.device_kind == "tpu" else 0.20
    overhead_reserve = int(sum(components.values()) * reserve_fraction)
    components["xla_overhead_reserve"] = overhead_reserve

    warnings: list[str] = []
    if teacher_logits_targets > 0:
        warnings.append(
            "Full-vocab logits target memory is included explicitly and can dominate."
        )
    largest_name, largest_value = max(components.items(), key=lambda item: item[1])
    if largest_name == "teacher_logits_targets_per_batch" and largest_value > 0:
        warnings.append(
            "Full logits targets dominate estimated memory. "
            "Consider hidden-only, sampled logits, chunked logits, "
            "or disabling logits KL for this resource class."
        )
    if hardware.device_kind == "tpu" and hardware.device_count > 1:
        warnings.append(
            "TPU memory is treated as an aggregate planning budget; "
            "no pjit/model sharding is implied."
        )
    if mode.losses.ce_loss:
        warnings.append(
            "CE-loss planning is schematic only; "
            "no dedicated large-scale CE runtime is proven here."
        )

    assumptions = [
        "Activation memory uses a conservative "
        "batch*sequence*hidden*layers*dtype*multiplier formula.",
        "Microbatch size, not full batch size, drives activation "
        "and recurrent-state peak estimates.",
        "SGD uses weights plus gradients; Adam/AdamW add first and "
        "second moments and optional fp32 master weights.",
        "CPU/GPU reserve defaults to 20%; TPU/XLA reserve defaults to 30%.",
        "Checkpoint size is a reference artifact estimate, not peak live memory.",
    ]
    return MemoryEstimate(
        total_bytes=sum(components.values()),
        components=components,
        warnings=warnings,
        assumptions=assumptions,
        microbatch_size=resolved_microbatch,
        grad_accum_steps=grad_accum_steps,
    )


def classify_fit(
    memory: MemoryEstimate,
    hardware: HardwareProfile,
    *,
    yes_threshold: float = YES_THRESHOLD,
    maybe_threshold: float = MAYBE_THRESHOLD,
) -> FitEstimate:
    if not 0.0 < yes_threshold < maybe_threshold:
        raise ValueError("thresholds must satisfy 0 < yes_threshold < maybe_threshold")
    available_gb = hardware.memory_gb * hardware.usable_memory_fraction
    estimated_gb = memory.total_bytes / GIB
    if available_gb <= 0:
        return FitEstimate(
            fit="unknown",
            limiting_factor="available_memory_gb",
            available_memory_gb=available_gb,
            estimated_total_gb=estimated_gb,
            safety_margin_gb=available_gb - estimated_gb,
            warnings=["Available memory is unknown or non-positive."],
            utilization=0.0,
        )

    utilization = estimated_gb / available_gb
    if utilization <= yes_threshold:
        fit = "yes"
    elif utilization <= maybe_threshold:
        fit = "maybe"
    else:
        fit = "no"
    warnings = list(memory.warnings)
    if fit == "maybe":
        warnings.append("Estimate is above the conservative yes threshold.")
    elif fit == "no":
        warnings.append("Estimate exceeds the conservative maybe threshold.")

    return FitEstimate(
        fit=fit,
        limiting_factor=max(memory.components.items(), key=lambda item: item[1])[0],
        available_memory_gb=available_gb,
        estimated_total_gb=estimated_gb,
        safety_margin_gb=available_gb - estimated_gb,
        warnings=warnings,
        utilization=utilization,
    )


def format_bytes(num_bytes: int) -> str:
    gib = num_bytes / GIB
    if abs(gib) >= 0.1:
        return f"{gib:.2f} GiB"
    mib = num_bytes / (1024**2)
    return f"{mib:.2f} MiB"
