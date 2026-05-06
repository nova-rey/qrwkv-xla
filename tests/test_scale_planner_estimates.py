from __future__ import annotations

from dataclasses import replace

from qrwkv_xla.scale_planner import (
    HARDWARE_PROFILES,
    MODEL_PROFILES,
    TRAINING_MODES,
    ScalePlanRequest,
    classify_fit,
    estimate_qwen_reference_parameters,
    estimate_training_memory,
    make_plan,
)


def test_parameter_estimator_matches_tiny_qwen_surface_formula() -> None:
    profile = MODEL_PROFILES["tiny_debug"]
    estimate = estimate_qwen_reference_parameters(profile)

    assert estimate.embedding_params == 512 * 128
    assert estimate.attention_or_time_mix_params > 0
    assert estimate.mlp_params == 2 * 3 * 128 * 512
    assert estimate.total_params == sum(
        value for key, value in estimate.components.items() if key != "per_layer_params"
    )
    assert any("planning" in item.lower() for item in estimate.assumptions)


def test_memory_estimator_includes_logits_dominance_warning() -> None:
    model = replace(MODEL_PROFILES["tiny_debug"], vocab_size=50000)
    memory = estimate_training_memory(
        model,
        HARDWARE_PROFILES["local_cpu_16gb"],
        TRAINING_MODES["smoke_hidden_logits_sgd"],
        sequence_length=512,
        batch_size=8,
    )

    assert memory.components["teacher_logits_targets_per_batch"] > 0
    warnings = "\n".join(memory.warnings)
    assert "Full-vocab logits target memory" in warnings
    assert "dominate estimated memory" in warnings


def test_memory_estimator_tracks_microbatch_and_reserve() -> None:
    memory = estimate_training_memory(
        MODEL_PROFILES["qwen_0_5b_candidate"],
        HARDWARE_PROFILES["colab_tpu_v3_8"],
        TRAINING_MODES["tpu_hidden_bf16_adamw"],
        sequence_length=1024,
        batch_size=8,
        microbatch_size=2,
        grad_accum_steps=4,
    )

    assert memory.microbatch_size == 2
    assert memory.grad_accum_steps == 4
    assert memory.components["xla_overhead_reserve"] > 0
    assert memory.components["wkv_recurrent_state"] > 0
    assert memory.components["shift_state"] > 0


def test_fit_classifier_uses_yes_maybe_no_thresholds() -> None:
    hardware = HARDWARE_PROFILES["local_cpu_16gb"]
    mode = TRAINING_MODES["smoke_hidden_sgd"]
    memory = estimate_training_memory(
        MODEL_PROFILES["tiny_debug"],
        hardware,
        mode,
        sequence_length=32,
        batch_size=1,
    )

    fit = classify_fit(memory, hardware)

    assert fit.fit == "yes"
    assert fit.available_memory_gb > fit.estimated_total_gb
    assert fit.limiting_factor in memory.components


def test_auto_plan_reduces_batch_or_sequence_without_architecture_change() -> None:
    request = ScalePlanRequest(
        model_profile=MODEL_PROFILES["qwen_7b_stretch"],
        hardware_profile=HARDWARE_PROFILES["local_cpu_16gb"],
        training_mode=TRAINING_MODES["scale_hidden_only_bf16_adamw"],
        sequence_length=1024,
        batch_size=8,
        auto=True,
    )
    plan = make_plan(request)

    assert plan.auto_attempts
    assert plan.request.model_profile == request.model_profile
    assert plan.request.batch_size in {8, 4, 2, 1}
    assert plan.request.sequence_length in {1024, 512, 256, 128}
    assert plan.recommended["grad_accum_steps"] >= 1
    assert "planning_only" in plan.distill_config_skeleton
