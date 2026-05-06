from __future__ import annotations

import pytest

from qrwkv_xla.scale_planner import (
    HARDWARE_PROFILES,
    MODEL_PROFILES,
    TRAINING_MODES,
    HardwareProfile,
    LossProfile,
    ModelProfile,
    TrainingMode,
    validate_hardware_profile,
    validate_model_profile,
    validate_training_mode,
)


def test_builtin_profiles_are_present_and_valid() -> None:
    assert set(MODEL_PROFILES) == {
        "tiny_debug",
        "small_cpu",
        "colab_tpu_smoke",
        "qwen_0_5b_candidate",
        "qwen_1_5b_candidate",
        "qwen_7b_stretch",
    }
    assert set(HARDWARE_PROFILES) == {
        "local_cpu_16gb",
        "local_cpu_32gb",
        "local_cpu_64gb",
        "colab_tpu_v2_8",
        "colab_tpu_v3_8",
        "single_l4_24gb",
        "single_a100_40gb",
        "grant_tpu_v5e_8",
        "big_budget_tpu_placeholder",
    }
    assert set(TRAINING_MODES) == {
        "smoke_hidden_sgd",
        "smoke_hidden_logits_sgd",
        "local_hidden_adamw",
        "tpu_hidden_bf16_adamw",
        "tpu_hidden_logits_bf16_adamw",
        "scale_hidden_only_bf16_adamw",
        "scale_sampled_logits_bf16_adamw_placeholder",
    }
    for profile in MODEL_PROFILES.values():
        validate_model_profile(profile)
        assert profile.backend in {
            "rwkv7_qwen_reference",
            "rwkv7_radlads_reference",
            "rwkv7_reference",
        }
    for profile in HARDWARE_PROFILES.values():
        validate_hardware_profile(profile)
        assert profile.resolved_per_device_memory_gb > 0
    for mode in TRAINING_MODES.values():
        validate_training_mode(mode)
        assert mode.dtype in {"fp32", "bf16", "fp16"}


def test_model_validation_rejects_bad_head_shape() -> None:
    with pytest.raises(ValueError, match="divisible"):
        validate_model_profile(
            ModelProfile(
                name="bad",
                backend="rwkv7_qwen_reference",
                vocab_size=128,
                hidden_size=130,
                num_layers=2,
                num_heads=8,
                num_kv_heads=1,
                mlp_hidden_size=1024,
            )
        )


def test_model_validation_rejects_bad_mlp_size() -> None:
    with pytest.raises(ValueError, match="mlp_hidden_size"):
        validate_model_profile(
            ModelProfile(
                name="bad_mlp",
                backend="rwkv7_qwen_reference",
                vocab_size=128,
                hidden_size=128,
                num_layers=2,
                num_heads=8,
                num_kv_heads=1,
                mlp_hidden_size=128,
            )
        )


def test_hardware_and_training_validation_reject_bad_values() -> None:
    with pytest.raises(ValueError, match="memory_gb"):
        validate_hardware_profile(
            HardwareProfile(name="bad", device_kind="cpu", memory_gb=0)
        )
    with pytest.raises(ValueError, match="at least one target family"):
        validate_training_mode(
            TrainingMode(
                name="bad",
                losses=LossProfile(hidden_mse=False, logits_kl=False, ce_loss=False),
                optimizer="adamw",
                dtype="fp32",
            )
        )
