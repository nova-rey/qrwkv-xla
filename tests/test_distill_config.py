from __future__ import annotations

from pathlib import Path

import pytest

from qrwkv_xla.distill import load_distill_stage_config

ROOT = Path(__file__).resolve().parents[1]


def test_distill_stage0_stub_loads() -> None:
    config = load_distill_stage_config(ROOT / "configs" / "distill_stage0_stub.yaml")
    assert config.stage == 0
    assert config.student.architecture == "rwkv7_reference"
    assert config.student.hidden_size is None
    assert config.student.num_layers is None


def test_missing_sections_use_defaults(tmp_path: Path) -> None:
    path = tmp_path / "distill.yaml"
    path.write_text("distillation:\n  training:\n    max_steps: 7\n", encoding="utf-8")
    config = load_distill_stage_config(path)
    assert config.student.architecture == "rwkv7_reference"
    assert config.training.max_steps == 7
    assert config.losses.hidden_mse.enabled is True
    assert config.checkpoint.checkpoint_out is None
    assert config.gradients.max_grad_norm is None


def test_missing_top_level_distillation_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "distill.yaml"
    path.write_text("training:\n  max_steps: 7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level distillation"):
        load_distill_stage_config(path)


def test_invalid_architecture_raises(tmp_path: Path) -> None:
    path = tmp_path / "distill.yaml"
    path.write_text(
        "distillation:\n  student:\n    architecture: nope\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="student.architecture"):
        load_distill_stage_config(path)


def test_adam_optimizer_config_loads(tmp_path: Path) -> None:
    path = tmp_path / "distill.yaml"
    path.write_text(
        (
            "distillation:\n"
            "  optimizer:\n"
            "    type: adam\n"
            "    learning_rate: 0.0003\n"
            "    beta1: 0.8\n"
        ),
        encoding="utf-8",
    )
    config = load_distill_stage_config(path)

    assert config.optimizer.type == "adam"
    assert config.optimizer.learning_rate == 0.0003
    assert config.optimizer.beta1 == 0.8


def test_lr_schedule_config_loads(tmp_path: Path) -> None:
    path = tmp_path / "distill.yaml"
    path.write_text(
        (
            "distillation:\n"
            "  optimizer:\n"
            "    learning_rate: 0.001\n"
            "  lr_schedule:\n"
            "    type: warmup_cosine\n"
            "    warmup_steps: 1\n"
            "    total_steps: 4\n"
            "    min_learning_rate: 0.0001\n"
        ),
        encoding="utf-8",
    )
    config = load_distill_stage_config(path)

    assert config.lr_schedule.type == "warmup_cosine"
    assert config.lr_schedule.warmup_steps == 1
    assert config.lr_schedule.total_steps == 4
    assert config.lr_schedule.min_learning_rate == 0.0001


def test_gradient_config_loads(tmp_path: Path) -> None:
    path = tmp_path / "distill.yaml"
    path.write_text(
        (
            "distillation:\n"
            "  gradients:\n"
            "    max_grad_norm: 1.0\n"
            "    clip_epsilon: 0.000001\n"
        ),
        encoding="utf-8",
    )
    config = load_distill_stage_config(path)

    assert config.gradients.max_grad_norm == 1.0
    assert config.gradients.clip_epsilon == 0.000001


def test_invalid_gradient_config_raises(tmp_path: Path) -> None:
    path = tmp_path / "distill.yaml"
    path.write_text(
        ("distillation:\n  gradients:\n    max_grad_norm: 0\n"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max_grad_norm"):
        load_distill_stage_config(path)


def test_invalid_clip_epsilon_raises(tmp_path: Path) -> None:
    path = tmp_path / "distill.yaml"
    path.write_text(
        ("distillation:\n  gradients:\n    clip_epsilon: 0\n"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="clip_epsilon"):
        load_distill_stage_config(path)


def test_invalid_optimizer_raises(tmp_path: Path) -> None:
    path = tmp_path / "distill.yaml"
    path.write_text(
        "distillation:\n  optimizer:\n    type: nope\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="optimizer.type"):
        load_distill_stage_config(path)


def test_adam_weight_decay_raises(tmp_path: Path) -> None:
    path = tmp_path / "distill.yaml"
    path.write_text(
        "distillation:\n  optimizer:\n    type: adam\n    weight_decay: 0.1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Use adamw"):
        load_distill_stage_config(path)


def test_all_losses_disabled_or_zero_raises(tmp_path: Path) -> None:
    path = tmp_path / "distill.yaml"
    path.write_text(
        (
            "distillation:\n"
            "  losses:\n"
            "    hidden_mse:\n"
            "      enabled: true\n"
            "      weight: 0.0\n"
            "    logits_kl:\n"
            "      enabled: false\n"
            "      weight: 0.0\n"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least one enabled loss"):
        load_distill_stage_config(path)


def test_attention_or_mixer_enabled_loads(tmp_path: Path) -> None:
    path = tmp_path / "distill.yaml"
    path.write_text(
        (
            "distillation:\n"
            "  losses:\n"
            "    attention_or_mixer:\n"
            "      enabled: true\n"
            "      weight: 1.0\n"
        ),
        encoding="utf-8",
    )
    config = load_distill_stage_config(path)
    assert config.losses.attention_or_mixer.enabled is True
    assert config.losses.attention_or_mixer.weight == 1.0


def test_checkpoint_config_loads(tmp_path: Path) -> None:
    path = tmp_path / "distill.yaml"
    path.write_text(
        (
            "distillation:\n"
            "  checkpoint:\n"
            "    checkpoint_out: checkpoints/unit/out\n"
            "    resume_from: checkpoints/unit/in\n"
            "    overwrite: true\n"
        ),
        encoding="utf-8",
    )
    config = load_distill_stage_config(path)

    assert config.checkpoint.checkpoint_out == Path("checkpoints/unit/out")
    assert config.checkpoint.resume_from == Path("checkpoints/unit/in")
    assert config.checkpoint.overwrite is True


def test_tracking_config_loads(tmp_path: Path) -> None:
    path = tmp_path / "distill.yaml"
    path.write_text(
        (
            "distillation:\n"
            "  tracking:\n"
            "    enabled: true\n"
            "    run_root: runs/unit\n"
            "    run_name: unit smoke\n"
            "    overwrite: true\n"
            "    tags: [unit, smoke]\n"
            "    notes: [local only]\n"
        ),
        encoding="utf-8",
    )
    config = load_distill_stage_config(path)

    assert config.tracking.enabled is True
    assert config.tracking.run_root == Path("runs/unit")
    assert config.tracking.run_name == "unit smoke"
    assert config.tracking.overwrite is True
    assert config.tracking.tags == ["unit", "smoke"]
    assert config.tracking.notes == ["local only"]
