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


def test_invalid_optimizer_raises(tmp_path: Path) -> None:
    path = tmp_path / "distill.yaml"
    path.write_text(
        "distillation:\n  optimizer:\n    type: adam\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="optimizer.type"):
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


def test_attention_or_mixer_enabled_raises(tmp_path: Path) -> None:
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
    with pytest.raises(ValueError, match="attention_or_mixer"):
        load_distill_stage_config(path)


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
