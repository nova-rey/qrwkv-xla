from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from qrwkv_xla.teacher_export import load_teacher_export_config
from qrwkv_xla.teacher_export.config import (
    TeacherExportConfig,
    validate_teacher_export_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_qwen_dryrun_config_loads() -> None:
    config = load_teacher_export_config(
        ROOT / "configs" / "teacher_export_qwen_dryrun.yaml"
    )

    assert config.teacher.family == "qwen"
    assert config.teacher.policy_label == "Qwen3.latest"
    assert config.teacher.resolved_model_id is None
    assert config.runtime.exporter_backend == "hf"
    assert config.runtime.require_resolved_model is True
    assert config.runtime.qwen_policy_path == ROOT / "configs" / "qwen_policy.yaml"


def test_qwen_small_manual_config_loads() -> None:
    config = load_teacher_export_config(
        ROOT / "configs" / "teacher_export_qwen_small_manual.yaml"
    )

    assert config.teacher.policy_label == "qwen-tiny-smoke"
    assert (
        config.runtime.output_dir
        == ROOT / "artifacts/teacher_targets/qwen_manual_smoke"
    )
    assert config.targets.vocab_size == 0


def test_require_resolved_model_behavior_is_validated() -> None:
    config = TeacherExportConfig()
    config = replace(
        config,
        teacher=replace(config.teacher, resolved_model_id=None),
        runtime=replace(
            config.runtime,
            require_resolved_model=True,
            qwen_policy_path=None,
        ),
    )

    with pytest.raises(ValueError, match="require_resolved_model"):
        validate_teacher_export_config(config)


def test_fake_and_hf_tiny_configs_still_load() -> None:
    fake = load_teacher_export_config(ROOT / "configs" / "teacher_export_stub.yaml")
    hf_tiny = load_teacher_export_config(
        ROOT / "configs" / "teacher_export_hf_tiny.yaml"
    )

    assert fake.runtime.exporter_backend == "fake"
    assert hf_tiny.runtime.exporter_backend == "hf"
