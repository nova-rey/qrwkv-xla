from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from qrwkv_xla.teacher_export import (
    ExportRequest,
    FakeTeacherExporter,
    TeacherExportConfig,
)
from qrwkv_xla.xla import get_jax_runtime_info, run_xla_distill_smoke


def test_run_xla_distill_smoke_cpu_safe(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path)
    result = run_xla_distill_smoke(targets_dir=bundle_dir, max_steps=2)

    assert result.steps == 2
    assert math.isfinite(result.initial_loss)
    assert math.isfinite(result.final_loss)
    assert result.backend
    assert result.device_count >= 1


def test_run_xla_distill_smoke_require_tpu_behavior(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path)
    runtime = get_jax_runtime_info()
    if runtime.has_tpu:
        result = run_xla_distill_smoke(
            targets_dir=bundle_dir,
            max_steps=2,
            require_tpu=True,
        )
        assert result.has_tpu is True
    else:
        with pytest.raises(RuntimeError, match="no TPU was detected"):
            run_xla_distill_smoke(
                targets_dir=bundle_dir,
                max_steps=2,
                require_tpu=True,
            )


def _fake_bundle(tmp_path: Path) -> Path:
    config = TeacherExportConfig()
    config = replace(
        config,
        targets=replace(
            config.targets,
            sequence_length=8,
            hidden_size=4,
            num_layers=2,
            vocab_size=32,
        ),
        runtime=replace(
            config.runtime,
            output_dir=tmp_path / "bundle",
            batch_size=2,
            num_shards=1,
        ),
    )
    FakeTeacherExporter().export(
        ExportRequest(config=config, output_dir=config.runtime.output_dir)
    )
    return config.runtime.output_dir
