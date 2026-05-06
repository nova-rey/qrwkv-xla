from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path

import pytest

from qrwkv_xla.distill import (
    DistillCheckpointConfig,
    DistillStageConfig,
    DistillStudentConfig,
    load_distill_stage_config,
    run_distill_stage,
)
from qrwkv_xla.targets import validate_target_bundle
from qrwkv_xla.teacher_export import (
    ExportRequest,
    HFTeacherExporter,
    load_teacher_export_config,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    os.environ.get("QRWKV_RUN_HF_INTEGRATION") != "1",
    reason="set QRWKV_RUN_HF_INTEGRATION=1 to run optional HF integration",
)
def test_hf_tiny_integration_opt_in(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    config = load_teacher_export_config(
        ROOT / "configs" / "teacher_export_hf_tiny.yaml"
    )
    config = replace(
        config,
        runtime=replace(config.runtime, output_dir=tmp_path / "hf_tiny"),
    )

    result = HFTeacherExporter().export(
        ExportRequest(config=config, output_dir=config.runtime.output_dir)
    )

    validate_target_bundle(result.output_dir)


@pytest.mark.skipif(
    os.environ.get("QRWKV_RUN_HF_INTEGRATION") != "1",
    reason="set QRWKV_RUN_HF_INTEGRATION=1 to run optional HF integration",
)
def test_p29_tiny_hf_export_to_radlads_reference_resume_opt_in(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    export_config = load_teacher_export_config(
        ROOT / "configs" / "teacher_export_tiny_hf_smoke.yaml"
    )
    export_config = replace(
        export_config,
        runtime=replace(export_config.runtime, output_dir=tmp_path / "tiny_hf"),
    )

    export_result = HFTeacherExporter().export(
        ExportRequest(config=export_config, output_dir=export_config.runtime.output_dir)
    )
    validate_target_bundle(export_result.output_dir)

    distill_config = DistillStageConfig(
        targets_dir=export_result.output_dir,
        student=DistillStudentConfig(
            architecture="rwkv7_radlads_reference",
            vocab_size=50257,
            num_heads=1,
            emit_logits=False,
        ),
        training=replace(DistillStageConfig().training, max_steps=1),
        optimizer=replace(DistillStageConfig().optimizer, learning_rate=0.001),
    )
    first_dir = tmp_path / "checkpoints" / "p29_first"
    second_dir = tmp_path / "checkpoints" / "p29_second"

    run_distill_stage(
        replace(
            distill_config,
            checkpoint=DistillCheckpointConfig(
                checkpoint_out=first_dir,
                overwrite=True,
            ),
        )
    )
    resumed = run_distill_stage(
        replace(
            distill_config,
            checkpoint=DistillCheckpointConfig(
                resume_from=first_dir,
                checkpoint_out=second_dir,
                overwrite=True,
            ),
        )
    )

    assert resumed.start_step == 1
    assert resumed.end_step == 2
    assert math.isfinite(resumed.final_loss)


@pytest.mark.skipif(
    os.environ.get("QRWKV_RUN_HF_INTEGRATION") != "1",
    reason="set QRWKV_RUN_HF_INTEGRATION=1 to run optional HF integration",
)
def test_p30_tiny_hf_logits_export_to_radlads_reference_resume_opt_in(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    export_config = load_teacher_export_config(
        ROOT / "configs" / "teacher_export_tiny_hf_smoke.yaml"
    )
    export_config = replace(
        export_config,
        runtime=replace(export_config.runtime, output_dir=tmp_path / "tiny_hf_logits"),
    )

    export_result = HFTeacherExporter().export(
        ExportRequest(config=export_config, output_dir=export_config.runtime.output_dir)
    )
    validate_target_bundle(export_result.output_dir)

    distill_config = load_distill_stage_config(
        ROOT / "configs" / "distill_stage0_radlads_reference_tiny_hf_logits.yaml"
    )
    distill_config = replace(
        distill_config,
        targets_dir=export_result.output_dir,
    )
    first_dir = tmp_path / "checkpoints" / "p30_logits_first"
    second_dir = tmp_path / "checkpoints" / "p30_logits_second"

    first = run_distill_stage(
        replace(
            distill_config,
            checkpoint=DistillCheckpointConfig(
                checkpoint_out=first_dir,
                overwrite=True,
            ),
        )
    )
    resumed = run_distill_stage(
        replace(
            distill_config,
            checkpoint=DistillCheckpointConfig(
                resume_from=first_dir,
                checkpoint_out=second_dir,
                overwrite=True,
            ),
        )
    )

    assert first.final_logits_kl is not None
    assert math.isfinite(first.final_logits_kl)
    assert resumed.start_step == 1
    assert resumed.end_step == 2
    assert resumed.final_logits_kl is not None
    assert math.isfinite(resumed.final_logits_kl)
