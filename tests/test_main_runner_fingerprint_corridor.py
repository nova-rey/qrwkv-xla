from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from qrwkv_xla.checkpointing import load_checkpoint
from qrwkv_xla.distill import (
    DISTILL_MODE_FINGERPRINT_CORRIDOR,
    DistillFingerprintConfig,
    DistillStageConfig,
    DistillStudentConfig,
    load_distill_stage_config,
    run_distill_stage,
)
from qrwkv_xla.teacher_export import (
    ExportRequest,
    FakeTeacherExporter,
    TeacherExportConfig,
)
from qrwkv_xla.training import (
    RealStudentFingerprintForwardConfig,
    run_real_student_fingerprint_forward_smoke,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_distill_stage.py"
FINGERPRINT_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "behavioral_fingerprint"
    / "v0_1_with_exemplars_tiny"
)


def test_main_runner_fingerprint_corridor_completes(tmp_path: Path) -> None:
    result = run_distill_stage(_fingerprint_config(tmp_path, steps=3))

    assert result.status == "pass"
    assert result.distill_mode == DISTILL_MODE_FINGERPRINT_CORRIDOR
    assert result.training_path_kind == "main_runner_fingerprint_corridor"
    assert result.optimizer_steps_completed == 3
    assert result.batches_consumed == 3
    assert result.real_student_backend_integrated is True
    assert result.main_runner_integrated is True
    assert result.teacher_required is False
    assert result.exemplar_reservoir_enabled is False
    assert result.student_backend == "current_qrwkv"
    assert result.student_uses_input_ids is True
    assert math.isfinite(result.final_loss)
    assert result.final_metrics is not None
    assert math.isfinite(result.final_metrics["train/loss"])
    assert result.final_metrics["train/loss"] == pytest.approx(result.final_loss)
    assert "fingerprint/corridor/loss_total" in result.final_metrics
    assert result.final_metrics["fingerprint/runner/optimizer_steps_completed"] == 3.0
    assert result.final_metrics["fingerprint/runner/batches_consumed"] == 3.0
    assert result.final_metrics["fingerprint/runner/artifact_num_records"] == 4.0


def test_main_runner_fingerprint_corridor_requires_no_teacher(tmp_path: Path) -> None:
    path = tmp_path / "p141.yaml"
    path.write_text(
        "\n".join(
            (
                "distillation:",
                "  mode: fingerprint_corridor",
                "  training:",
                "    max_steps: 1",
                "  optimizer:",
                "    learning_rate: 0.01",
                "  fingerprint:",
                f"    artifact_dir: {FINGERPRINT_FIXTURE}",
                "    batch_size: 2",
                f"    output_dir: {tmp_path / 'runner_out'}",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_distill_stage_config(path)
    result = run_distill_stage(config)

    assert config.mode == DISTILL_MODE_FINGERPRINT_CORRIDOR
    assert result.teacher_required is False
    assert result.target_bundle is None
    assert result.fingerprint_artifact == FINGERPRINT_FIXTURE


def test_main_runner_fingerprint_corridor_writes_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "runner_out"
    result = run_distill_stage(_fingerprint_config(tmp_path, steps=2))

    assert result.checkpoint_out == output_dir / "checkpoints" / "final"
    assert (result.checkpoint_out / "checkpoint.json").is_file()
    assert (result.checkpoint_out / "params.npz").is_file()
    assert result.metrics_path == output_dir / "metrics.json"
    assert result.report_path == output_dir / "fingerprint_corridor_report.json"
    assert result.summary_path == output_dir / "fingerprint_run_summary.md"
    assert result.metrics_path.is_file()
    assert result.report_path.is_file()
    assert result.summary_path.is_file()

    checkpoint = load_checkpoint(result.checkpoint_out)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    summary = result.summary_path.read_text(encoding="utf-8")

    assert checkpoint.manifest.student_architecture == "current_qrwkv"
    assert checkpoint.manifest.step == 2
    assert checkpoint.manifest.target_manifest["distill_mode"] == "fingerprint_corridor"
    assert report["phase"] == "P141"
    assert report["distill_mode"] == "fingerprint_corridor"
    assert report["optimizer_steps_completed"] == 2
    assert report["teacher_required"] is False
    assert "main-runner corridor-only fingerprint training" in summary
    assert "No teacher backend is required." in summary


def test_main_runner_fingerprint_corridor_vocab_mismatch_fails(
    tmp_path: Path,
) -> None:
    config = _fingerprint_config(tmp_path, steps=1)
    config = replace(
        config,
        fingerprint=replace(config.fingerprint, student_vocab_size=17),
    )

    with pytest.raises(
        ValueError,
        match="Fingerprint artifact vocab_size=16 but student vocab_size=17",
    ):
        run_distill_stage(config)


def test_main_runner_fingerprint_corridor_zero_batch_guard(tmp_path: Path) -> None:
    config = _fingerprint_config(tmp_path, steps=1)
    config = replace(
        config,
        fingerprint=replace(
            config.fingerprint,
            max_records=1,
            batch_size=2,
            drop_remainder=True,
        ),
    )

    with pytest.raises(ValueError, match="yielded zero batches"):
        run_distill_stage(config)


def test_main_runner_fingerprint_corridor_cli(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(ROOT / "configs" / "distill_stage0_stub.yaml"),
            "--distill-mode",
            "fingerprint_corridor",
            "--fingerprint-artifact",
            str(FINGERPRINT_FIXTURE),
            "--student-backend",
            "current_qrwkv",
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--learning-rate",
            "0.01",
            "--output-dir",
            str(tmp_path / "cli_out"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "distill_mode: fingerprint_corridor" in completed.stdout
    assert "student_backend: current_qrwkv" in completed.stdout
    assert "optimizer_steps_completed: 1" in completed.stdout
    assert (tmp_path / "cli_out" / "fingerprint_corridor_report.json").is_file()
    assert (
        tmp_path / "cli_out" / "checkpoints" / "final" / "checkpoint.json"
    ).is_file()


def test_existing_distill_mode_and_p140_still_work(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path)
    distill = run_distill_stage(
        DistillStageConfig(
            targets_dir=bundle_dir,
            student=DistillStudentConfig(architecture="tiny_student", vocab_size=32),
            training=replace(DistillStageConfig().training, max_steps=1),
            optimizer=replace(DistillStageConfig().optimizer, learning_rate=0.01),
        )
    )
    forward = run_real_student_fingerprint_forward_smoke(
        RealStudentFingerprintForwardConfig(
            artifact_dir=FINGERPRINT_FIXTURE,
            output_dir=tmp_path / "p140",
            batch_size=2,
        )
    )

    assert distill.steps == 1
    assert math.isfinite(distill.final_loss)
    assert distill.distill_mode == "teacher_targets"
    assert forward.status == "pass"


def _fingerprint_config(tmp_path: Path, *, steps: int) -> DistillStageConfig:
    return DistillStageConfig(
        mode=DISTILL_MODE_FINGERPRINT_CORRIDOR,
        training=replace(DistillStageConfig().training, max_steps=steps),
        optimizer=replace(DistillStageConfig().optimizer, learning_rate=0.01),
        fingerprint=DistillFingerprintConfig(
            artifact_dir=FINGERPRINT_FIXTURE,
            batch_size=2,
            student_backend="current_qrwkv",
            output_dir=tmp_path / "runner_out",
        ),
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
