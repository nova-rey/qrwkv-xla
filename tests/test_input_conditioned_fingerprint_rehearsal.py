from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp

from qrwkv_xla.artifacts import load_fingerprint_targets, summarize_fingerprint_artifact
from qrwkv_xla.checkpointing import load_checkpoint
from qrwkv_xla.contracts import VocabContract
from qrwkv_xla.distill import (
    DISTILL_MODE_FINGERPRINT_CORRIDOR,
    DistillFingerprintConfig,
    DistillStageConfig,
    DistillStudentConfig,
    load_distill_stage_config,
    run_distill_stage,
)
from qrwkv_xla.students import create_student_backend
from qrwkv_xla.teacher_export import (
    ExportRequest,
    FakeTeacherExporter,
    TeacherExportConfig,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_distill_stage.py"
FINGERPRINT_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "behavioral_fingerprint"
    / "v0_1_with_exemplars_tiny"
)


def test_input_conditioning_detected_for_current_qrwkv_backend() -> None:
    summary = summarize_fingerprint_artifact(FINGERPRINT_FIXTURE)
    backend = create_student_backend(
        vocab_contract=VocabContract(
            tokenizer_id=summary.tokenizer_name or "fingerprint-artifact",
            vocab_size=summary.vocab_size,
        ),
        architecture_id="current_qrwkv",
    )
    params = backend.init_params(jax.random.PRNGKey(0))
    batch = next(
        load_fingerprint_targets(FINGERPRINT_FIXTURE, batch_size=2).iter_batches()
    )

    output, _state = backend.forward_full(
        params,
        jnp.asarray(batch.input_ids, dtype=jnp.int32),
    )
    logits = backend.logits(output)

    assert logits.shape == (2, 8, 16)
    assert float(jnp.linalg.norm(logits[0] - logits[1])) > 1e-7
    assert not bool(jnp.allclose(logits[0], logits[1]))


def test_input_conditioned_rehearsal_completes_with_diagnostics(
    tmp_path: Path,
) -> None:
    result = run_distill_stage(_rehearsal_config(tmp_path, steps=4))

    assert result.status == "pass"
    assert result.distill_mode == DISTILL_MODE_FINGERPRINT_CORRIDOR
    assert result.training_path_kind == "main_runner_fingerprint_corridor"
    assert result.optimizer_steps_completed == 4
    assert result.batches_consumed == 4
    assert result.main_runner_integrated is True
    assert result.real_student_backend_integrated is True
    assert result.teacher_required is False
    assert result.exemplar_reservoir_enabled is False
    assert result.student_uses_input_ids is True
    assert result.final_metrics is not None
    assert (
        result.final_metrics["fingerprint/rehearsal/input_conditioning_detected"] == 1.0
    )
    assert result.final_metrics["fingerprint/rehearsal/params_changed"] == 1.0
    assert result.final_metrics["fingerprint/rehearsal/param_delta_norm"] > 0.0
    assert math.isfinite(result.final_metrics["fingerprint/rehearsal/initial_loss"])
    assert math.isfinite(result.final_metrics["fingerprint/rehearsal/final_loss"])
    assert math.isfinite(result.final_metrics["fingerprint/rehearsal/loss_delta"])
    assert "fingerprint/corridor/loss_total" in result.final_metrics
    assert "fingerprint/runner/batches_consumed" in result.final_metrics


def test_input_conditioned_rehearsal_report_summary_and_checkpoint(
    tmp_path: Path,
) -> None:
    result = run_distill_stage(_rehearsal_config(tmp_path, steps=3))
    assert result.checkpoint_out is not None
    assert result.report_path is not None
    assert result.summary_path is not None

    checkpoint = load_checkpoint(result.checkpoint_out)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    summary = result.summary_path.read_text(encoding="utf-8")

    assert checkpoint.manifest.student_architecture == "current_qrwkv"
    assert checkpoint.manifest.step == 3
    assert checkpoint.manifest.target_manifest["distill_mode"] == "fingerprint_corridor"
    assert report["phase"] == "P142"
    assert report["distill_mode"] == "fingerprint_corridor"
    assert report["input_conditioned_rehearsal"] is True
    assert report["input_conditioning_detected"] is True
    assert report["params_changed"] is True
    assert report["param_delta_norm"] > 0.0
    assert report["loss_non_increasing_required"] is False
    assert isinstance(report["loss_non_increasing"], bool)
    assert "Input-Conditioned Tiny Fingerprint Rehearsal Summary" in summary
    assert "This is an input-conditioned tiny rehearsal." in summary
    assert "It uses the main fingerprint_corridor runner mode." in summary
    assert "No teacher backend is required." in summary


def test_input_conditioned_rehearsal_requires_no_teacher_config(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "p142.yaml"
    config_path.write_text(
        "\n".join(
            (
                "distillation:",
                "  mode: fingerprint_corridor",
                "  training:",
                "    max_steps: 2",
                "  optimizer:",
                "    learning_rate: 0.01",
                "  fingerprint:",
                f"    artifact_dir: {FINGERPRINT_FIXTURE}",
                "    batch_size: 2",
                "    input_conditioned_rehearsal: true",
                f"    output_dir: {tmp_path / 'rehearsal'}",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_distill_stage_config(config_path)
    result = run_distill_stage(config)

    assert config.mode == "fingerprint_corridor"
    assert result.status == "pass"
    assert result.teacher_required is False
    assert result.target_bundle is None
    assert result.fingerprint_artifact == FINGERPRINT_FIXTURE


def test_input_conditioned_rehearsal_cli(tmp_path: Path) -> None:
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
            "2",
            "--batch-size",
            "2",
            "--learning-rate",
            "0.01",
            "--fingerprint-input-conditioned-rehearsal",
            "--output-dir",
            str(tmp_path / "cli"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "distill_mode: fingerprint_corridor" in completed.stdout
    report = json.loads(
        (tmp_path / "cli" / "fingerprint_corridor_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["phase"] == "P142"
    assert report["input_conditioning_detected"] is True
    assert report["params_changed"] is True


def test_p141_and_teacher_target_modes_still_work(tmp_path: Path) -> None:
    p141 = run_distill_stage(_p141_config(tmp_path, steps=1))
    bundle_dir = _fake_bundle(tmp_path)
    teacher_targets = run_distill_stage(
        DistillStageConfig(
            targets_dir=bundle_dir,
            student=DistillStudentConfig(architecture="tiny_student", vocab_size=32),
            training=replace(DistillStageConfig().training, max_steps=1),
            optimizer=replace(DistillStageConfig().optimizer, learning_rate=0.01),
        )
    )

    assert p141.status == "pass"
    assert p141.final_metrics is not None
    assert p141.final_metrics["fingerprint/rehearsal/params_changed"] == 1.0
    assert teacher_targets.distill_mode == "teacher_targets"
    assert math.isfinite(teacher_targets.final_loss)


def _rehearsal_config(tmp_path: Path, *, steps: int) -> DistillStageConfig:
    config = _p141_config(tmp_path, steps=steps)
    return replace(
        config,
        fingerprint=replace(config.fingerprint, input_conditioned_rehearsal=True),
    )


def _p141_config(tmp_path: Path, *, steps: int) -> DistillStageConfig:
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
