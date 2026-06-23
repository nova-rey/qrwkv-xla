from __future__ import annotations

import json
from pathlib import Path

from qrwkv_xla.checkpointing import load_checkpoint
from qrwkv_xla.fingerprint.adaptive_corridor_pass import (
    AdaptiveCorridorPassConfig,
    run_adaptive_corridor_pass,
)
from qrwkv_xla.fingerprint.adaptive_corridor_scheduler import (
    AdaptiveCorridorSchedulerConfig,
)
from qrwkv_xla.fingerprint.mode_plateau_controller import ModePlateauConfig
from qrwkv_xla.fingerprint.provenance import hash_checkpoint_bundle
from scripts.run_adaptive_corridor_pass import (
    _build_three_mode_artifact,
    _calibration_override,
    run_smoke,
)


def test_real_cpu_adaptive_corridor_smoke(tmp_path: Path) -> None:
    report = run_smoke(tmp_path / "p156_3")
    output = tmp_path / "p156_3" / "resumed"
    assert report["phase"] == "P156.3.1"
    assert report["status"] == "pass"
    assert report["cycle_one_complete"] is True
    assert report["global_completion_reason"] == "all_required_modes_stably_frozen"
    assert report["reactivation_count"] == 1
    assert report["resume_equivalent"] is True
    assert report["fraction_mode_work_saved"] > 0
    assert report["exemplar_training_launched"] is False
    assert report["modes"]["0"]["freeze_step"] == 1
    assert report["modes"]["1"]["freeze_step"] == 4
    assert report["modes"]["2"]["reactivation_steps"] == [3]
    assert report["modes"]["2"]["refreeze_steps"] == [5]
    checkpoint = output / "checkpoints" / "adaptive_corridor_final_checkpoint"
    loaded = load_checkpoint(checkpoint)
    assert loaded.manifest.step == report["optimizer_steps_completed"]
    assert loaded.optimizer_state is not None
    assert (checkpoint / "adaptive_state.json").is_file()
    assert (output / "checkpoints" / "step_000001_event" / "checkpoint.json").is_file()
    assert (output / "checkpoints" / "step_000003_event" / "checkpoint.json").is_file()
    assert (output / "checkpoints" / "step_000005_event" / "checkpoint.json").is_file()
    for name in (
        "adaptive_corridor_report.json",
        "adaptive_corridor_summary.md",
        "adaptive_corridor_transitions.jsonl",
        "adaptive_corridor_calibration_trajectory.jsonl",
        "adaptive_corridor_weight_trajectory.jsonl",
        "adaptive_corridor_checkpoint_lineage.json",
        "adaptive_corridor_resume_receipt.json",
    ):
        assert (output / name).is_file()
    trajectory = [
        json.loads(line)
        for line in (output / "adaptive_corridor_calibration_trajectory.jsonl")
        .read_text()
        .splitlines()
    ]
    assert {row["mode_id"] for row in trajectory} == {"0", "1", "2"}
    assert any(
        row["controller_state_before"] == "FROZEN" and row["mode_id"] == "2"
        for row in trajectory
    )
    transitions = [
        json.loads(line)
        for line in (output / "adaptive_corridor_transitions.jsonl")
        .read_text()
        .splitlines()
    ]
    step_one_freezes = [
        row for row in transitions if row["step"] == 1 and row["event"] == "mode_frozen"
    ]
    assert [row["active_modes_before"] for row in step_one_freezes] == [3, 2]
    assert [row["active_modes_after"] for row in step_one_freezes] == [2, 1]


def test_maximum_step_cap_reports_incomplete(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _build_three_mode_artifact(artifact)
    scheduler = AdaptiveCorridorSchedulerConfig(
        controller=ModePlateauConfig(
            required_modes=("0", "1", "2"),
            minimum_observations=3,
            progress_window_observations=3,
            plateau_patience_observations=2,
            maximum_corridor_steps=2,
        ),
        mode_weights={"0": 1.0, "1": 1.0, "2": 1.0},
    )

    def never_enter(step: int, mode_id: str, metrics: object) -> dict[str, float]:
        del step, mode_id, metrics
        return {
            "corridor_loss": 1.0,
            "inside_corridor_rate": 0.0,
            "mean_distance_outside_corridor": 1.0,
            "worst_stat_violation": 1.0,
        }

    result = run_adaptive_corridor_pass(
        AdaptiveCorridorPassConfig(
            training_fingerprint_artifact=artifact,
            calibration_fingerprint_artifact=artifact,
            output_dir=tmp_path / "capped",
            scheduler=scheduler,
            student_backend="tiny_debug",
            overwrite=True,
        ),
        calibration_override=never_enter,
    )
    report = json.loads(result.report_path.read_text())
    assert result.status == "incomplete"
    assert result.cycle_one_complete is False
    assert report["global_completion_reason"] == "maximum_step_cap"
    assert report["optimizer_steps_completed"] == 2
    assert report["confirmation_only_evaluations_completed"] == 0
    assert report["optimizer_step_cap_respected"] is True


def test_final_freeze_at_cap_confirms_and_resumes_exactly(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _build_three_mode_artifact(artifact)
    scheduler = _boundary_scheduler(maximum_corridor_steps=5)
    common = {
        "training_fingerprint_artifact": artifact,
        "calibration_fingerprint_artifact": artifact,
        "scheduler": scheduler,
        "student_backend": "tiny_debug",
        "overwrite": True,
    }

    def freeze_at_cap(step: int, mode_id: str, metrics: object) -> dict[str, float]:
        result = _calibration_override(step, mode_id, metrics)
        if mode_id == "1" and step >= 4:
            result["corridor_loss"] = 0.2
        return result

    uninterrupted = run_adaptive_corridor_pass(
        AdaptiveCorridorPassConfig(output_dir=tmp_path / "uninterrupted", **common),
        calibration_override=freeze_at_cap,
    )
    freeze_checkpoint = uninterrupted.output_dir / "checkpoints" / "step_000005_event"
    freeze_hash = hash_checkpoint_bundle(freeze_checkpoint)
    final_hash = hash_checkpoint_bundle(uninterrupted.final_checkpoint)
    assert freeze_hash["params_sha256"] == final_hash["params_sha256"]
    freeze_loaded = load_checkpoint(freeze_checkpoint)
    final_loaded = load_checkpoint(uninterrupted.final_checkpoint)
    assert freeze_loaded.optimizer_state is not None
    assert final_loaded.optimizer_state is not None
    assert freeze_loaded.optimizer_state.step == final_loaded.optimizer_state.step

    boundary = run_adaptive_corridor_pass(
        AdaptiveCorridorPassConfig(output_dir=tmp_path / "boundary_resume", **common),
        calibration_override=freeze_at_cap,
        interrupt_after_optimizer_step=5,
    )
    assert boundary.interruption_checkpoint is not None
    resumed_boundary = run_adaptive_corridor_pass(
        AdaptiveCorridorPassConfig(
            output_dir=tmp_path / "boundary_resume",
            resume_checkpoint=boundary.interruption_checkpoint,
            **common,
        ),
        calibration_override=freeze_at_cap,
    )

    mid = run_adaptive_corridor_pass(
        AdaptiveCorridorPassConfig(output_dir=tmp_path / "mid_confirmation", **common),
        calibration_override=freeze_at_cap,
        interrupt_after_confirmation_only_evaluation=1,
    )
    assert mid.interruption_checkpoint is not None
    resumed_mid = run_adaptive_corridor_pass(
        AdaptiveCorridorPassConfig(
            output_dir=tmp_path / "mid_confirmation",
            resume_checkpoint=mid.interruption_checkpoint,
            **common,
        ),
        calibration_override=freeze_at_cap,
    )

    expected_hash = hash_checkpoint_bundle(uninterrupted.final_checkpoint)[
        "checkpoint_bundle_sha256"
    ]
    for result in (uninterrupted, resumed_boundary, resumed_mid):
        report = json.loads(result.report_path.read_text())
        assert result.cycle_one_complete
        assert report["optimizer_steps_completed"] == 5
        assert report["maximum_corridor_steps"] == 5
        assert report["confirmation_only_evaluations_completed"] == 2
        assert report["global_completion_reason"] == (
            "all_required_modes_stably_frozen"
        )
        assert report["optimizer_step_cap_respected"] is True
        assert (
            hash_checkpoint_bundle(result.final_checkpoint)["checkpoint_bundle_sha256"]
            == expected_hash
        )


def test_reactivation_at_optimizer_cap_is_incomplete(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _build_three_mode_artifact(artifact)
    scheduler = _boundary_scheduler(maximum_corridor_steps=2)

    def regress_during_confirmation(
        step: int, mode_id: str, metrics: object
    ) -> dict[str, float]:
        del metrics
        regressing = mode_id == "0" and step >= 3
        loss = 1.0 if step == 0 else 0.8
        return {
            "corridor_loss": loss,
            "inside_corridor_rate": 0.5 if regressing else 0.97,
            "mean_distance_outside_corridor": 0.2 if regressing else 0.03,
            "worst_stat_violation": 0.2 if regressing else 0.03,
        }

    result = run_adaptive_corridor_pass(
        AdaptiveCorridorPassConfig(
            training_fingerprint_artifact=artifact,
            calibration_fingerprint_artifact=artifact,
            output_dir=tmp_path / "reactivation_cap",
            scheduler=scheduler,
            student_backend="tiny_debug",
            overwrite=True,
        ),
        calibration_override=regress_during_confirmation,
    )
    report = json.loads(result.report_path.read_text())
    assert not result.cycle_one_complete
    assert report["global_completion_reason"] == ("reactivation_at_optimizer_step_cap")
    assert report["optimizer_steps_completed"] == 2
    assert report["optimizer_step_cap_respected"] is True


def test_confirmation_evaluation_safety_cap_is_distinct(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _build_three_mode_artifact(artifact)
    base = _boundary_scheduler(maximum_corridor_steps=5)
    scheduler = AdaptiveCorridorSchedulerConfig(
        controller=base.controller,
        mode_weights=base.mode_weights,
        global_freeze_confirmation_observations=2,
        maximum_confirmation_only_evaluations=2,
    )

    def unstable_confirmation(
        step: int, mode_id: str, metrics: object
    ) -> dict[str, float]:
        del metrics
        regressing = mode_id == "0" and step == 3
        loss = 1.0 if step == 0 else 0.8
        return {
            "corridor_loss": loss,
            "inside_corridor_rate": 0.5 if regressing else 0.97,
            "mean_distance_outside_corridor": 0.2 if regressing else 0.03,
            "worst_stat_violation": 0.2 if regressing else 0.03,
        }

    result = run_adaptive_corridor_pass(
        AdaptiveCorridorPassConfig(
            training_fingerprint_artifact=artifact,
            calibration_fingerprint_artifact=artifact,
            output_dir=tmp_path / "confirmation_cap",
            scheduler=scheduler,
            student_backend="tiny_debug",
            overwrite=True,
        ),
        calibration_override=unstable_confirmation,
    )
    report = json.loads(result.report_path.read_text())
    assert result.status == "incomplete"
    assert report["global_completion_reason"] == "confirmation_evaluation_cap"
    assert report["confirmation_only_evaluations_completed"] == 2
    assert report["optimizer_steps_completed"] == 2


def _boundary_scheduler(
    *, maximum_corridor_steps: int
) -> AdaptiveCorridorSchedulerConfig:
    return AdaptiveCorridorSchedulerConfig(
        controller=ModePlateauConfig(
            required_modes=("0", "1", "2"),
            minimum_observations=2,
            progress_window_observations=2,
            plateau_patience_observations=1,
            regression_patience_observations=2,
            reactivation_cooldown_observations=1,
            plateau_absolute_improvement_threshold=0.02,
            plateau_relative_improvement_threshold=0.02,
            maximum_corridor_steps=maximum_corridor_steps,
        ),
        mode_weights={"0": 1.0, "1": 1.0, "2": 1.0},
        global_freeze_confirmation_observations=2,
        maximum_confirmation_only_evaluations=8,
    )
