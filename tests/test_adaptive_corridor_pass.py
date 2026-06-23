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
from scripts.run_adaptive_corridor_pass import _build_three_mode_artifact, run_smoke


def test_real_cpu_adaptive_corridor_smoke(tmp_path: Path) -> None:
    report = run_smoke(tmp_path / "p156_3")
    output = tmp_path / "p156_3" / "resumed"
    assert report["phase"] == "P156.3"
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
