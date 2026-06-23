from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import jax
import numpy as np

from qrwkv_xla.checkpointing import load_checkpoint
from qrwkv_xla.fingerprint.adaptive_corridor_pass import (
    AdaptiveCorridorPassConfig,
    run_adaptive_corridor_pass,
    write_resume_equivalence_receipt,
)
from qrwkv_xla.fingerprint.adaptive_corridor_scheduler import (
    AdaptiveCorridorSchedulerConfig,
)
from qrwkv_xla.fingerprint.mode_plateau_controller import ModePlateauConfig


def run_smoke(output_dir: Path) -> dict[str, object]:
    artifact = output_dir / "three_mode_artifact"
    _build_three_mode_artifact(artifact)
    scheduler_config = AdaptiveCorridorSchedulerConfig(
        controller=ModePlateauConfig(
            required_modes=("0", "1", "2"),
            minimum_observations=2,
            progress_window_observations=2,
            plateau_patience_observations=1,
            regression_patience_observations=2,
            reactivation_cooldown_observations=1,
            plateau_absolute_improvement_threshold=0.02,
            plateau_relative_improvement_threshold=0.02,
            maximum_corridor_steps=10,
        ),
        mode_weights={"0": 1.0, "1": 1.0, "2": 1.0},
        global_freeze_confirmation_observations=2,
    )
    common = {
        "training_fingerprint_artifact": artifact,
        "calibration_fingerprint_artifact": artifact,
        "scheduler": scheduler_config,
        "evaluation_interval_steps": 1,
        "checkpoint_interval_steps": 2,
        "optimizer": "sgd",
        "learning_rate": 1e-3,
        "student_backend": "tiny_debug",
        "overwrite": True,
    }
    uninterrupted = run_adaptive_corridor_pass(
        AdaptiveCorridorPassConfig(output_dir=output_dir / "uninterrupted", **common),
        calibration_override=_calibration_override,
    )
    interrupted = run_adaptive_corridor_pass(
        AdaptiveCorridorPassConfig(output_dir=output_dir / "resumed", **common),
        calibration_override=_calibration_override,
        interrupt_after_optimizer_step=2,
    )
    if interrupted.interruption_checkpoint is None:
        raise RuntimeError("adaptive smoke did not write interruption checkpoint")
    resumed = run_adaptive_corridor_pass(
        AdaptiveCorridorPassConfig(
            output_dir=output_dir / "resumed",
            resume_checkpoint=interrupted.interruption_checkpoint,
            **common,
        ),
        calibration_override=_calibration_override,
    )
    uninterrupted_checkpoint = load_checkpoint(uninterrupted.final_checkpoint)
    resumed_checkpoint = load_checkpoint(resumed.final_checkpoint)
    params_equal = _trees_equal(
        uninterrupted_checkpoint.params, resumed_checkpoint.params
    )
    uninterrupted_state = _read_json(
        uninterrupted.final_checkpoint / "adaptive_state.json"
    )["scheduler_state"]
    resumed_state = _read_json(resumed.final_checkpoint / "adaptive_state.json")[
        "scheduler_state"
    ]
    scheduler_equal = uninterrupted_state == resumed_state
    equivalent = params_equal and scheduler_equal
    comparison = {
        "final_parameters_equal": params_equal,
        "scheduler_state_equal": scheduler_equal,
        "uninterrupted_optimizer_step": uninterrupted_checkpoint.manifest.step,
        "resumed_optimizer_step": resumed_checkpoint.manifest.step,
    }
    write_resume_equivalence_receipt(
        uninterrupted.output_dir,
        equivalent=equivalent,
        comparison=comparison,
    )
    write_resume_equivalence_receipt(
        resumed.output_dir,
        equivalent=equivalent,
        comparison=comparison,
    )
    report = _read_json(resumed.report_path)
    if not resumed.cycle_one_complete or not equivalent:
        raise RuntimeError("P156.3 adaptive smoke failed")
    return report


def _calibration_override(
    step: int, mode_id: str, real_metrics: object
) -> dict[str, float]:
    del real_metrics
    losses = {
        "0": {0: 1.0, 1: 1.0},
        "1": {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4, 4: 0.4},
        "2": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0},
    }
    loss = losses[mode_id].get(step, list(losses[mode_id].values())[-1])
    regressing = mode_id == "2" and step in {2, 3}
    return {
        "corridor_loss": loss,
        "inside_corridor_rate": 0.5 if regressing else 0.97,
        "mean_distance_outside_corridor": 0.2 if regressing else 0.03,
        "worst_stat_violation": 0.2 if regressing else 0.03,
    }


def _build_three_mode_artifact(path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "behavioral_fingerprint"
        / "v0_1_valid_tiny"
    )
    if path.exists():
        shutil.rmtree(path)
    shutil.copytree(source, path)
    modes_path = path / "modes.json"
    modes = _read_json(modes_path)
    third_mode = json.loads(json.dumps(modes["modes"][0]))
    third_mode.update(
        {
            "mode_id": 2,
            "name": "reactivating_synthetic_mode",
            "description": "P156.3 CPU smoke reactivation mode.",
        }
    )
    modes["modes"].append(third_mode)
    _write_json(modes_path, modes)
    shard = path / "targets" / "targets-00000.jsonl"
    rows = [json.loads(line) for line in shard.read_text().splitlines() if line]
    third_rows = []
    for index, row in enumerate(rows[:2]):
        copied = json.loads(json.dumps(row))
        copied["example_id"] = f"p1563-mode2-{index}"
        copied["mode_id"] = 2
        third_rows.append(copied)
    rows.extend(third_rows)
    shard.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_path = path / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["created_by"] = "p156_3_adaptive_cpu_smoke"
    manifest["sequence"]["target_positions"] = len(rows)
    manifest["target_shards"][0]["num_records"] = len(rows)
    _write_json(manifest_path, manifest)


def _trees_equal(left: object, right: object) -> bool:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    return left_tree == right_tree and all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the P156.3 adaptive CPU smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_smoke(args.output_dir)
    print("phase=P156.3.1")
    print(f"status={report['status']}")
    print(f"optimizer_steps={report['optimizer_steps_completed']}")
    print(f"reactivations={report['reactivation_count']}")
    print(f"global_completion_step={report['global_completion_step']}")
    print(f"fraction_mode_work_saved={report['fraction_mode_work_saved']:.6f}")
    print(f"resume_equivalent={str(report['resume_equivalent']).lower()}")


if __name__ == "__main__":
    main()
