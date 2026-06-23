from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from qrwkv_xla.fingerprint.mode_plateau_controller import (
    ModePlateauConfig,
    MultiModePlateauController,
)

MODE_IDS = tuple(f"mode_{index}" for index in range(1, 7))


def _metrics(
    loss: float,
    *,
    inside: float = 0.97,
    distance: float = 0.03,
    violation: float = 0.03,
) -> dict[str, float]:
    return {
        "corridor_loss": loss,
        "inside_corridor_rate": inside,
        "mean_distance_outside_corridor": distance,
        "worst_stat_violation": violation,
    }


def _trajectory() -> list[dict[str, dict[str, float]]]:
    rows: list[dict[str, dict[str, float]]] = []
    losses = {
        "mode_1": [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "mode_2": [1.0, 0.8, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        "mode_3": [1.0] * 10,
        "mode_4": [1.0, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "mode_5": [1.0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
        "mode_6": [1.0, 0.9, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    }
    for step in range(10):
        row = {mode_id: _metrics(values[step]) for mode_id, values in losses.items()}
        if step in {4, 5}:
            row["mode_3"] = _metrics(1.0, inside=0.5, distance=0.2, violation=0.2)
        rows.append(row)
    return rows


def _run(
    config: ModePlateauConfig,
    trajectory: list[dict[str, dict[str, float]]],
    *,
    resume_at: int | None = None,
) -> MultiModePlateauController:
    controller = MultiModePlateauController(config)
    for step, observations in enumerate(trajectory):
        controller.observe_all(step=step, observations=observations)
        if resume_at == step:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "state.json"
                path.write_text(json.dumps(controller.to_dict()), encoding="utf-8")
                controller = MultiModePlateauController.load(path)
    return controller


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the P156.2 controller smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = ModePlateauConfig(
        required_modes=MODE_IDS,
        minimum_observations=3,
        progress_window_observations=3,
        plateau_patience_observations=2,
        regression_patience_observations=2,
        reactivation_cooldown_observations=1,
        plateau_absolute_improvement_threshold=0.02,
        plateau_relative_improvement_threshold=0.02,
    )
    trajectory = _trajectory()
    uninterrupted = _run(config, trajectory)
    resumed = _run(config, trajectory, resume_at=4)
    resume_equivalent = resumed.to_dict() == uninterrupted.to_dict()
    resumed.write_receipts(args.output_dir)
    report = resumed.report()
    if not report["global_cycle_one_complete"] or not resume_equivalent:
        raise RuntimeError("P156.2 synthetic smoke did not complete")
    reactivations = sum(state.reactivation_count for state in resumed.modes.values())
    print("phase=P156.2")
    print("status=pass")
    print(f"required_modes={len(config.required_modes)}")
    print(f"frozen_modes={len(resumed.frozen_mode_ids)}")
    print(f"reactivations={reactivations}")
    print(f"global_completion_step={resumed.global_completion_step}")
    print("resume_equivalent=true")


if __name__ == "__main__":
    main()
