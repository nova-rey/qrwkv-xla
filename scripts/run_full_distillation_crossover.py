from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from qrwkv_xla.fingerprint.adaptive_corridor_scheduler import (
    AdaptiveCorridorSchedulerConfig,
)
from qrwkv_xla.fingerprint.full_distillation_crossover import (
    ARMS,
    AdaptiveDiscovery,
    ArmCheckpoint,
    CheckpointEvaluation,
    FullDistillationCrossoverConfig,
    SharedInitialization,
    build_crossover_plan,
    run_full_distillation_crossover,
)
from qrwkv_xla.fingerprint.mode_plateau_controller import ModePlateauConfig
from qrwkv_xla.fingerprint.radjax_crossover_backend import (
    RadjaxCrossoverBackend,
    RadjaxCrossoverBackendConfig,
)
from qrwkv_xla.teachers import HFTeacherBackend


class TinyCpuCrossoverBackend:
    """Deterministic scalar student used only for the P156.4 plumbing smoke."""

    def create_shared_initialization(
        self, *, seed: int, output_dir: Path
    ) -> SharedInitialization:
        rng = np.random.default_rng(seed)
        params = rng.normal(size=(4,)).astype(np.float32)
        checkpoint = output_dir / "checkpoints" / "initial"
        checkpoint_hash = _save_checkpoint(checkpoint, params, {"step": 0})
        return SharedInitialization(
            checkpoint=checkpoint,
            parameter_tree_hash=_array_hash(params),
            student_config_hash=_hash_json({"kind": "tiny_scalar_student", "width": 4}),
            checkpoint_hash=checkpoint_hash,
        )

    def discover_adaptive_cycle_one(
        self,
        *,
        seed: int,
        shared_initialization: SharedInitialization,
        output_dir: Path,
    ) -> AdaptiveDiscovery:
        del seed
        params = np.load(shared_initialization.checkpoint / "params.npy")
        optimizer_state = np.zeros_like(params)
        for _ in range(2):
            gradient = params - 0.25
            optimizer_state = 0.9 * optimizer_state + gradient
            params = params - 0.02 * optimizer_state
        checkpoint = output_dir / "checkpoints" / "adaptive_corridor_final_checkpoint"
        checkpoint_hash = _save_checkpoint(checkpoint, params, {"step": 2})
        return AdaptiveDiscovery(
            cycle_one_complete=True,
            optimizer_steps_completed=2,
            completion_reason="all_required_modes_stably_frozen",
            checkpoint=checkpoint,
            checkpoint_hash=checkpoint_hash,
            controller_config_hash=_hash_json({"smoke": "controller"}),
            scheduler_config_hash=_hash_json({"smoke": "scheduler"}),
            corridor_optimizer_state_hash=_array_hash(optimizer_state),
            mode_freeze_steps={"0": 1, "1": 2, "2": 2},
            reactivation_steps={"0": [], "1": [], "2": [1]},
            confirmation_only_evaluations=2,
        )

    def train_arm(
        self,
        *,
        arm: str,
        seed: int,
        shared_initialization: SharedInitialization,
        adaptive_discovery: AdaptiveDiscovery,
        checkpoint_steps: tuple[int, ...],
        output_dir: Path,
    ) -> list[ArmCheckpoint]:
        del seed
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        if arm == "adaptive_two_cycle":
            params = np.load(adaptive_discovery.checkpoint / "params.npy")
            start = int(adaptive_discovery.optimizer_steps_completed or 0)
            parent_hash = adaptive_discovery.checkpoint_hash
        else:
            params = np.load(shared_initialization.checkpoint / "params.npy")
            start = 0
            parent_hash = shared_initialization.checkpoint_hash
        optimizer_state = np.zeros_like(params)
        initial_optimizer_hash = _array_hash(optimizer_state)
        checkpoints = []
        learning_target = {
            "vanilla": 0.0,
            "exemplar_only": 0.15,
            "adaptive_two_cycle": 0.1,
        }[arm]
        for total_step in range(start, checkpoint_steps[-1] + 1):
            if total_step > start:
                gradient = params - learning_target
                optimizer_state = 0.8 * optimizer_state + gradient
                params = params - 0.01 * optimizer_state
            if total_step not in checkpoint_steps:
                continue
            checkpoint = output_dir / f"step_{total_step}" / "checkpoints" / "final"
            checkpoint_hash = _save_checkpoint(
                checkpoint, params, {"step": total_step, "arm": arm}
            )
            checkpoints.append(
                ArmCheckpoint(
                    arm=arm,
                    total_step=total_step,
                    checkpoint=checkpoint,
                    checkpoint_hash=checkpoint_hash,
                    parent_checkpoint_hash=parent_hash,
                    initial_parameter_hash=shared_initialization.parameter_tree_hash,
                    corridor_steps=start if arm == "adaptive_two_cycle" else 0,
                    exemplar_steps=(
                        total_step - start
                        if arm == "adaptive_two_cycle"
                        else total_step
                        if arm == "exemplar_only"
                        else 0
                    ),
                    vanilla_steps=total_step if arm == "vanilla" else 0,
                    optimizer_initial_state_hash=initial_optimizer_hash,
                    optimizer_final_state_hash=_array_hash(optimizer_state),
                    resource_accounting={
                        "optimizer_steps": total_step,
                        "training_records_consumed": max(total_step, 0),
                        "training_tokens_consumed": max(total_step, 0) * 8,
                        "teacher_artifact_bytes_consumed": (
                            0 if arm == "vanilla" else max(total_step, 0) * 16
                        ),
                        "corridor_artifact_bytes_consumed": start * 16,
                        "exemplar_artifact_bytes_consumed": (
                            max(total_step - start, 0) * 16
                            if arm == "adaptive_two_cycle"
                            else total_step * 16
                            if arm == "exemplar_only"
                            else 0
                        ),
                        "active_mode_step_equivalents": start * 2,
                        "source_text_bytes_consumed": (
                            max(total_step, 0) * 8 if arm == "vanilla" else 0
                        ),
                        "wall_clock_training_seconds": 0.0,
                        "checkpoint_seconds": 0.0,
                        "total_seconds": 0.0,
                        "expected_fresh_exemplar_optimizer_hash": (
                            initial_optimizer_hash
                            if arm == "adaptive_two_cycle"
                            else None
                        ),
                        "actual_initial_exemplar_optimizer_hash": (
                            initial_optimizer_hash
                            if arm == "adaptive_two_cycle" and total_step > start
                            else None
                        ),
                        "cycle_two_optimizer_instantiated": (
                            arm == "adaptive_two_cycle" and total_step > start
                        ),
                        "fresh_optimizer_exact_match": (
                            True
                            if arm == "adaptive_two_cycle" and total_step > start
                            else None
                        ),
                        "fresh_optimizer_proof_status": (
                            "proven"
                            if arm == "adaptive_two_cycle" and total_step > start
                            else "not_applicable"
                            if arm == "adaptive_two_cycle"
                            else None
                        ),
                    },
                )
            )
        return checkpoints

    def evaluate_checkpoint(
        self,
        *,
        seed: int,
        checkpoint: ArmCheckpoint,
        final_test_artifact: Path,
        output_dir: Path,
    ) -> CheckpointEvaluation:
        del output_dir
        params = np.load(checkpoint.checkpoint / "params.npy")
        record_ids = tuple(
            line.strip()
            for line in (final_test_artifact / "records.txt").read_text().splitlines()
            if line.strip()
        )
        base = float(np.mean(np.square(params - 0.1)))
        records = [
            {
                "record_id": record_id,
                "teacher_student_kl": base + index * 1e-4,
            }
            for index, record_id in enumerate(record_ids)
        ]
        kl = float(np.mean([row["teacher_student_kl"] for row in records]))
        teacher_forced = {
            "teacher_student_kl": kl,
            "top1_agreement": float(max(0.0, 1.0 - kl)),
            "topk_overlap": float(max(0.0, 1.0 - kl / 2)),
            "teacher_entropy": 1.0,
            "student_entropy": 1.0 + kl,
            "entropy_absolute_error": kl,
            "held_out_exemplar_loss": kl,
            "held_out_corridor_loss": kl,
            "inside_all_rate": float(max(0.0, 1.0 - kl)),
            "mean_distance_outside_corridor": kl,
            "worst_stat_violation": kl,
        }
        student_prefix = {
            "student_prefix_teacher_student_kl": kl * 1.1,
            "student_prefix_top1_agreement": float(max(0.0, 1.0 - kl)),
            "student_prefix_topk_overlap": float(max(0.0, 1.0 - kl / 2)),
            "student_prefix_entropy_error": kl,
            "trajectory_length": 2,
            "contexts_generated_by_student": True,
        }
        free_running = {
            "decoding_policies": ["greedy", "temperature_0.8_seeded"],
            "teacher_likelihood_of_student_continuation": -kl,
            "token_frequency_distance": kl,
            "repetition_rate": 0.0,
            "sequence_entropy": 1.0,
            "mode_occupancy": {"0": 1.0},
            "length_distribution": {"mean": 2.0},
            "exact_text_match_required": False,
            "seed": seed,
        }
        return CheckpointEvaluation(
            teacher_forced=teacher_forced,
            teacher_forced_records=records,
            student_prefix=student_prefix,
            free_running=free_running,
            final_test_record_order_hash=_hash_json(record_ids),
        )


def build_smoke_inputs(root: Path) -> dict[str, Path]:
    paths = {}
    for role in ("training", "calibration", "final_test"):
        path = root / role
        path.mkdir(parents=True, exist_ok=True)
        (path / "records.txt").write_text(
            "record-0\nrecord-1\nrecord-2\n", encoding="utf-8"
        )
        paths[role] = path
    for name in ("source_texts", "student_config", "selected_profile_receipt"):
        path = root / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        paths[name] = path
    return paths


def _save_checkpoint(path: Path, params: np.ndarray, metadata: dict) -> str:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "params.npy", params)
    (path / "checkpoint.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return _directory_hash(path)


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _hash_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    ):
        digest.update(item.name.encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the P156.4 crossover harness")
    parser.add_argument("--training-artifact", type=Path, required=True)
    parser.add_argument("--calibration-artifact", type=Path, required=True)
    parser.add_argument("--final-test-artifact", type=Path, required=True)
    parser.add_argument("--source-texts", type=Path, required=True)
    parser.add_argument("--student-config", type=Path, required=True)
    parser.add_argument("--selected-profile-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--checkpoint-fractions", default="0.0,0.1,0.25,0.5,1.0")
    parser.add_argument(
        "--target-quality-thresholds", default='{"teacher_student_kl":1.0}'
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=1564)
    parser.add_argument("--require-backend", choices=("cpu",), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--implementation-smoke", action="store_true")
    parser.add_argument(
        "--backend", choices=("tiny_cpu_smoke", "radjax"), default="radjax"
    )
    parser.add_argument("--teacher-model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--student-backend", default="tiny_debug")
    parser.add_argument("--max-new-training-runs", type=int)
    parser.add_argument("--maximum-steps", type=int)
    parser.add_argument("--enable-convergence", action="store_true")
    args = parser.parse_args()
    config = FullDistillationCrossoverConfig(
        training_artifact=args.training_artifact,
        calibration_artifact=args.calibration_artifact,
        final_test_artifact=args.final_test_artifact,
        source_texts=args.source_texts,
        student_config=args.student_config,
        selected_profile_receipt=args.selected_profile_receipt,
        output_dir=args.output_dir,
        seeds=tuple(int(value) for value in args.seeds.split(",")),
        checkpoint_fractions=tuple(
            float(value) for value in args.checkpoint_fractions.split(",")
        ),
        target_quality_thresholds=json.loads(args.target_quality_thresholds),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        require_backend=args.require_backend,
        resume=args.resume,
        max_new_training_runs=args.max_new_training_runs,
        maximum_steps=args.maximum_steps,
        convergence_enabled=args.enable_convergence,
    )
    if args.plan_only:
        print(
            json.dumps(build_crossover_plan(config).to_dict(), indent=2, sort_keys=True)
        )
        return
    if args.backend == "tiny_cpu_smoke":
        if not args.implementation_smoke:
            raise SystemExit(
                "tiny_cpu_smoke backend is allowed only with --implementation-smoke"
            )
        backend = TinyCpuCrossoverBackend()
    else:
        mode_payload = json.loads(
            (args.training_artifact / "modes.json").read_text(encoding="utf-8")
        )
        mode_ids = tuple(str(mode["mode_id"]) for mode in mode_payload["modes"])
        scheduler = AdaptiveCorridorSchedulerConfig(
            controller=ModePlateauConfig(
                required_modes=mode_ids,
                maximum_corridor_steps=args.maximum_steps or 100,
            ),
            mode_weights={mode_id: 1.0 for mode_id in mode_ids},
        )
        backend = RadjaxCrossoverBackend(
            RadjaxCrossoverBackendConfig(
                training_artifact=args.training_artifact,
                calibration_artifact=args.calibration_artifact,
                final_test_artifact=args.final_test_artifact,
                source_texts=args.source_texts,
                selected_profile_receipt=args.selected_profile_receipt,
                receipt_root=args.output_dir,
                adaptive_scheduler=scheduler,
                teacher_backend=HFTeacherBackend(
                    model_id=args.teacher_model, local_files_only=True
                ),
                student_backend=args.student_backend,
            )
        )
    report = run_full_distillation_crossover(config, backend=backend)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
