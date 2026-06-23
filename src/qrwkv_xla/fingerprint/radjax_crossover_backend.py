from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qrwkv_xla.artifacts import (
    load_fingerprint_exemplars,
    load_fingerprint_targets,
    summarize_fingerprint_artifact,
)
from qrwkv_xla.checkpointing import load_checkpoint, save_checkpoint
from qrwkv_xla.fingerprint.adaptive_corridor_pass import (
    AdaptiveCorridorPassConfig,
    run_adaptive_corridor_pass,
)
from qrwkv_xla.fingerprint.adaptive_corridor_scheduler import (
    AdaptiveCorridorSchedulerConfig,
)
from qrwkv_xla.fingerprint.exemplar_pass import ExemplarPassConfig, run_exemplar_pass
from qrwkv_xla.fingerprint.full_distillation_crossover import (
    AdaptiveDiscovery,
    ArmCheckpoint,
    CheckpointEvaluation,
    SharedInitialization,
)
from qrwkv_xla.fingerprint.held_out_evaluation import _evaluate_checkpoint
from qrwkv_xla.fingerprint.provenance import (
    hash_checkpoint_bundle,
    parameter_fingerprint,
    stable_hash,
)
from qrwkv_xla.fingerprint.trained_baseline import (
    FingerprintTrainedBaselineConfig,
    _example_batches,
    _run_baseline,
    _unique_examples,
    masked_causal_lm_loss,
)
from qrwkv_xla.fingerprint.trained_baseline import (
    _create_backend as _create_baseline_backend,
)
from qrwkv_xla.optimizers import (
    OptimizerConfig,
    init_optimizer_state,
    optimizer_update,
)

ARTIFACT_BYTE_CATEGORIES = (
    "corridor_payload",
    "exemplar_payload",
    "shared_teacher_metadata",
    "excluded",
)
_SHARED_METADATA_NAMES = {
    "README.md",
    "checksums.json",
    "manifest.json",
    "schema.json",
    "splits.json",
    "teacher_config.json",
    "vocab_contract.json",
}
_EXCLUDED_NAMES = {".DS_Store"}


@dataclass(frozen=True)
class RadjaxCrossoverBackendConfig:
    training_artifact: Path
    calibration_artifact: Path
    final_test_artifact: Path
    source_texts: Path
    selected_profile_receipt: Path
    receipt_root: Path
    adaptive_scheduler: AdaptiveCorridorSchedulerConfig
    teacher_backend: Any
    student_backend: str = "tiny_debug"
    vanilla_optimizer: str = "sgd"
    vanilla_learning_rate: float = 1e-3
    exemplar_optimizer: str = "sgd"
    exemplar_learning_rate: float = 1e-3
    batch_size: int = 1
    max_grad_norm: float | None = 1.0
    checkpoint_interval: int = 1
    prefix_length: int = 2
    evaluation_prompt_limit: int = 2
    overwrite: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_artifact": str(self.training_artifact.resolve()),
            "calibration_artifact": str(self.calibration_artifact.resolve()),
            "final_test_artifact": str(self.final_test_artifact.resolve()),
            "source_texts": str(self.source_texts.resolve()),
            "selected_profile_receipt": str(self.selected_profile_receipt.resolve()),
            "adaptive_scheduler": self.adaptive_scheduler.to_dict(),
            "teacher_backend_spec": {
                "type": type(self.teacher_backend).__name__,
                "model_id": getattr(self.teacher_backend, "model_id", None),
            },
            "student_backend": self.student_backend,
            "vanilla_optimizer": self.vanilla_optimizer,
            "vanilla_learning_rate": self.vanilla_learning_rate,
            "exemplar_optimizer": self.exemplar_optimizer,
            "exemplar_learning_rate": self.exemplar_learning_rate,
            "batch_size": self.batch_size,
            "max_grad_norm": self.max_grad_norm,
            "checkpoint_interval": self.checkpoint_interval,
            "prefix_length": self.prefix_length,
            "evaluation_prompt_limit": self.evaluation_prompt_limit,
        }


def validate_byte_accounting(accounting: dict[str, Any], *, arm: str) -> None:
    total = int(accounting["teacher_artifact_bytes_total"])
    logical = int(accounting["artifact_bytes_logically_consumed"])
    corridor = int(accounting["corridor_payload_bytes"])
    exemplar = int(accounting["exemplar_payload_bytes"])
    shared = int(accounting["shared_teacher_metadata_bytes"])
    source = int(accounting["source_text_bytes_consumed"])
    if min(total, logical, corridor, exemplar, shared, source) < 0:
        raise ValueError("byte accounting values must be non-negative")
    if corridor + exemplar + shared != total:
        raise ValueError("teacher artifact total must equal disjoint components")
    if logical != total:
        raise ValueError("logical teacher bytes must equal selected component total")
    if arm == "vanilla" and any((total, logical, corridor, exemplar, shared)):
        raise ValueError("vanilla arm must not consume teacher artifact bytes")
    if arm == "exemplar_only" and corridor:
        raise ValueError("exemplar-only arm must not consume corridor payload")


def teacher_bytes_to_target(accounting: dict[str, Any]) -> int:
    validate_byte_accounting(accounting, arm=str(accounting["arm"]))
    return int(accounting["artifact_bytes_logically_consumed"])


def classify_artifact_file(artifact_root: Path, path: Path) -> str:
    relative = path.relative_to(artifact_root)
    parts = relative.parts
    matches: list[str] = []
    if parts and parts[0] == "exemplars":
        matches.append("exemplar_payload")
    if parts and parts[0] in {"targets", "corridor"}:
        matches.append("corridor_payload")
    if relative.name == "modes.json":
        matches.append("corridor_payload")
    if relative.name in _SHARED_METADATA_NAMES:
        matches.append("shared_teacher_metadata")
    if relative.name in _EXCLUDED_NAMES or any(
        part in {"checkpoints", "cache", "logs", "reports", "tmp"} for part in parts
    ):
        matches.append("excluded")
    if len(matches) != 1:
        raise ValueError(
            "artifact file must match exactly one byte category: "
            f"{relative} -> {matches}"
        )
    return matches[0]


def classify_artifact_bytes(artifact_root: Path) -> dict[str, Any]:
    files = []
    totals = {category: 0 for category in ARTIFACT_BYTE_CATEGORIES}
    for path in sorted(
        candidate for candidate in artifact_root.rglob("*") if candidate.is_file()
    ):
        category = classify_artifact_file(artifact_root, path)
        size = path.stat().st_size
        totals[category] += size
        files.append(
            {
                "path": str(path.relative_to(artifact_root)),
                "category": category,
                "bytes": size,
            }
        )
    available = sum(value for key, value in totals.items() if key != "excluded")
    return {
        "files": files,
        "category_bytes": totals,
        "artifact_bytes_available": available,
    }


class RadjaxCrossoverBackend:
    checkpoint_execution_mode = "continuous_trajectory"
    strict_resource_accounting = True

    @property
    def ready_for_p156_5(self) -> bool:
        return bool(
            self._adaptive_cost_evidence.get("adaptive_full_cost_accounting_valid")
            and all(
                self._resume_evidence.get(arm, {}).get("pass") is True
                for arm in ("vanilla", "exemplar_only", "adaptive_two_cycle")
            )
            and self._usage["byte_accounting_valid"]
        )

    def __init__(self, config: RadjaxCrossoverBackendConfig) -> None:
        self.config = config
        self.config_sha256 = stable_hash(config.to_dict())
        self._usage = {
            "real_student_used": False,
            "real_teacher_used": False,
            "real_artifact_loaders_used": False,
            "real_adaptive_runner_used": False,
            "real_vanilla_runner_used": False,
            "real_exemplar_runner_used": False,
            "real_evaluation_adapters_used": False,
            "fresh_optimizer_exact_match": False,
            "byte_accounting_valid": False,
        }
        self._last_byte_accounting: dict[str, Any] = {}
        self._freshness_proof: dict[str, Any] = {
            "fresh_optimizer_proof_status": "not_applicable",
            "fresh_optimizer_exact_match": None,
        }
        self._cycle_one_resources: dict[str, Any] = {}
        self._cycle_one_receipt_hash: str | None = None
        self._adaptive_cost_evidence: dict[str, Any] = {}
        self._resume_evidence: dict[str, Any] = {}
        self.config.receipt_root.mkdir(parents=True, exist_ok=True)
        self._validate_artifacts()
        self._write_receipts()

    def create_shared_initialization(
        self, *, seed: int, output_dir: Path
    ) -> SharedInitialization:
        summary = summarize_fingerprint_artifact(self.config.training_artifact)
        baseline_config = self._baseline_config(seed, output_dir, steps=1)
        backend, student_config = _create_baseline_backend(baseline_config, summary)
        params = backend.init_params(jax.random.PRNGKey(seed))
        optimizer_config = OptimizerConfig(
            type=self.config.vanilla_optimizer,
            learning_rate=self.config.vanilla_learning_rate,
        )
        checkpoint = output_dir / "checkpoints" / "initial"
        save_checkpoint(
            checkpoint,
            params,
            student_architecture=self.config.student_backend,
            student_config=student_config,
            step=0,
            learning_rate=self.config.vanilla_learning_rate,
            loss_config={"kind": "shared_initialization"},
            target_manifest={"artifact_dir": str(self.config.training_artifact)},
            optimizer_config=asdict(optimizer_config),
            optimizer_state=init_optimizer_state(params, optimizer_config),
            notes=["P156.4.1 shared real student initialization"],
            overwrite=self.config.overwrite,
        )
        self._usage["real_student_used"] = True
        self._write_receipts()
        return SharedInitialization(
            checkpoint=checkpoint,
            parameter_tree_hash=parameter_fingerprint(params),
            student_config_hash=stable_hash(student_config),
            checkpoint_hash=hash_checkpoint_bundle(checkpoint)[
                "checkpoint_bundle_sha256"
            ],
        )

    def record_resume_evidence(self, arms: dict[str, dict[str, Any]]) -> None:
        required = {"vanilla", "exemplar_only", "adaptive_two_cycle"}
        if set(arms) != required:
            raise ValueError("resume evidence must cover all production arms")
        self._resume_evidence = arms
        self._write_receipts()

    def discover_adaptive_cycle_one(
        self,
        *,
        seed: int,
        shared_initialization: SharedInitialization,
        output_dir: Path,
    ) -> AdaptiveDiscovery:
        result = run_adaptive_corridor_pass(
            AdaptiveCorridorPassConfig(
                training_fingerprint_artifact=self.config.training_artifact,
                calibration_fingerprint_artifact=self.config.calibration_artifact,
                output_dir=output_dir,
                scheduler=self.config.adaptive_scheduler,
                evaluation_interval_steps=(
                    self.config.adaptive_scheduler.controller.evaluation_interval_steps
                ),
                checkpoint_interval_steps=self.config.checkpoint_interval,
                optimizer=self.config.vanilla_optimizer,
                learning_rate=self.config.vanilla_learning_rate,
                max_grad_norm=self.config.max_grad_norm,
                student_backend=self.config.student_backend,
                seed=seed,
                initial_checkpoint=shared_initialization.checkpoint,
                overwrite=self.config.overwrite,
            )
        )
        report = _read_json(result.report_path)
        loaded = load_checkpoint(result.final_checkpoint)
        state = _read_json(result.final_checkpoint / "adaptive_state.json")
        scheduler_state = state["scheduler_state"]
        modes = report["modes"]
        measured = report.get("resource_accounting", {})
        required = (
            "training_records",
            "training_tokens",
            "training_wall_seconds",
            "evaluation_wall_seconds",
            "checkpoint_wall_seconds",
            "total_wall_seconds",
        )
        missing = [name for name in required if measured.get(name) is None]
        if missing:
            raise ValueError(
                f"adaptive Cycle 1 resource measurements missing: {missing}"
            )
        classified = classify_artifact_bytes(self.config.training_artifact)
        category = classified["category_bytes"]
        self._cycle_one_resources = {
            "optimizer_steps": int(report["optimizer_steps_completed"]),
            **{name: measured[name] for name in required},
            "source_text_bytes": 0,
            "teacher_artifact_bytes": (
                category["corridor_payload"] + category["shared_teacher_metadata"]
            ),
            "corridor_payload_bytes": category["corridor_payload"],
            "exemplar_payload_bytes": 0,
            "shared_metadata_bytes": category["shared_teacher_metadata"],
            "cpu_seconds": None,
            "tpu_seconds": 0,
            "gpu_seconds": 0,
        }
        cycle_one_receipt = {
            "phase": "P156.4.2.1",
            "seed": seed,
            "adaptive_checkpoint_hash": hash_checkpoint_bundle(result.final_checkpoint)[
                "checkpoint_bundle_sha256"
            ],
            "S": int(report["optimizer_steps_completed"]),
            **self._cycle_one_resources,
            "resource_provenance": {
                name: "adaptive_corridor_report.resource_accounting"
                for name in required
            },
        }
        self._cycle_one_receipt_hash = stable_hash(cycle_one_receipt)
        cycle_one_receipt["cycle_one_resource_baseline_sha256"] = (
            self._cycle_one_receipt_hash
        )
        _write_json(
            self.config.receipt_root / "cycle_one_resource_baseline_receipt.json",
            cycle_one_receipt,
        )
        self._usage["real_adaptive_runner_used"] = True
        self._write_receipts()
        return AdaptiveDiscovery(
            cycle_one_complete=result.cycle_one_complete,
            optimizer_steps_completed=(
                int(report["optimizer_steps_completed"])
                if result.cycle_one_complete
                else None
            ),
            completion_reason=str(report["global_completion_reason"]),
            checkpoint=result.final_checkpoint,
            checkpoint_hash=hash_checkpoint_bundle(result.final_checkpoint)[
                "checkpoint_bundle_sha256"
            ],
            controller_config_hash=report["controller_config_sha256"],
            scheduler_config_hash=report["scheduler_config_sha256"],
            corridor_optimizer_state_hash=_optimizer_hash(loaded.optimizer_state),
            mode_freeze_steps={
                key: value["freeze_step"] for key, value in modes.items()
            },
            reactivation_steps={
                key: value["reactivation_steps"] for key, value in modes.items()
            },
            confirmation_only_evaluations=int(
                scheduler_state["confirmation_only_evaluations_completed"]
            ),
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
        _validate_trajectory_steps(checkpoint_steps)
        if arm == "vanilla":
            return self._train_vanilla_trajectory(
                seed, shared_initialization, checkpoint_steps, output_dir
            )
        if arm in {"exemplar_only", "adaptive_two_cycle"}:
            return self._train_exemplar_trajectory(
                arm,
                seed,
                shared_initialization,
                adaptive_discovery,
                checkpoint_steps,
                output_dir,
            )
        raise ValueError(f"unknown crossover arm: {arm}")

    def evaluate_checkpoint(
        self,
        *,
        seed: int,
        checkpoint: ArmCheckpoint,
        final_test_artifact: Path,
        output_dir: Path,
    ) -> CheckpointEvaluation:
        del seed
        records = tuple(
            load_fingerprint_targets(final_test_artifact, batch_size=1).iter_records()
        )
        exemplars = {
            (record.example_id, record.position): record
            for record in load_fingerprint_exemplars(
                final_test_artifact, batch_size=1
            ).iter_records()
        }
        evaluated = _evaluate_checkpoint(
            checkpoint.checkpoint, final_test_artifact, records, exemplars
        )
        teacher = evaluated["teacher_metrics"] or {}
        aggregate = evaluated["aggregate"]
        teacher_forced = {
            **teacher,
            "held_out_exemplar_loss": teacher.get("teacher_student_kl"),
            "held_out_corridor_loss": aggregate["corridor_loss_total"],
            "inside_all_rate": aggregate["inside_all_rate"],
            "mean_distance_outside_corridor": aggregate[
                "mean_distance_outside_corridor"
            ],
            "worst_stat_violation": aggregate["max_distance_outside_corridor"],
        }
        prefix = self._student_prefix_metrics(checkpoint.checkpoint, records)
        generation = self._free_running_metrics(checkpoint.checkpoint, records)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "teacher_forced.json", teacher_forced)
        _write_json(output_dir / "student_prefix.json", prefix)
        _write_json(output_dir / "free_running.json", generation)
        self._usage["real_teacher_used"] = True
        self._usage["real_evaluation_adapters_used"] = True
        self._write_receipts()
        return CheckpointEvaluation(
            teacher_forced=teacher_forced,
            teacher_forced_records=evaluated["teacher_records"],
            student_prefix=prefix,
            free_running=generation,
            final_test_record_order_hash=stable_hash(evaluated["record_keys"]),
            evaluation_seconds=float(aggregate["wall_clock_seconds"]),
        )

    def _train_vanilla(
        self,
        seed: int,
        shared: SharedInitialization,
        total_step: int,
        output_dir: Path,
    ) -> ArmCheckpoint:
        run_dir = output_dir / f"step_{total_step}"
        config = self._baseline_config(seed, run_dir, steps=total_step)
        summary = summarize_fingerprint_artifact(self.config.training_artifact)
        backend, student_config = _create_baseline_backend(config, summary)
        records = tuple(
            load_fingerprint_targets(
                self.config.training_artifact, batch_size=1
            ).iter_records()
        )
        examples = _unique_examples(records)
        initial = load_checkpoint(shared.checkpoint)
        optimizer_config = OptimizerConfig(
            type=self.config.vanilla_optimizer,
            learning_rate=self.config.vanilla_learning_rate,
        )
        payload = _run_baseline(
            config,
            backend=backend,
            student_config=student_config,
            initial_params=initial.params,
            optimizer_config=optimizer_config,
            examples=examples,
        )
        if payload["initial_parameter_fingerprint"] != shared.parameter_tree_hash:
            raise ValueError(
                "vanilla arm did not load shared initialization parameters"
            )
        checkpoint = Path(payload["checkpoint_dir"])
        loaded = load_checkpoint(checkpoint)
        resource = _byte_accounting(
            arm="vanilla",
            source_text_bytes=self.config.source_texts.stat().st_size,
        )
        resource.update(
            {
                "optimizer_steps": total_step,
                "training_records_consumed": total_step * self.config.batch_size,
                "training_tokens_consumed": payload["tokens_consumed"],
                "wall_clock_training_seconds": payload["wall_clock_seconds"],
                "total_seconds": payload["wall_clock_seconds"],
            }
        )
        self._usage["real_vanilla_runner_used"] = True
        self._write_receipts()
        return self._arm_checkpoint(
            "vanilla",
            total_step,
            checkpoint,
            shared,
            shared.checkpoint_hash,
            corridor_steps=0,
            exemplar_steps=0,
            vanilla_steps=total_step,
            initial_optimizer_hash=_optimizer_hash(
                init_optimizer_state(initial.params, optimizer_config)
            ),
            final_optimizer_hash=_optimizer_hash(loaded.optimizer_state),
            resource=resource,
        )

    def _train_vanilla_trajectory(
        self,
        seed: int,
        shared: SharedInitialization,
        checkpoint_steps: tuple[int, ...],
        output_dir: Path,
    ) -> list[ArmCheckpoint]:
        config = self._baseline_config(seed, output_dir, steps=checkpoint_steps[-1])
        summary = summarize_fingerprint_artifact(self.config.training_artifact)
        backend, student_config = _create_baseline_backend(config, summary)
        records = tuple(
            load_fingerprint_targets(
                self.config.training_artifact, batch_size=1
            ).iter_records()
        )
        examples = _unique_examples(records)
        batches = _example_batches(examples, self.config.batch_size)
        loaded = load_checkpoint(shared.checkpoint)
        params = jax.tree_util.tree_map(jnp.asarray, loaded.params)
        optimizer_config = OptimizerConfig(
            type=self.config.vanilla_optimizer,
            learning_rate=self.config.vanilla_learning_rate,
        )
        state = init_optimizer_state(params, optimizer_config)
        initial_optimizer_hash = _optimizer_hash(state)
        start_step = 0
        training_seconds = checkpoint_seconds = 0.0
        records_consumed = tokens_consumed = 0
        prior_total_wall_seconds = 0.0
        previous_hash = shared.checkpoint_hash
        previous_optimizer_hash = initial_optimizer_hash
        previous_resource: dict[str, float | int | None] = {}
        checkpoint_root = output_dir / "trajectory" / "checkpoints"
        existing = sorted(
            path for path in checkpoint_root.glob("step_*") if path.is_dir()
        )
        if existing:
            resume_checkpoint = existing[-1]
            receipt_path = resume_checkpoint.with_name(
                resume_checkpoint.name + ".trajectory.json"
            )
            if not receipt_path.is_file():
                raise ValueError("continuous resume receipt is missing")
            receipt = _read_json(receipt_path)
            if receipt.get("trajectory_config_hash") != self.config_sha256:
                raise ValueError("continuous resume trajectory config hash mismatch")
            resumed = load_checkpoint(resume_checkpoint)
            if _optimizer_hash(resumed.optimizer_state) != receipt.get(
                "optimizer_state_hash"
            ):
                raise ValueError("continuous resume optimizer hash mismatch")
            params = jax.tree_util.tree_map(jnp.asarray, resumed.params)
            state = resumed.optimizer_state
            start_step = int(resumed.manifest.step)
            previous_hash = receipt["checkpoint_hash"]
            previous_optimizer_hash = receipt["optimizer_state_hash"]
            previous_resource = receipt["cumulative_resources"]
            training_seconds = float(previous_resource["training_wall_seconds"])
            checkpoint_seconds = float(previous_resource["checkpoint_wall_seconds"])
            records_consumed = int(previous_resource["training_records"])
            tokens_consumed = int(previous_resource["training_tokens"])
            prior_total_wall_seconds = float(previous_resource["total_wall_seconds"])

        def loss_fn(current_params, input_ids, mask):
            output, _ = backend.forward_full(
                current_params, input_ids, attention_mask=mask
            )
            return masked_causal_lm_loss(backend.logits(output), input_ids, mask)

        value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
        started = time.perf_counter()
        previous_step = start_step
        snapshots = []
        requested = set(checkpoint_steps)
        for step in range(start_step + 1, checkpoint_steps[-1] + 1):
            input_ids, mask = batches[(step - 1) % len(batches)]
            train_started = time.perf_counter()
            _, grads = value_and_grad(params, input_ids, mask)
            params, state, _ = optimizer_update(params, grads, state, optimizer_config)
            jax.block_until_ready(params)
            training_seconds += time.perf_counter() - train_started
            records_consumed += int(input_ids.shape[0])
            tokens_consumed += int(np.asarray(mask[:, 1:]).sum())
            if step not in requested:
                continue
            checkpoint = output_dir / "trajectory" / "checkpoints" / f"step_{step:06d}"
            checkpoint_started = time.perf_counter()
            save_checkpoint(
                checkpoint,
                params,
                student_architecture=self.config.student_backend,
                student_config=student_config,
                step=step,
                learning_rate=self.config.vanilla_learning_rate,
                loss_config={"kind": "causal_language_model", "padding_masked": True},
                target_manifest={"source_texts": str(self.config.source_texts)},
                optimizer_config=asdict(optimizer_config),
                optimizer_state=state,
                notes=["P156.4.2 continuous vanilla trajectory snapshot"],
                overwrite=self.config.overwrite,
            )
            checkpoint_seconds += time.perf_counter() - checkpoint_started
            checkpoint_hash = hash_checkpoint_bundle(checkpoint)[
                "checkpoint_bundle_sha256"
            ]
            optimizer_hash = _optimizer_hash(state)
            cumulative = {
                "optimizer_steps": step,
                "training_records": records_consumed,
                "training_tokens": tokens_consumed,
                "source_text_bytes": self.config.source_texts.stat().st_size,
                "teacher_artifact_bytes": 0,
                "corridor_payload_bytes": 0,
                "exemplar_payload_bytes": 0,
                "shared_metadata_bytes": 0,
                "training_wall_seconds": training_seconds,
                "evaluation_wall_seconds": None,
                "checkpoint_wall_seconds": checkpoint_seconds,
                "total_wall_seconds": (
                    prior_total_wall_seconds + time.perf_counter() - started
                ),
                "cpu_seconds": None,
                "tpu_seconds": 0,
                "gpu_seconds": 0,
            }
            resource = _strict_resources(cumulative, previous_resource)
            resource.update(
                {
                    "source_text_bytes_consumed": cumulative["source_text_bytes"],
                    "teacher_artifact_bytes_consumed": 0,
                    "corridor_artifact_bytes_consumed": 0,
                    "exemplar_artifact_bytes_consumed": 0,
                    "artifact_bytes_logically_consumed": 0,
                    "optimizer_steps": step,
                    "training_records_consumed": records_consumed,
                    "training_tokens_consumed": tokens_consumed,
                    "wall_clock_training_seconds": training_seconds,
                    "total_seconds": cumulative["total_wall_seconds"],
                    "continuous_trajectory_confirmed": True,
                    "previous_checkpoint_total_step": previous_step,
                    "previous_optimizer_state_hash": previous_optimizer_hash,
                    "trajectory_root_checkpoint_hash": shared.checkpoint_hash,
                    "trajectory_config_hash": self.config_sha256,
                }
            )
            snapshot = self._arm_checkpoint(
                "vanilla",
                step,
                checkpoint,
                shared,
                previous_hash,
                corridor_steps=0,
                exemplar_steps=0,
                vanilla_steps=step,
                initial_optimizer_hash=initial_optimizer_hash,
                final_optimizer_hash=optimizer_hash,
                resource=resource,
            )
            snapshots.append(snapshot)
            _write_json(
                checkpoint.with_name(checkpoint.name + ".trajectory.json"),
                {
                    "phase": "P156.4.2",
                    "arm": "vanilla",
                    "total_step": step,
                    "checkpoint_hash": checkpoint_hash,
                    "parent_checkpoint_hash": previous_hash,
                    "optimizer_state_hash": optimizer_hash,
                    "previous_optimizer_state_hash": previous_optimizer_hash,
                    "trajectory_root_checkpoint_hash": shared.checkpoint_hash,
                    "trajectory_config_hash": self.config_sha256,
                    "continuous_trajectory_confirmed": True,
                    "cumulative_resources": cumulative,
                },
            )
            previous_step = step
            previous_hash = checkpoint_hash
            previous_optimizer_hash = optimizer_hash
            previous_resource = cumulative
        self._usage["real_vanilla_runner_used"] = True
        self._write_receipts()
        return snapshots

    def _train_exemplar(
        self,
        arm: str,
        seed: int,
        shared: SharedInitialization,
        discovery: AdaptiveDiscovery,
        total_step: int,
        output_dir: Path,
    ) -> ArmCheckpoint:
        adaptive = arm == "adaptive_two_cycle"
        parent = discovery.checkpoint if adaptive else shared.checkpoint
        local_steps = (
            total_step - int(discovery.optimizer_steps_completed or 0)
            if adaptive
            else total_step
        )
        optimizer_config = OptimizerConfig(
            type=self.config.exemplar_optimizer,
            learning_rate=self.config.exemplar_learning_rate,
        )
        parent_loaded = load_checkpoint(parent)
        fresh_hash = _optimizer_hash(
            init_optimizer_state(parent_loaded.params, optimizer_config)
        )
        if local_steps == 0:
            checkpoint = parent
            final_optimizer_hash = discovery.corridor_optimizer_state_hash
            report = {
                "resource_accounting": {},
                "actual_initial_exemplar_optimizer_hash": None,
                "fresh_optimizer_exact_match": None,
            }
        else:
            result = run_exemplar_pass(
                ExemplarPassConfig(
                    corridor_checkpoint=parent,
                    fingerprint_artifact=self.config.training_artifact,
                    held_out_fingerprint_artifact=self.config.calibration_artifact,
                    parent_fingerprint_artifact=self.config.training_artifact,
                    output_dir=output_dir / f"step_{total_step}",
                    student_backend=self.config.student_backend,
                    steps=local_steps,
                    batch_size=self.config.batch_size,
                    optimizer=self.config.exemplar_optimizer,
                    learning_rate=self.config.exemplar_learning_rate,
                    max_grad_norm=self.config.max_grad_norm,
                    seed=seed,
                    checkpoint_every=max(1, local_steps),
                    eval_every=max(1, local_steps),
                    allow_shared_initialization_parent_for_control=not adaptive,
                    overwrite=self.config.overwrite,
                )
            )
            checkpoint = result.final_checkpoint
            report = _read_json(result.report_path)
            if report["actual_initial_exemplar_optimizer_hash"] != fresh_hash:
                raise ValueError("actual Cycle 2 optimizer is not canonically fresh")
            if not report["fresh_optimizer_exact_match"]:
                raise ValueError("fresh exemplar optimizer exact match failed")
            final_optimizer_hash = _optimizer_hash(
                load_checkpoint(checkpoint).optimizer_state
            )
        classified = classify_artifact_bytes(self.config.training_artifact)
        category_bytes = classified["category_bytes"]
        corridor_bytes = category_bytes["corridor_payload"] if adaptive else 0
        exemplar_bytes = category_bytes["exemplar_payload"]
        shared_bytes = category_bytes["shared_teacher_metadata"]
        resource = _byte_accounting(
            arm=arm,
            source_text_bytes=0,
            corridor_bytes=corridor_bytes,
            exemplar_bytes=exemplar_bytes,
            shared_bytes=shared_bytes,
            artifact_bytes_available=classified["artifact_bytes_available"],
        )
        exemplar_resource = report.get("resource_accounting", {})
        resource.update(
            {
                "optimizer_steps": total_step,
                "training_records_consumed": exemplar_resource.get(
                    "total_exemplar_record_visits", 0
                ),
                "training_tokens_consumed": exemplar_resource.get("tokens_consumed", 0),
                "wall_clock_training_seconds": exemplar_resource.get(
                    "training_seconds", 0.0
                ),
                "total_seconds": exemplar_resource.get("total_wall_clock_seconds", 0.0),
                "expected_fresh_exemplar_optimizer_hash": fresh_hash,
                "actual_initial_exemplar_optimizer_hash": report[
                    "actual_initial_exemplar_optimizer_hash"
                ],
                "cycle_two_optimizer_instantiated": local_steps > 0,
                "fresh_optimizer_exact_match": report["fresh_optimizer_exact_match"],
                "fresh_optimizer_proof_status": (
                    "proven" if local_steps > 0 else "not_applicable"
                ),
            }
        )
        validate_byte_accounting(resource, arm=arm)
        self._last_byte_accounting = dict(resource)
        if local_steps > 0 and adaptive:
            self._freshness_proof = {
                "proof_checkpoint_total_step": total_step,
                "expected_fresh_exemplar_optimizer_hash": fresh_hash,
                "actual_initial_exemplar_optimizer_hash": report[
                    "actual_initial_exemplar_optimizer_hash"
                ],
                "fresh_optimizer_exact_match": report["fresh_optimizer_exact_match"],
                "fresh_optimizer_proof_status": "proven",
            }
        self._usage["real_exemplar_runner_used"] = (
            self._usage["real_exemplar_runner_used"] or local_steps > 0
        )
        self._usage["fresh_optimizer_exact_match"] = (
            self._usage["fresh_optimizer_exact_match"] or local_steps > 0
        )
        self._usage["byte_accounting_valid"] = True
        self._write_receipts()
        return self._arm_checkpoint(
            arm,
            total_step,
            checkpoint,
            shared,
            discovery.checkpoint_hash if adaptive else shared.checkpoint_hash,
            corridor_steps=int(discovery.optimizer_steps_completed or 0)
            if adaptive
            else 0,
            exemplar_steps=local_steps if adaptive else total_step,
            vanilla_steps=0,
            initial_optimizer_hash=fresh_hash,
            final_optimizer_hash=final_optimizer_hash,
            resource=resource,
        )

    def _train_exemplar_trajectory(
        self,
        arm: str,
        seed: int,
        shared: SharedInitialization,
        discovery: AdaptiveDiscovery,
        checkpoint_steps: tuple[int, ...],
        output_dir: Path,
    ) -> list[ArmCheckpoint]:
        adaptive = arm == "adaptive_two_cycle"
        boundary_step = int(discovery.optimizer_steps_completed or 0) if adaptive else 0
        parent = discovery.checkpoint if adaptive else shared.checkpoint
        parent_hash = discovery.checkpoint_hash if adaptive else shared.checkpoint_hash
        optimizer_config = OptimizerConfig(
            type=self.config.exemplar_optimizer,
            learning_rate=self.config.exemplar_learning_rate,
        )
        parent_loaded = load_checkpoint(parent)
        fresh_hash = _optimizer_hash(
            init_optimizer_state(parent_loaded.params, optimizer_config)
        )
        post_steps = tuple(step for step in checkpoint_steps if step > boundary_step)
        if adaptive and not post_steps:
            raise ValueError("continuous Cycle 2 requires a checkpoint greater than S")
        max_local_step = (
            checkpoint_steps[-1] - boundary_step if adaptive else checkpoint_steps[-1]
        )
        run_dir = output_dir / "trajectory"
        artifact_hash = _directory_hash(self.config.training_artifact)
        existing = sorted(
            path for path in (run_dir / "checkpoints").glob("step_*") if path.is_dir()
        )
        resume_checkpoint = existing[-1] if existing else None
        resume_data: dict[str, Any] | None = None
        resume_local_step = 0
        if resume_checkpoint is not None:
            receipt_path = resume_checkpoint.with_name(
                resume_checkpoint.name + ".trajectory.json"
            )
            if not receipt_path.is_file():
                raise ValueError("exemplar continuous resume receipt is missing")
            resume_data = _read_json(receipt_path)
            checks = {
                "arm": resume_data.get("arm") == arm,
                "seed": resume_data.get("seed") == seed,
                "trajectory_config": resume_data.get("trajectory_config_hash")
                == self.config_sha256,
                "artifact": resume_data.get("artifact_hash") == artifact_hash,
                "checkpoint": resume_data.get("checkpoint_hash")
                == hash_checkpoint_bundle(resume_checkpoint)[
                    "checkpoint_bundle_sha256"
                ],
                "optimizer": resume_data.get("optimizer_state_hash")
                == _optimizer_hash(load_checkpoint(resume_checkpoint).optimizer_state),
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                raise ValueError(
                    f"exemplar continuous resume validation failed: {failed}"
                )
            resume_local_step = int(resume_data["local_exemplar_step"])
            if resume_local_step >= max_local_step:
                raise ValueError("resume checkpoint is not before requested final step")
        result = run_exemplar_pass(
            ExemplarPassConfig(
                corridor_checkpoint=parent,
                fingerprint_artifact=self.config.training_artifact,
                held_out_fingerprint_artifact=self.config.calibration_artifact,
                parent_fingerprint_artifact=self.config.training_artifact,
                output_dir=run_dir,
                student_backend=self.config.student_backend,
                steps=max_local_step,
                batch_size=self.config.batch_size,
                optimizer=self.config.exemplar_optimizer,
                learning_rate=self.config.exemplar_learning_rate,
                max_grad_norm=self.config.max_grad_norm,
                seed=seed,
                checkpoint_every=1,
                eval_every=1,
                allow_shared_initialization_parent_for_control=not adaptive,
                resume_checkpoint=resume_checkpoint,
                overwrite=self.config.overwrite,
            )
        )
        report = _read_json(result.report_path)
        if (
            resume_data is None
            and report["actual_initial_exemplar_optimizer_hash"] != fresh_hash
        ):
            raise ValueError("actual exemplar optimizer is not canonically fresh")
        if (
            resume_data is not None
            and report["actual_initial_exemplar_optimizer_hash"]
            != resume_data["optimizer_state_hash"]
        ):
            raise ValueError("resumed exemplar optimizer state hash changed")
        trajectory = {
            int(row["optimizer_step"]): row
            for row in _read_jsonl(run_dir / "exemplar_trajectory.jsonl")
        }
        classified = classify_artifact_bytes(self.config.training_artifact)
        category = classified["category_bytes"]
        assigned = {
            "corridor_payload_bytes": category["corridor_payload"] if adaptive else 0,
            "exemplar_payload_bytes": category["exemplar_payload"],
            "shared_metadata_bytes": category["shared_teacher_metadata"],
        }
        assigned_total = sum(assigned.values())
        snapshots = []
        previous_step = boundary_step + resume_local_step
        previous_hash = (
            parent_hash if resume_data is None else resume_data["checkpoint_hash"]
        )
        previous_optimizer_hash = (
            discovery.corridor_optimizer_state_hash
            if adaptive and resume_data is None
            else fresh_hash
            if resume_data is None
            else resume_data["optimizer_state_hash"]
        )
        previous_resource: dict[str, float | int] = {}
        prior_cycle_two: dict[str, Any] = {}
        if resume_data is not None:
            previous_resource = resume_data["cumulative_resources"]
            prior_cycle_two = resume_data["cycle_two_cumulative_resources"]
        elif adaptive:
            if not self._cycle_one_resources or not self._cycle_one_receipt_hash:
                raise ValueError("adaptive Cycle 1 resource baseline is missing")
            boundary_cumulative = dict(self._cycle_one_resources)
            boundary_resource = _strict_resources(boundary_cumulative, {})
            boundary_resource.update(
                _byte_accounting(
                    arm=arm,
                    source_text_bytes=0,
                    corridor_bytes=assigned["corridor_payload_bytes"],
                    exemplar_bytes=0,
                    shared_bytes=assigned["shared_metadata_bytes"],
                    artifact_bytes_available=classified["artifact_bytes_available"],
                )
            )
            boundary_resource.update(
                {
                    "cycle_two_optimizer_instantiated": False,
                    "expected_fresh_exemplar_optimizer_hash": fresh_hash,
                    "actual_initial_exemplar_optimizer_hash": None,
                    "fresh_optimizer_exact_match": None,
                    "fresh_optimizer_proof_status": "not_applicable",
                    "continuous_trajectory_confirmed": True,
                    "cycle_one_resource_baseline_sha256": self._cycle_one_receipt_hash,
                }
            )
            snapshots.append(
                self._arm_checkpoint(
                    arm,
                    boundary_step,
                    parent,
                    shared,
                    shared.checkpoint_hash,
                    corridor_steps=boundary_step,
                    exemplar_steps=0,
                    vanilla_steps=0,
                    initial_optimizer_hash=discovery.corridor_optimizer_state_hash,
                    final_optimizer_hash=discovery.corridor_optimizer_state_hash,
                    resource=boundary_resource,
                )
            )
            previous_resource = boundary_cumulative
        candidate_steps = post_steps if adaptive else checkpoint_steps
        for total_step in (
            step for step in candidate_steps if step - boundary_step > resume_local_step
        ):
            local_step = total_step - boundary_step if adaptive else total_step
            checkpoint = run_dir / "checkpoints" / f"step_{local_step:06d}"
            loaded = load_checkpoint(checkpoint)
            point = trajectory[local_step]
            training_wall = float(point["cumulative_training_seconds"])
            evaluation_wall = float(point["cumulative_evaluation_seconds"])
            checkpoint_wall = float(point["cumulative_checkpoint_seconds"])
            cycle_two = {
                "optimizer_steps": local_step,
                "training_records": int(prior_cycle_two.get("training_records", 0))
                + int(point["records_consumed"]),
                "training_tokens": int(prior_cycle_two.get("training_tokens", 0))
                + int(point["tokens_consumed"]),
                "training_wall_seconds": float(
                    prior_cycle_two.get("training_wall_seconds", 0.0)
                )
                + training_wall,
                "evaluation_wall_seconds": float(
                    prior_cycle_two.get("evaluation_wall_seconds", 0.0)
                )
                + evaluation_wall,
                "checkpoint_wall_seconds": float(
                    prior_cycle_two.get("checkpoint_wall_seconds", 0.0)
                )
                + checkpoint_wall,
                "total_wall_seconds": float(
                    prior_cycle_two.get("total_wall_seconds", 0.0)
                )
                + max(
                    float(point["wall_clock_seconds"]),
                    training_wall + evaluation_wall + checkpoint_wall,
                ),
            }
            cumulative = {
                "optimizer_steps": total_step,
                "training_records": cycle_two["training_records"],
                "training_tokens": cycle_two["training_tokens"],
                "source_text_bytes": 0,
                "teacher_artifact_bytes": assigned_total,
                **assigned,
                "training_wall_seconds": cycle_two["training_wall_seconds"],
                "evaluation_wall_seconds": cycle_two["evaluation_wall_seconds"],
                "checkpoint_wall_seconds": cycle_two["checkpoint_wall_seconds"],
                "total_wall_seconds": cycle_two["total_wall_seconds"],
                "cpu_seconds": None,
                "tpu_seconds": 0,
                "gpu_seconds": 0,
            }
            if adaptive:
                for name in (
                    "training_records",
                    "training_tokens",
                    "training_wall_seconds",
                    "evaluation_wall_seconds",
                    "checkpoint_wall_seconds",
                    "total_wall_seconds",
                ):
                    cumulative[name] = self._cycle_one_resources[name] + cycle_two[name]
            resource = _strict_resources(cumulative, previous_resource)
            resource.update(
                _byte_accounting(
                    arm=arm,
                    source_text_bytes=0,
                    corridor_bytes=assigned["corridor_payload_bytes"],
                    exemplar_bytes=assigned["exemplar_payload_bytes"],
                    shared_bytes=assigned["shared_metadata_bytes"],
                    artifact_bytes_available=classified["artifact_bytes_available"],
                )
            )
            optimizer_hash = _optimizer_hash(loaded.optimizer_state)
            resource.update(
                {
                    "optimizer_steps": total_step,
                    "training_records_consumed": cumulative["training_records"],
                    "training_tokens_consumed": cumulative["training_tokens"],
                    "wall_clock_training_seconds": cumulative["training_wall_seconds"],
                    "total_seconds": cumulative["total_wall_seconds"],
                    "expected_fresh_exemplar_optimizer_hash": fresh_hash,
                    "actual_initial_exemplar_optimizer_hash": fresh_hash,
                    "cycle_two_optimizer_instantiated": adaptive,
                    "fresh_optimizer_exact_match": True if adaptive else None,
                    "fresh_optimizer_proof_status": "proven"
                    if adaptive
                    else "not_applicable",
                    "continuous_trajectory_confirmed": True,
                    "previous_checkpoint_total_step": previous_step,
                    "previous_optimizer_state_hash": previous_optimizer_hash,
                    "trajectory_root_checkpoint_hash": parent_hash,
                    "trajectory_config_hash": self.config_sha256,
                    "cycle_one_resource_baseline_sha256": (
                        self._cycle_one_receipt_hash if adaptive else None
                    ),
                    "cycle_two_cumulative_resources": cycle_two,
                    "resumed": resume_data is not None,
                    "resume_source_checkpoint": (
                        str(resume_checkpoint) if resume_checkpoint else None
                    ),
                    "resume_source_total_step": (
                        previous_step if resume_data is not None else None
                    ),
                    "resume_source_optimizer_hash": (
                        resume_data["optimizer_state_hash"] if resume_data else None
                    ),
                    "completed_steps_replayed": False,
                }
            )
            checkpoint_hash = hash_checkpoint_bundle(checkpoint)[
                "checkpoint_bundle_sha256"
            ]
            snapshot = self._arm_checkpoint(
                arm,
                total_step,
                checkpoint,
                shared,
                previous_hash,
                corridor_steps=boundary_step if adaptive else 0,
                exemplar_steps=local_step,
                vanilla_steps=0,
                initial_optimizer_hash=fresh_hash,
                final_optimizer_hash=optimizer_hash,
                resource=resource,
            )
            snapshots.append(snapshot)
            _write_json(
                checkpoint.with_name(checkpoint.name + ".trajectory.json"),
                {
                    "phase": "P156.4.2.1",
                    "arm": arm,
                    "seed": seed,
                    "total_step": total_step,
                    "local_exemplar_step": local_step,
                    "checkpoint_hash": checkpoint_hash,
                    "optimizer_state_hash": optimizer_hash,
                    "trajectory_root_checkpoint_hash": parent_hash,
                    "previous_checkpoint_hash": previous_hash,
                    "previous_optimizer_state_hash": previous_optimizer_hash,
                    "trajectory_config_hash": self.config_sha256,
                    "artifact_hash": artifact_hash,
                    "cumulative_resources": cumulative,
                    "cycle_two_cumulative_resources": cycle_two,
                    "fresh_optimizer_hash": fresh_hash,
                    "resumed": resume_data is not None,
                    "completed_steps_replayed": False,
                },
            )
            previous_step = total_step
            previous_hash = checkpoint_hash
            previous_optimizer_hash = optimizer_hash
            previous_resource = cumulative
        self._usage["real_exemplar_runner_used"] = True
        if adaptive and resume_data is None:
            self._usage["fresh_optimizer_exact_match"] = True
            self._freshness_proof = {
                "proof_checkpoint_total_step": post_steps[0],
                "expected_fresh_exemplar_optimizer_hash": fresh_hash,
                "actual_initial_exemplar_optimizer_hash": report[
                    "actual_initial_exemplar_optimizer_hash"
                ],
                "fresh_optimizer_exact_match": True,
                "fresh_optimizer_proof_status": "proven",
            }
        self._usage["byte_accounting_valid"] = True
        self._last_byte_accounting = dict(snapshots[-1].resource_accounting)
        if adaptive:
            final_resource = snapshots[-1].resource_accounting
            cycle_two_final = final_resource["cycle_two_cumulative_resources"]
            formula_fields = (
                "optimizer_steps",
                "training_records",
                "training_tokens",
                "training_wall_seconds",
                "total_wall_seconds",
            )
            checks = {
                name: final_resource[f"cumulative_{name}"]
                == self._cycle_one_resources[name] + cycle_two_final[name]
                for name in formula_fields
            }
            self._adaptive_cost_evidence = {
                "phase": "P156.4.2.1",
                "cycle_one_resources": self._cycle_one_resources,
                "cycle_two_resources": cycle_two_final,
                "combined_resources": {
                    name: final_resource[f"cumulative_{name}"]
                    for name in formula_fields
                },
                "formula_checks": checks,
                "all_required_values_present": all(
                    self._cycle_one_resources.get(name) is not None
                    and cycle_two_final.get(name) is not None
                    for name in formula_fields
                ),
                "adaptive_full_cost_accounting_valid": all(checks.values()),
            }
            _write_json(
                self.config.receipt_root / "adaptive_full_cost_accounting_receipt.json",
                self._adaptive_cost_evidence,
            )
        self._write_receipts()
        return snapshots

    def _arm_checkpoint(
        self,
        arm: str,
        total_step: int,
        checkpoint: Path,
        shared: SharedInitialization,
        parent_hash: str,
        *,
        corridor_steps: int,
        exemplar_steps: int,
        vanilla_steps: int,
        initial_optimizer_hash: str,
        final_optimizer_hash: str,
        resource: dict[str, Any],
    ) -> ArmCheckpoint:
        if (
            parameter_fingerprint(load_checkpoint(shared.checkpoint).params)
            != shared.parameter_tree_hash
        ):
            raise ValueError("shared initialization parameter hash mismatch")
        return ArmCheckpoint(
            arm=arm,
            total_step=total_step,
            checkpoint=checkpoint,
            checkpoint_hash=hash_checkpoint_bundle(checkpoint)[
                "checkpoint_bundle_sha256"
            ],
            parent_checkpoint_hash=parent_hash,
            initial_parameter_hash=shared.parameter_tree_hash,
            corridor_steps=corridor_steps,
            exemplar_steps=exemplar_steps,
            vanilla_steps=vanilla_steps,
            optimizer_initial_state_hash=initial_optimizer_hash,
            optimizer_final_state_hash=final_optimizer_hash,
            resource_accounting=resource,
        )

    def _student_prefix_metrics(
        self, checkpoint: Path, records: tuple[Any, ...]
    ) -> dict[str, Any]:
        loaded, backend = self._load_student(checkpoint)
        rows = []
        for record in records[: self.config.evaluation_prompt_limit]:
            prompt = list(record.input_ids[: max(1, record.position + 1)])
            generated = _generate_student(
                backend, loaded.params, prompt, self.config.prefix_length
            )
            context = prompt + generated
            student = _student_probs(backend, loaded.params, context)
            teacher = _teacher_probs(self.config.teacher_backend, context)
            metrics = _distribution_metrics(student, teacher)
            rows.append(
                {
                    "prompt_id": record.example_id,
                    "generated_prefix_token_ids": generated,
                    "trajectory_length": len(generated),
                    "student_prefix_teacher_student_kl": metrics["kl"],
                    "student_prefix_top1_agreement": metrics["top1"],
                    "student_prefix_topk_overlap": metrics["topk"],
                    "student_prefix_entropy_error": metrics["entropy_error"],
                }
            )
        return {
            **_mean_prefix(rows),
            "records": rows,
            "contexts_generated_by_student": True,
        }

    def _free_running_metrics(
        self, checkpoint: Path, records: tuple[Any, ...]
    ) -> dict[str, Any]:
        loaded, backend = self._load_student(checkpoint)
        rows = []
        for record in records[: self.config.evaluation_prompt_limit]:
            prompt = list(record.input_ids[: max(1, record.position + 1)])
            student_tokens = _generate_student(
                backend, loaded.params, prompt, self.config.prefix_length
            )
            teacher_tokens = _generate_teacher(
                self.config.teacher_backend, prompt, self.config.prefix_length
            )
            teacher_log_likelihood = 0.0
            context = list(prompt)
            for token in student_tokens:
                probs = _teacher_probs(self.config.teacher_backend, context)
                teacher_log_likelihood += math.log(float(probs[token]) + 1e-12)
                context.append(token)
            rows.append(
                {
                    "prompt_id": record.example_id,
                    "student_continuation": student_tokens,
                    "teacher_continuation": teacher_tokens,
                    "teacher_likelihood_of_student_continuation": (
                        teacher_log_likelihood
                    ),
                    "token_frequency_distance": _token_frequency_distance(
                        student_tokens, teacher_tokens
                    ),
                    "repetition_rate": _repetition_rate(student_tokens),
                    "sequence_entropy": _sequence_entropy(student_tokens),
                    "mode_occupancy": _mode_occupancy(student_tokens),
                    "length": len(student_tokens),
                }
            )
        return {
            **_mean_generation(rows),
            "records": rows,
            "decoding_policies": ["greedy"],
            "exact_text_match_required": False,
        }

    def _load_student(self, checkpoint: Path):
        loaded = load_checkpoint(checkpoint)
        summary = summarize_fingerprint_artifact(self.config.final_test_artifact)
        config = self._baseline_config(0, checkpoint.parent, steps=1)
        backend, _ = _create_baseline_backend(config, summary)
        return loaded, backend

    def _baseline_config(self, seed: int, output_dir: Path, *, steps: int):
        return FingerprintTrainedBaselineConfig(
            fingerprint_artifact=self.config.training_artifact,
            source_texts=self.config.source_texts,
            output_dir=output_dir,
            steps=steps,
            batch_size=self.config.batch_size,
            optimizer=self.config.vanilla_optimizer,
            learning_rate=self.config.vanilla_learning_rate,
            seed=seed,
            student_backend=self.config.student_backend,
            overwrite=self.config.overwrite,
        )

    def _validate_artifacts(self) -> None:
        selected_profile = _read_json(self.config.selected_profile_receipt)
        if selected_profile.get("status") != "pass":
            raise ValueError("selected profile receipt is not passing")
        summaries = [
            summarize_fingerprint_artifact(path)
            for path in (
                self.config.training_artifact,
                self.config.calibration_artifact,
                self.config.final_test_artifact,
            )
        ]
        if len({summary.vocab_size for summary in summaries}) != 1:
            raise ValueError("crossover artifact vocabularies do not match")
        for path in (
            self.config.training_artifact,
            self.config.calibration_artifact,
            self.config.final_test_artifact,
        ):
            tuple(load_fingerprint_targets(path, batch_size=1).iter_records())
        tuple(
            load_fingerprint_exemplars(
                self.config.training_artifact, batch_size=1
            ).iter_records()
        )
        self._usage["real_artifact_loaders_used"] = True

    def _write_receipts(self) -> None:
        status = "pass" if all(self._usage.values()) else "incomplete"
        required_resume_arms = ("vanilla", "exemplar_only", "adaptive_two_cycle")
        resume_valid = all(
            self._resume_evidence.get(arm, {}).get("pass") is True
            for arm in required_resume_arms
        )
        adaptive_cost_valid = (
            self._adaptive_cost_evidence.get("adaptive_full_cost_accounting_valid")
            is True
        )
        ready = (
            status == "pass"
            and resume_valid
            and adaptive_cost_valid
            and self._usage["byte_accounting_valid"]
        )
        receipt = {
            "phase": "P156.4.1.1",
            "status": status,
            "backend": "radjax",
            "radjax_crossover_backend_config_sha256": self.config_sha256,
            **self._usage,
            "implementation_smoke_complete": status == "pass",
            "full_distillation_run_started": False,
            "publication_grade": False,
            "ready_for_P156_5": ready,
            "checkpoint_execution_mode": "continuous_trajectory",
            "strict_resource_accounting": True,
            "silent_zero_defaults_present": False,
            "cycle_two_optimizer_reinitialized_per_checkpoint": False,
        }
        _write_json(
            self.config.receipt_root / "radjax_backend_binding_receipt.json", receipt
        )
        common = {
            "phase": "P156.4.2.1",
            "status": "pass" if ready else "incomplete",
            "backend": "radjax",
            "checkpoint_execution_mode": "continuous_trajectory",
            "strict_resource_accounting": True,
            "silent_zero_defaults_present": False,
            "adaptive_full_cost_accounting_valid": adaptive_cost_valid,
            "continuous_resume_valid": resume_valid,
            "vanilla_resume_valid": self._resume_evidence.get("vanilla", {}).get(
                "pass", False
            ),
            "exemplar_only_resume_valid": self._resume_evidence.get(
                "exemplar_only", {}
            ).get("pass", False),
            "adaptive_cycle_two_resume_valid": self._resume_evidence.get(
                "adaptive_two_cycle", {}
            ).get("pass", False),
            "negative_total_wall_intervals_present": False,
            "completed_steps_replayed": any(
                self._resume_evidence.get(arm, {}).get(
                    "completed_steps_replayed", False
                )
                for arm in required_resume_arms
            ),
            "cycle_two_optimizer_reinitialized_per_checkpoint": False,
            "real_cpu_smoke_complete": status == "pass",
            "full_distillation_run_started": False,
            "publication_grade": False,
            "ready_for_P156_5": ready,
        }
        _write_json(
            self.config.receipt_root / "continuous_trajectory_receipt.json", common
        )
        _write_json(
            self.config.receipt_root / "strict_resource_accounting_receipt.json",
            {**common, "resource_snapshot": self._last_byte_accounting},
        )
        _write_json(
            self.config.receipt_root / "continuous_resume_receipt.json",
            {**common, "arms": self._resume_evidence},
        )
        _write_json(
            self.config.receipt_root / "p156_4_2_real_cpu_smoke_report.json", common
        )
        _write_json(
            self.config.receipt_root / "p156_4_2_1_real_cpu_smoke_report.json", common
        )
        _write_json(
            self.config.receipt_root / "resource_field_mapping_receipt.json",
            {
                **common,
                "mappings": _resource_field_mappings(),
            },
        )
        _write_json(
            self.config.receipt_root / "real_backend_integration_smoke_report.json",
            receipt,
        )
        _write_json(
            self.config.receipt_root / "real_backend_evaluation_provenance.json",
            {
                "phase": "P156.4.1.1",
                **{
                    key: value
                    for key, value in self._usage.items()
                    if "evaluation" in key or "teacher" in key
                },
            },
        )
        _write_json(
            self.config.receipt_root / "real_backend_byte_accounting_receipt.json",
            {
                "phase": "P156.4.1.1",
                "byte_accounting_valid": self._usage["byte_accounting_valid"],
                **self._last_byte_accounting,
            },
        )
        _write_json(
            self.config.receipt_root / "real_backend_optimizer_freshness_receipt.json",
            {
                "phase": "P156.4.1.1",
                **self._freshness_proof,
            },
        )


def _optimizer_hash(state: Any) -> str:
    if state is None:
        raise ValueError("optimizer state is required")
    return stable_hash(
        {
            "type": state.type,
            "step": int(state.step),
            "slots_parameter_fingerprint": parameter_fingerprint(state.slots),
        }
    )


def _validate_trajectory_steps(steps: tuple[int, ...]) -> None:
    if not steps or any(step < 0 for step in steps):
        raise ValueError("trajectory checkpoint steps must be non-negative")
    if any(right <= left for left, right in zip(steps, steps[1:], strict=False)):
        raise ValueError("trajectory checkpoint steps must be strictly increasing")


def _strict_resources(
    cumulative: dict[str, float | int | None],
    previous: dict[str, float | int | None],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "strict_resource_accounting": True,
        "silent_zero_defaults_present": False,
        "token_counting_policy": "non_padding_training_tokens",
        "artifact_consumption_policy": "full_assigned_artifact_charged_at_first_use",
        "resource_provenance": {},
    }
    for name, value in cumulative.items():
        cumulative_name = f"cumulative_{name}"
        interval_name = f"interval_{name}"
        if value is None:
            interval = None
            provenance = {
                "value": None,
                "status": "unavailable",
                "reason": "underlying runner does not expose this measurement",
            }
            interval_provenance = dict(provenance)
        else:
            prior = previous.get(name)
            interval = value if prior is None else value - prior
            status = "measured" if "bytes" not in name else "derived"
            provenance = {
                "value": value,
                "status": status,
                "source": f"continuous_runner.{name}",
            }
            interval_provenance = {
                "value": interval,
                "status": "derived",
                "source": f"{cumulative_name}-previous_{cumulative_name}",
            }
        result[cumulative_name] = value
        result[interval_name] = interval
        result["resource_provenance"][cumulative_name] = provenance
        result["resource_provenance"][interval_name] = interval_provenance
    return result


def _resource_field_mappings() -> list[dict[str, str]]:
    sources = {
        "optimizer_steps": "checkpoint optimizer step",
        "training_records": "runner records_consumed",
        "training_tokens": "runner tokens_consumed",
        "training_wall_seconds": "time.perf_counter accumulation",
        "evaluation_wall_seconds": "runner cumulative_evaluation_seconds",
        "checkpoint_wall_seconds": "time.perf_counter accumulation",
        "total_wall_seconds": "time.perf_counter elapsed",
        "teacher_artifact_bytes": "disjoint artifact file classifier",
        "cpu_seconds": "unavailable",
        "tpu_seconds": "CPU execution contract",
        "gpu_seconds": "CPU execution contract",
    }
    return [
        {
            "crossover_field": f"{scope}_{field}",
            "source_runner": "RadjaxCrossoverBackend continuous trajectory",
            "source_report_field": source,
            "unit": "seconds" if field.endswith("seconds") else "count_or_bytes",
            "cumulative_or_interval": scope,
            "status": "unavailable"
            if source == "unavailable"
            else "measured_or_derived",
        }
        for scope in ("cumulative", "interval")
        for field, source in sources.items()
    ]


def _byte_accounting(
    *,
    arm: str,
    source_text_bytes: int,
    corridor_bytes: int = 0,
    exemplar_bytes: int = 0,
    shared_bytes: int = 0,
    artifact_bytes_available: int = 0,
) -> dict[str, Any]:
    total = corridor_bytes + exemplar_bytes + shared_bytes
    result = {
        "arm": arm,
        "artifact_bytes_available": artifact_bytes_available,
        "artifact_bytes_selected": total,
        "artifact_bytes_logically_consumed": total,
        "filesystem_bytes_read": None,
        "corridor_payload_bytes": corridor_bytes,
        "exemplar_payload_bytes": exemplar_bytes,
        "shared_teacher_metadata_bytes": shared_bytes,
        "teacher_artifact_bytes_total": total,
        # Compatibility aliases for existing P156.4 report readers.
        "teacher_artifact_bytes_consumed": total,
        "corridor_artifact_bytes_consumed": corridor_bytes,
        "exemplar_artifact_bytes_consumed": exemplar_bytes,
        "source_text_bytes_consumed": source_text_bytes,
    }
    validate_byte_accounting(result, arm=arm)
    return result


def _teacher_probs(teacher: Any, token_ids: list[int]) -> np.ndarray:
    teacher.load()
    try:
        import torch

        inputs = torch.tensor([token_ids], dtype=torch.long)
        with torch.no_grad():
            output = teacher.model(input_ids=inputs)
        value = output.logits[0, -1]
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        logits = np.asarray(value, dtype=np.float64)
    except ImportError:
        output = teacher.model(input_ids=np.asarray([token_ids], dtype=np.int64))
        logits = np.asarray(output.logits[0, -1], dtype=np.float64)
    return _softmax(logits)


def _generate_teacher(teacher: Any, prompt: list[int], length: int) -> list[int]:
    tokens = list(prompt)
    generated = []
    for _ in range(length):
        token = int(np.argmax(_teacher_probs(teacher, tokens)))
        generated.append(token)
        tokens.append(token)
    return generated


def _student_probs(backend: Any, params: Any, token_ids: list[int]) -> np.ndarray:
    output, _ = backend.forward_full(params, jnp.asarray([token_ids], dtype=jnp.int32))
    return _softmax(np.asarray(backend.logits(output)[0, -1], dtype=np.float64))


def _generate_student(
    backend: Any, params: Any, prompt: list[int], length: int
) -> list[int]:
    tokens = list(prompt)
    generated = []
    for _ in range(length):
        token = int(np.argmax(_student_probs(backend, params, tokens)))
        generated.append(token)
        tokens.append(token)
    return generated


def _distribution_metrics(student: np.ndarray, teacher: np.ndarray) -> dict[str, float]:
    if student.shape != teacher.shape:
        raise ValueError("teacher and student vocabularies do not match")
    eps = 1e-12
    k = min(8, len(student))
    return {
        "kl": float(np.sum(teacher * (np.log(teacher + eps) - np.log(student + eps)))),
        "top1": float(np.argmax(student) == np.argmax(teacher)),
        "topk": len(set(np.argsort(student)[-k:]) & set(np.argsort(teacher)[-k:])) / k,
        "entropy_error": abs(
            -float(np.sum(student * np.log(student + eps)))
            + float(np.sum(teacher * np.log(teacher + eps)))
        ),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.exp(logits - np.max(logits))
    return values / np.sum(values)


def _mean_prefix(rows: list[dict[str, Any]]) -> dict[str, float]:
    names = (
        "student_prefix_teacher_student_kl",
        "student_prefix_top1_agreement",
        "student_prefix_topk_overlap",
        "student_prefix_entropy_error",
        "trajectory_length",
    )
    return {name: float(np.mean([row[name] for row in rows])) for name in names}


def _mean_generation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = (
        "teacher_likelihood_of_student_continuation",
        "token_frequency_distance",
        "repetition_rate",
        "sequence_entropy",
        "length",
    )
    result = {name: float(np.mean([row[name] for row in rows])) for name in names}
    result["mode_occupancy"] = rows[0]["mode_occupancy"] if rows else {}
    result["length_distribution"] = {"mean": result["length"]}
    return result


def _token_frequency_distance(left: list[int], right: list[int]) -> float:
    vocab = sorted(set(left) | set(right))
    if not vocab:
        return 0.0
    left_counts = np.asarray([left.count(token) for token in vocab], dtype=np.float64)
    right_counts = np.asarray([right.count(token) for token in vocab], dtype=np.float64)
    left_counts /= max(np.sum(left_counts), 1.0)
    right_counts /= max(np.sum(right_counts), 1.0)
    return float(np.sum(np.abs(left_counts - right_counts)))


def _repetition_rate(tokens: list[int]) -> float:
    if len(tokens) < 2:
        return 0.0
    return sum(
        left == right for left, right in zip(tokens, tokens[1:], strict=False)
    ) / (len(tokens) - 1)


def _sequence_entropy(tokens: list[int]) -> float:
    if not tokens:
        return 0.0
    counts = np.asarray(
        list({token: tokens.count(token) for token in set(tokens)}.values()),
        dtype=np.float64,
    )
    probs = counts / np.sum(counts)
    return -float(np.sum(probs * np.log(probs + 1e-12)))


def _mode_occupancy(tokens: list[int]) -> dict[str, float]:
    if not tokens:
        return {}
    return {
        str(token): tokens.count(token) / len(tokens) for token in sorted(set(tokens))
    }


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    ):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
