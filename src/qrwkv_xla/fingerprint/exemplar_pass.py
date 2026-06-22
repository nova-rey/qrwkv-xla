from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qrwkv_xla.artifacts import (
    FingerprintExemplarBatch,
    FingerprintExemplarRecord,
    FingerprintTargetRecord,
    load_fingerprint_exemplars,
    load_fingerprint_targets,
    summarize_fingerprint_artifact,
)
from qrwkv_xla.artifacts._json import read_json_object, write_json
from qrwkv_xla.checkpointing import load_checkpoint, save_checkpoint
from qrwkv_xla.contracts import VocabContract
from qrwkv_xla.fingerprint.corridor_measurement import _evaluate_held_out
from qrwkv_xla.fingerprint.provenance import (
    file_sha256,
    hash_checkpoint_bundle,
    parameter_fingerprint,
    stable_hash,
)
from qrwkv_xla.optimizers import OptimizerConfig, init_optimizer_state, optimizer_update
from qrwkv_xla.students import create_student_backend
from qrwkv_xla.trainers.state import TrainState
from qrwkv_xla.training.fingerprint_exemplar_loss import (
    compute_fingerprint_exemplar_loss_at_positions,
)
from qrwkv_xla.training.fingerprint_stats import select_position_logits
from qrwkv_xla.training.gradients import clip_gradients_by_global_norm


@dataclass(frozen=True)
class ExemplarPassConfig:
    corridor_checkpoint: Path
    fingerprint_artifact: Path
    output_dir: Path
    student_backend: str = "current_qrwkv"
    student_architecture: str | None = None
    steps: int = 25
    batch_size: int = 1
    optimizer: str = "adamw"
    learning_rate: float = 5e-5
    max_grad_norm: float | None = 1.0
    seed: int = 0
    checkpoint_every: int = 25
    eval_every: int | None = None
    held_out_fingerprint_artifact: Path | None = None
    p153_report: Path | None = None
    selected_profile: Path | None = None
    exemplar_max_records: int | None = None
    exemplar_sampling_policy: str = "sequential"
    resume_checkpoint: Path | None = None
    corridor_entry_threshold: float = 0.95
    corridor_retention_tolerance: float = 0.0
    allow_shared_initialization_parent_for_control: bool = False
    overwrite: bool = False


@dataclass(frozen=True)
class ExemplarPassResult:
    status: str
    output_dir: Path
    report_path: Path
    trajectory_path: Path
    final_checkpoint: Path
    best_checkpoint: Path
    completed_steps: int


def validate_corridor_checkpoint_lineage(
    config: ExemplarPassConfig,
    *,
    artifact_vocab_size: int,
) -> tuple[Any, dict[str, Any]]:
    loaded = load_checkpoint(config.corridor_checkpoint)
    manifest = loaded.manifest
    hashes = hash_checkpoint_bundle(config.corridor_checkpoint)
    expected_manifest_hash = file_sha256(config.fingerprint_artifact / "manifest.json")
    p153_report_valid = True
    if config.p153_report is not None:
        p153 = read_json_object(config.p153_report)
        p153_report_valid = bool(
            p153.get("status") == "pass"
            and p153.get("training_cycle") == "corridor_only"
            and int(p153.get("completed_steps", -1)) == manifest.step
            and p153.get("checkpoint", {}).get("checkpoint_bundle_sha256")
            == hashes["checkpoint_bundle_sha256"]
            and p153.get("checkpoint", {}).get("final_parameter_fingerprint")
            == parameter_fingerprint(loaded.params)
            and p153.get("lineage", {}).get("artifact_manifest_sha256")
            == expected_manifest_hash
        )
    selected_profile_valid = True
    calibration_binding: dict[str, Any] = {
        "calibration_receipt_sha256": None,
        "selected_profile_name": None,
        "selected_profile_config_sha256": None,
        "parent_corridor_checkpoint_bundle_sha256": hashes["checkpoint_bundle_sha256"],
        "parent_corridor_parameter_fingerprint": parameter_fingerprint(loaded.params),
        "calibration_parent_binding_valid": config.selected_profile is None,
    }
    if config.selected_profile is not None:
        selected = read_json_object(config.selected_profile)
        selected_name = selected.get("selected_profile") or selected.get("profile_name")
        selected_profile_valid = bool(
            selected.get("status") == "pass"
            and selected.get("selection_allowed") is True
            and selected.get("winner_declared") is True
            and selected_name
            and selected.get("selected_profile_config_sha256")
            == manifest.target_manifest.get("selected_profile_config_sha256")
            and selected.get("selected_corridor_checkpoint_bundle_sha256")
            == hashes["checkpoint_bundle_sha256"]
            and selected.get("selected_corridor_parameter_fingerprint")
            == parameter_fingerprint(loaded.params)
            and manifest.target_manifest.get("selected_aggressiveness_profile")
            == selected_name
        )
        calibration_binding = {
            "calibration_receipt_sha256": file_sha256(config.selected_profile),
            "selected_profile_name": selected_name,
            "selected_profile_config_sha256": selected.get(
                "selected_profile_config_sha256"
            ),
            "parent_corridor_checkpoint_bundle_sha256": hashes[
                "checkpoint_bundle_sha256"
            ],
            "parent_corridor_parameter_fingerprint": parameter_fingerprint(
                loaded.params
            ),
            "calibration_parent_binding_valid": selected_profile_valid,
        }
    control_parent = config.allow_shared_initialization_parent_for_control
    checks = {
        "checkpoint_bundle_hash_valid": bool(hashes["checkpoint_bundle_sha256"]),
        "parameter_fingerprint_valid": bool(parameter_fingerprint(loaded.params)),
        "training_cycle_is_corridor": (
            manifest.loss_config.get("kind") == "shared_initialization"
            if control_parent
            else manifest.loss_config.get("cycle") == 1
        ),
        "distill_mode_is_fingerprint_corridor": manifest.loss_config.get("kind")
        == ("shared_initialization" if control_parent else "fingerprint_corridor"),
        "optimizer_steps_completed": (
            manifest.step == 0 if control_parent else manifest.step > 0
        ),
        "completed_corridor_checkpoint": (
            config.corridor_checkpoint.name == "initial"
            if control_parent
            else config.corridor_checkpoint.name == "final"
        ),
        "student_backend_match": manifest.student_architecture
        == config.student_backend,
        "student_architecture_match": config.student_architecture is None
        or manifest.student_config.get("architecture") == config.student_architecture,
        "vocab_size_match": manifest.student_config.get("vocab_size")
        == artifact_vocab_size,
        "artifact_manifest_hash_match": manifest.target_manifest.get(
            "artifact_dir" if control_parent else "artifact_manifest_sha256"
        )
        == (
            str(config.fingerprint_artifact)
            if control_parent
            else expected_manifest_hash
        ),
        "p153_parent_binding_valid": p153_report_valid,
        "calibration_parent_binding_valid": selected_profile_valid,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    receipt = {
        "phase": "P154.1.1",
        "status": "pass" if not blockers else "fail",
        "lineage_valid": not blockers,
        "blockers": blockers,
        "checks": checks,
        "training_cycle": "shared_initialization_control"
        if control_parent
        else "corridor",
        "distill_mode": "fingerprint_exemplar_control"
        if control_parent
        else "fingerprint_corridor",
        "optimizer_steps_completed": manifest.step,
        "parameter_fingerprint": parameter_fingerprint(loaded.params),
        **calibration_binding,
        **hashes,
    }
    if blockers:
        if not selected_profile_valid:
            raise ValueError("calibration_parent_lineage_mismatch")
        raise ValueError(
            "P154 corridor checkpoint lineage failed: " + ", ".join(blockers)
        )
    return loaded, receipt


def deterministic_record_order(
    records: tuple[FingerprintExemplarRecord, ...], policy: str, seed: int
) -> tuple[FingerprintExemplarRecord, ...]:
    if policy == "sequential":
        return records
    if policy == "uniform_without_replacement":
        rng = np.random.default_rng(seed)
        return tuple(records[index] for index in rng.permutation(len(records)))
    raise ValueError(f"unsupported exemplar sampling policy: {policy}")


def record_order_sha256(records: tuple[FingerprintExemplarRecord, ...]) -> str:
    return stable_hash(
        [
            {
                "example_id": record.example_id,
                "position": record.position,
                "mode_id": record.mode_id,
            }
            for record in records
        ]
    )


def run_exemplar_pass(config: ExemplarPassConfig) -> ExemplarPassResult:
    _validate_config(config)
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_fingerprint_artifact(config.fingerprint_artifact)
    dataset = load_fingerprint_exemplars(
        config.fingerprint_artifact,
        batch_size=config.batch_size,
        max_records=config.exemplar_max_records,
    )
    records = deterministic_record_order(
        tuple(dataset.iter_records()), config.exemplar_sampling_policy, config.seed
    )
    if not records:
        raise ValueError("exemplar reservoir contains zero records")
    _validate_exemplar_records(records, dataset.vocab_size)
    batches = _records_to_batches(records, config.batch_size, dataset.max_seq_len)
    sampling_contract = _sampling_contract(config, records)
    parent, lineage = validate_corridor_checkpoint_lineage(
        config, artifact_vocab_size=dataset.vocab_size
    )
    write_json(config.output_dir / "checkpoint_lineage_validation.json", lineage)
    parent_hash = lineage["checkpoint_bundle_sha256"]
    parent_fingerprint = lineage["parameter_fingerprint"]
    backend, student_config = _create_backend(config, summary)
    optimizer_config = OptimizerConfig(
        type=config.optimizer, learning_rate=config.learning_rate
    )
    resume_receipt = _load_start_state(
        config,
        parent=parent,
        parent_hash=parent_hash,
        optimizer_config=optimizer_config,
        sampling_contract=sampling_contract,
    )
    state = resume_receipt.pop("state")
    initial_params = parent.params
    initial_fingerprint = parameter_fingerprint(initial_params)
    start_step = int(state.step)
    train_step = _make_train_step(backend, optimizer_config, config.max_grad_norm)
    evaluation_interval = config.eval_every or max(1, config.steps // 20)
    evaluation_records, held_out_receipt = _resolve_evaluation_records(config, records)
    write_json(config.output_dir / "held_out_evaluation_receipt.json", held_out_receipt)
    corridor_records = _corridor_records(config)
    trajectory: list[dict[str, Any]] = []
    training_seconds = evaluation_seconds = checkpoint_seconds = 0.0
    records_consumed = tokens_consumed = 0
    visited: set[str] = set()
    best_loss = math.inf
    best_step = start_step
    best_checkpoint = config.output_dir / "checkpoints" / "best"

    def evaluate(grad_metrics: dict[str, float] | None) -> dict[str, Any]:
        nonlocal evaluation_seconds, best_loss, best_step, checkpoint_seconds
        eval_started = time.perf_counter()
        exemplar = _evaluate_exemplars(backend, state.params, evaluation_records)
        corridor = (
            _evaluate_held_out(backend, state.params, corridor_records)
            if corridor_records
            else None
        )
        point = {
            "optimizer_step": int(state.step),
            "wall_clock_seconds": time.perf_counter() - started,
            "exemplar_loss": exemplar["kl_loss"],
            "teacher_student_kl": exemplar["kl_loss"],
            "top1_agreement": exemplar["top1_agreement"],
            "topk_overlap": exemplar["topk_overlap"],
            "entropy_error": exemplar["entropy_error"],
            "parameter_delta_from_corridor": _tree_delta_norm(
                initial_params, state.params
            ),
            "gradient_norm": None
            if grad_metrics is None
            else grad_metrics["gradient_norm"],
            "clip_scale": None if grad_metrics is None else grad_metrics["clip_scale"],
            "records_consumed": records_consumed,
            "tokens_consumed": tokens_consumed,
            "corridor_metrics": corridor,
        }
        trajectory.append(point)
        evaluation_seconds += time.perf_counter() - eval_started
        if point["exemplar_loss"] < best_loss:
            best_loss = point["exemplar_loss"]
            best_step = int(state.step)
            checkpoint_started = time.perf_counter()
            _save_exemplar_checkpoint(
                config,
                state,
                student_config,
                best_checkpoint,
                parent_hash,
                parent_fingerprint,
                records_consumed,
                sampling_contract,
            )
            checkpoint_seconds += time.perf_counter() - checkpoint_started
        return point

    evaluate(None)
    stop_reason = "requested_steps_completed"
    grad_metrics: dict[str, float] | None = None
    for local_step in range(start_step + 1, config.steps + 1):
        batch = batches[(local_step - 1) % len(batches)]
        train_started = time.perf_counter()
        state, raw_metrics = train_step(state, _batch_to_jax(batch))
        jax.block_until_ready(state.params)
        training_seconds += time.perf_counter() - train_started
        grad_metrics = {key: float(value) for key, value in raw_metrics.items()}
        records_consumed += len(batch.example_id)
        tokens_consumed += int(batch.input_ids.size)
        visited.update(batch.example_id)
        if not all(math.isfinite(value) for value in grad_metrics.values()):
            stop_reason = "non_finite_training"
            break
        if local_step % evaluation_interval == 0 or local_step == config.steps:
            evaluate(grad_metrics)
        if local_step % config.checkpoint_every == 0 or local_step == config.steps:
            checkpoint_started = time.perf_counter()
            _save_exemplar_checkpoint(
                config,
                state,
                student_config,
                config.output_dir / "checkpoints" / f"step_{local_step:06d}",
                parent_hash,
                parent_fingerprint,
                records_consumed,
                sampling_contract,
            )
            checkpoint_seconds += time.perf_counter() - checkpoint_started
    if trajectory[-1]["optimizer_step"] != int(state.step):
        evaluate(grad_metrics)

    final_checkpoint = config.output_dir / "checkpoints" / "final"
    checkpoint_started = time.perf_counter()
    _save_exemplar_checkpoint(
        config,
        state,
        student_config,
        final_checkpoint,
        parent_hash,
        parent_fingerprint,
        records_consumed,
        sampling_contract,
    )
    checkpoint_seconds += time.perf_counter() - checkpoint_started
    status = "pass" if stop_reason == "requested_steps_completed" else "fail"
    initial, final = trajectory[0], trajectory[-1]
    resource = {
        "artifact_total_bytes_on_disk": sum(
            path.stat().st_size
            for path in config.fingerprint_artifact.rglob("*")
            if path.is_file()
        ),
        "exemplar_payload_bytes": sum(
            path.stat().st_size
            for path in (config.fingerprint_artifact / "exemplars").rglob("*")
            if path.is_file()
        ),
        "unique_exemplar_records_consumed": len(visited),
        "total_exemplar_record_visits": records_consumed,
        "tokens_consumed": tokens_consumed,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "checkpoint_write_seconds": checkpoint_seconds,
        "total_wall_clock_seconds": time.perf_counter() - started,
    }
    retention = _retention_report(
        trajectory,
        threshold=config.corridor_entry_threshold,
        tolerance=config.corridor_retention_tolerance,
    )
    report = {
        "phase": "P154.1.1",
        "status": status,
        "training_cycle": "exemplar",
        "parent_training_cycle": (
            "shared_initialization_control"
            if config.allow_shared_initialization_parent_for_control
            else "corridor"
        ),
        "exemplar_loss_type": "dense_probability_kl",
        "exemplar_loss_weight": 1.0,
        "corridor_loss_enabled": False,
        "causal_lm_loss_enabled": False,
        "mixed_objective_enabled": False,
        "input_checkpoint_optimizer_state_loaded": False,
        "exemplar_optimizer_state_fresh": config.resume_checkpoint is None,
        "fresh_optimizer_state": config.resume_checkpoint is None,
        "exemplar_local_step_start": start_step,
        "parent_corridor_optimizer_steps": parent.manifest.step,
        "requested_steps": config.steps,
        "completed_steps": int(state.step),
        "batch_size": config.batch_size,
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "max_grad_norm": config.max_grad_norm,
        "evaluation_interval_steps": evaluation_interval,
        "checkpoint_interval_steps": config.checkpoint_every,
        "stop_reason": stop_reason,
        "initial_exemplar_loss": initial["exemplar_loss"],
        "final_exemplar_loss": final["exemplar_loss"],
        "best_exemplar_loss": best_loss,
        "best_exemplar_step": best_step,
        "exemplar_loss_delta": final["exemplar_loss"] - initial["exemplar_loss"],
        "teacher_student_kl_delta": final["teacher_student_kl"]
        - initial["teacher_student_kl"],
        "top1_agreement_delta": final["top1_agreement"] - initial["top1_agreement"],
        "topk_overlap_delta": final["topk_overlap"] - initial["topk_overlap"],
        "parameter_delta_from_corridor": final["parameter_delta_from_corridor"],
        "params_changed": parameter_fingerprint(state.params) != initial_fingerprint,
        "final_vs_best_rebound": final["exemplar_loss"] - best_loss,
        "corridor_retention_evaluation_enabled": bool(corridor_records),
        **held_out_receipt,
        "resume_configuration_valid": resume_receipt["status"] == "pass",
        "calibration_parent_binding_valid": lineage["calibration_parent_binding_valid"],
        "p153_parent_binding_valid": lineage["checks"]["p153_parent_binding_valid"],
        **retention,
        "general_quality_claim_made": False,
        "quality_per_byte_claim_made": False,
        "scale_claim_made": False,
        "radlads_parity_claim_made": False,
    }
    efficiency = {
        "exemplar_improvement_per_step": (
            initial["exemplar_loss"] - final["exemplar_loss"]
        )
        / max(int(state.step) - start_step, 1),
        "exemplar_improvement_per_second": (
            initial["exemplar_loss"] - final["exemplar_loss"]
        )
        / max(training_seconds, 1e-12),
        "exemplar_improvement_per_record": (
            initial["exemplar_loss"] - final["exemplar_loss"]
        )
        / max(records_consumed, 1),
    }
    write_json(config.output_dir / "exemplar_pass_report.json", report)
    write_json(config.output_dir / "exemplar_efficiency_metrics.json", efficiency)
    write_json(config.output_dir / "corridor_retention_report.json", retention)
    write_json(config.output_dir / "resource_accounting.json", resource)
    write_json(config.output_dir / "resume_validation.json", resume_receipt)
    write_json(config.output_dir / "resume_receipt.json", resume_receipt)
    write_json(
        config.output_dir / "sampling_receipt.json",
        {
            **sampling_contract,
            "records_available": dataset.num_records,
        },
    )
    _write_jsonl(config.output_dir / "exemplar_trajectory.jsonl", trajectory)
    (config.output_dir / "exemplar_pass_summary.md").write_text(
        "# P154.1.1 Standalone Exemplar Pass Integrity\n\n"
        f"- Status: {status}\n"
        f"- Completed steps: {int(state.step)}\n"
        f"- Initial exemplar loss: {initial['exemplar_loss']:.8f}\n"
        f"- Final exemplar loss: {final['exemplar_loss']:.8f}\n"
        f"- Best exemplar loss: {best_loss:.8f} at step {best_step}\n"
        "- Corridor loss enabled: false\n"
        "- Mixed objective enabled: false\n"
        "- General quality claim: false\n",
        encoding="utf-8",
    )
    return ExemplarPassResult(
        status=status,
        output_dir=config.output_dir,
        report_path=config.output_dir / "exemplar_pass_report.json",
        trajectory_path=config.output_dir / "exemplar_trajectory.jsonl",
        final_checkpoint=final_checkpoint,
        best_checkpoint=best_checkpoint,
        completed_steps=int(state.step),
    )


def _sampling_contract(config, records):
    return {
        "sampling_policy": config.exemplar_sampling_policy,
        "sampling_seed": config.seed,
        "record_order_sha256": record_order_sha256(records),
        "records_selected": len(records),
        "exemplar_max_records": config.exemplar_max_records,
        "batch_size": config.batch_size,
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "max_grad_norm": config.max_grad_norm,
        "exemplar_artifact_sha256": file_sha256(
            config.fingerprint_artifact / "manifest.json"
        ),
    }


def _load_start_state(
    config, *, parent, parent_hash, optimizer_config, sampling_contract
):
    if config.resume_checkpoint is None:
        return {
            "status": "pass",
            "resumed": False,
            "input_checkpoint_optimizer_state_loaded": False,
            "sampling_policy_match": True,
            "sampling_seed_match": True,
            "record_order_match": True,
            "record_limit_match": True,
            "batch_size_match": True,
            "optimizer_match": True,
            "learning_rate_match": True,
            "max_grad_norm_match": True,
            "artifact_match": True,
            "parent_corridor_match": True,
            "state": TrainState(
                params=parent.params,
                step=0,
                learning_rate=config.learning_rate,
                optimizer_state=init_optimizer_state(parent.params, optimizer_config),
            ),
        }
    resumed = load_checkpoint(config.resume_checkpoint)
    stored = {
        **resumed.manifest.loss_config,
        **resumed.manifest.target_manifest,
        **resumed.manifest.optimizer_config,
        **resumed.manifest.gradients,
    }
    checks = {
        "training_cycle_match": resumed.manifest.loss_config.get("cycle") == 2,
        "parent_corridor_match": resumed.manifest.target_manifest.get(
            "parent_checkpoint_bundle_sha256"
        )
        == parent_hash,
        "sampling_policy_match": stored.get("sampling_policy")
        == sampling_contract["sampling_policy"],
        "sampling_seed_match": stored.get("sampling_seed")
        == sampling_contract["sampling_seed"],
        "record_limit_match": stored.get("exemplar_max_records")
        == sampling_contract["exemplar_max_records"],
        "batch_size_match": stored.get("batch_size") == sampling_contract["batch_size"],
        "record_order_match": stored.get("record_order_sha256")
        == sampling_contract["record_order_sha256"],
        "optimizer_match": stored.get("type") == sampling_contract["optimizer"],
        "learning_rate_match": stored.get("learning_rate")
        == sampling_contract["learning_rate"],
        "max_grad_norm_match": stored.get("max_grad_norm")
        == sampling_contract["max_grad_norm"],
        "artifact_match": stored.get("exemplar_artifact_sha256")
        == sampling_contract["exemplar_artifact_sha256"],
        "optimizer_state_present": resumed.optimizer_state is not None,
    }
    if not all(checks.values()):
        errors = {
            "training_cycle_match": "resume_training_cycle_mismatch",
            "sampling_policy_match": "resume_sampling_policy_mismatch",
            "sampling_seed_match": "resume_sampling_seed_mismatch",
            "record_order_match": "resume_record_order_mismatch",
            "record_limit_match": "resume_record_limit_mismatch",
            "batch_size_match": "resume_batch_size_mismatch",
            "optimizer_match": "resume_optimizer_mismatch",
            "learning_rate_match": "resume_learning_rate_mismatch",
            "max_grad_norm_match": "resume_max_grad_norm_mismatch",
            "artifact_match": "resume_artifact_mismatch",
            "parent_corridor_match": "resume_parent_corridor_mismatch",
            "optimizer_state_present": "resume_optimizer_state_missing",
        }
        failed = next(name for name, passed in checks.items() if not passed)
        raise ValueError(errors[failed])
    return {
        "status": "pass",
        "resumed": True,
        "input_checkpoint_optimizer_state_loaded": True,
        **checks,
        "state": TrainState(
            params=resumed.params,
            step=resumed.manifest.step,
            learning_rate=config.learning_rate,
            optimizer_state=resumed.optimizer_state,
        ),
    }


def _make_train_step(backend, optimizer_config, max_grad_norm):
    def train_step(state, batch):
        def loss_fn(params):
            output, _ = backend.forward_full(params, batch["input_ids"])
            loss = compute_fingerprint_exemplar_loss_at_positions(
                backend.logits(output), _jax_exemplar_batch(batch)
            )
            return loss.loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        clipped = clip_gradients_by_global_norm(grads, max_grad_norm=max_grad_norm)
        params, optimizer_state, _ = optimizer_update(
            state.params, clipped.gradients, state.optimizer_state, optimizer_config
        )
        return TrainState(
            params=params,
            step=state.step + 1,
            learning_rate=state.learning_rate,
            optimizer_state=optimizer_state,
        ), {
            "loss": loss,
            "gradient_norm": clipped.global_norm,
            "clip_scale": clipped.clip_scale,
        }

    return jax.jit(train_step)


def _evaluate_exemplars(backend, params, records):
    losses = []
    top1 = []
    overlaps = []
    entropy_errors = []
    for record in records:
        output, _ = backend.forward_full(
            params, jnp.asarray([record.input_ids], dtype=jnp.int32)
        )
        logits = select_position_logits(
            backend.logits(output), jnp.asarray([record.position], dtype=jnp.int32)
        )[0]
        probs = jax.nn.softmax(logits)
        teacher = jnp.asarray(record.teacher_probs, dtype=jnp.float32)
        losses.append(
            float(
                jnp.sum(
                    teacher * (jnp.log(teacher + 1e-8) - jax.nn.log_softmax(logits))
                )
            )
        )
        top1.append(int(jnp.argmax(probs) == jnp.argmax(teacher)))
        k = min(8, int(teacher.shape[0]))
        student_top = set(np.asarray(jnp.argsort(probs)[-k:]).tolist())
        teacher_top = set(np.asarray(jnp.argsort(teacher)[-k:]).tolist())
        overlaps.append(len(student_top & teacher_top) / k)
        student_entropy = -jnp.sum(probs * jnp.log(probs + 1e-8))
        teacher_entropy = -jnp.sum(teacher * jnp.log(teacher + 1e-8))
        entropy_errors.append(abs(float(student_entropy - teacher_entropy)))
    return {
        "kl_loss": float(np.mean(losses)),
        "top1_agreement": float(np.mean(top1)),
        "topk_overlap": float(np.mean(overlaps)),
        "entropy_error": float(np.mean(entropy_errors)),
    }


def _save_exemplar_checkpoint(
    config,
    state,
    student_config,
    path,
    parent_hash,
    parent_fingerprint,
    records_consumed,
    sampling_contract,
):
    save_checkpoint(
        path,
        state.params,
        student_architecture=config.student_backend,
        student_config=student_config,
        step=int(state.step),
        learning_rate=config.learning_rate,
        loss_config={
            "kind": "fingerprint_exemplar",
            "cycle": 2,
            "training_cycle": "exemplar",
            "parent_training_cycle": (
                "shared_initialization_control"
                if config.allow_shared_initialization_parent_for_control
                else "corridor"
            ),
            "sampling_policy": config.exemplar_sampling_policy,
            "sampling_seed": sampling_contract["sampling_seed"],
            "record_order_sha256": sampling_contract["record_order_sha256"],
            "records_selected": sampling_contract["records_selected"],
            "exemplar_max_records": sampling_contract["exemplar_max_records"],
            "batch_size": sampling_contract["batch_size"],
            "corridor_loss_enabled": False,
            "mixed_objective_enabled": False,
        },
        target_manifest={
            "artifact_dir": str(config.fingerprint_artifact),
            "exemplar_artifact_hash": sampling_contract["exemplar_artifact_sha256"],
            "exemplar_artifact_sha256": sampling_contract["exemplar_artifact_sha256"],
            "parent_checkpoint_bundle_sha256": parent_hash,
            "parent_parameter_fingerprint": parent_fingerprint,
            "exemplar_records_consumed": records_consumed,
            "exemplar_sampling_policy": config.exemplar_sampling_policy,
            "exemplar_optimizer_steps_completed": int(state.step),
        },
        optimizer_config={
            "type": config.optimizer,
            "learning_rate": config.learning_rate,
        },
        optimizer_state=state.optimizer_state,
        gradients={"max_grad_norm": config.max_grad_norm},
        notes=["P154 standalone exemplar-only checkpoint"],
        overwrite=True,
    )


def _create_backend(config, summary):
    contract = VocabContract(
        tokenizer_id=summary.tokenizer_name or "fingerprint-artifact",
        tokenizer_hash=summary.tokenizer_name or None,
        vocab_size=summary.vocab_size,
        model_id=summary.teacher_model_name or None,
        model_family="behavioral_fingerprint",
    )
    backend = create_student_backend(
        vocab_contract=contract, architecture_id=config.student_backend
    )
    raw = getattr(getattr(backend, "student", None), "config", None)
    student_config = {
        "architecture_id": config.student_backend,
        "backend_name": type(backend).__name__,
        "architecture": type(raw).__name__,
        "vocab_size": int(getattr(raw, "vocab_size", summary.vocab_size)),
        "hidden_size": int(getattr(raw, "hidden_size", 0)),
        "num_layers": int(getattr(raw, "num_layers", 0)),
        "num_heads": _optional_int(getattr(raw, "num_heads", None)),
        "num_kv_heads": _optional_int(getattr(raw, "num_kv_heads", None)),
        "emit_logits": bool(getattr(raw, "emit_logits", True)),
        "tie_embeddings": bool(getattr(raw, "tie_embeddings", False)),
        "emit_mixer_outputs": bool(getattr(raw, "emit_mixer_outputs", False)),
    }
    return backend, student_config


def _resolve_evaluation_records(config, training_records):
    if config.held_out_fingerprint_artifact is None:
        return training_records, {
            "status": "pass",
            "exemplar_evaluation_split": "training",
            "held_out_artifact_supplied": False,
            "held_out_exemplar_count": None,
            "training_exemplar_fallback_used": False,
        }
    try:
        dataset = load_fingerprint_exemplars(
            config.held_out_fingerprint_artifact,
            batch_size=1,
            require_exemplars=True,
        )
        records = tuple(dataset.iter_records())
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError("held_out_exemplar_reservoir_missing") from exc
    if not records:
        raise ValueError("held_out_exemplar_reservoir_missing")
    return records, {
        "status": "pass",
        "exemplar_evaluation_split": "held_out",
        "held_out_artifact_supplied": True,
        "held_out_exemplar_count": len(records),
        "training_exemplar_fallback_used": False,
    }


def _corridor_records(config) -> tuple[FingerprintTargetRecord, ...]:
    if config.held_out_fingerprint_artifact is None:
        return ()
    return tuple(
        load_fingerprint_targets(
            config.held_out_fingerprint_artifact, batch_size=1
        ).iter_records()
    )


def _retention_report(trajectory, *, threshold, tolerance=0.0):
    corridor_points = [
        point for point in trajectory if point.get("corridor_metrics") is not None
    ]
    if not corridor_points:
        return {
            "corridor_entry_threshold": threshold,
            "initial_inside_all_rate": None,
            "final_inside_all_rate": None,
            "initial_corridor_metrics": None,
            "final_corridor_metrics": None,
            "corridor_retention_delta": None,
            "corridor_retention_degraded": False,
            "corridor_exit_detected": False,
            "first_corridor_exit_step": None,
        }
    initial = corridor_points[0]["corridor_metrics"]
    final = corridor_points[-1]["corridor_metrics"]
    initial_inside = float(initial["inside_all_rate"])
    final_inside = float(final["inside_all_rate"])
    degraded = bool(
        final_inside < initial_inside - tolerance
        or float(final["mean_distance_outside_corridor"])
        > float(initial["mean_distance_outside_corridor"]) + tolerance
        or float(final["corridor_loss"]) > float(initial["corridor_loss"]) + tolerance
    )
    first_exit_step = None
    if initial_inside >= threshold:
        first_exit_step = next(
            (
                int(point["optimizer_step"])
                for point in corridor_points[1:]
                if float(point["corridor_metrics"]["inside_all_rate"]) < threshold
            ),
            None,
        )
    return {
        "corridor_entry_threshold": threshold,
        "initial_inside_all_rate": initial_inside,
        "final_inside_all_rate": final_inside,
        "initial_corridor_metrics": initial,
        "final_corridor_metrics": final,
        "corridor_retention_delta": {
            "inside_all_rate": final_inside - initial_inside,
            "mean_distance_outside_corridor": final["mean_distance_outside_corridor"]
            - initial["mean_distance_outside_corridor"],
            "held_out_corridor_loss": final["corridor_loss"] - initial["corridor_loss"],
        },
        "corridor_retention_degraded": degraded,
        "corridor_exit_detected": first_exit_step is not None,
        "first_corridor_exit_step": first_exit_step,
    }


def _records_to_batches(records, batch_size, max_seq_len):
    batches = []
    for start in range(0, len(records), batch_size):
        selected = records[start : start + batch_size]
        batches.append(
            FingerprintExemplarBatch(
                input_ids=np.asarray(
                    [r.input_ids for r in selected], dtype=np.int32
                ).reshape((len(selected), max_seq_len)),
                position=np.asarray([r.position for r in selected], dtype=np.int32),
                teacher_probs=np.asarray(
                    [r.teacher_probs for r in selected], dtype=np.float32
                ),
                weight=np.asarray([r.weight for r in selected], dtype=np.float32),
                mode_id=np.asarray(
                    [-1 if r.mode_id is None else r.mode_id for r in selected],
                    dtype=np.int32,
                ),
                interestingness_score=np.asarray(
                    [
                        np.nan
                        if r.interestingness_score is None
                        else r.interestingness_score
                        for r in selected
                    ],
                    dtype=np.float32,
                ),
                reason_codes=tuple(r.reason_codes for r in selected),
                example_id=tuple(r.example_id for r in selected),
            )
        )
    return tuple(batches)


def _batch_to_jax(batch):
    return {
        "input_ids": jnp.asarray(batch.input_ids),
        "position": jnp.asarray(batch.position),
        "teacher_probs": jnp.asarray(batch.teacher_probs),
        "weight": jnp.asarray(batch.weight),
    }


def _jax_exemplar_batch(batch):
    return FingerprintExemplarBatch(
        input_ids=batch["input_ids"],
        position=batch["position"],
        teacher_probs=batch["teacher_probs"],
        weight=batch["weight"],
        mode_id=jnp.zeros_like(batch["position"]),
        interestingness_score=jnp.zeros_like(batch["weight"]),
        reason_codes=(),
        example_id=(),
    )


def _validate_exemplar_records(records, vocab_size):
    for record in records:
        if not 0 <= record.position < len(record.input_ids):
            raise ValueError("exemplar position is outside input sequence")
        probs = np.asarray(record.teacher_probs, dtype=np.float64)
        if len(probs) != vocab_size or not np.all(np.isfinite(probs)):
            raise ValueError("exemplar target payload must be finite and match vocab")
        if np.any(probs < 0.0) or not np.isclose(np.sum(probs), 1.0, atol=1e-5):
            raise ValueError("exemplar teacher probabilities must be normalized")


def _validate_config(config):
    if config.output_dir.exists() and not config.overwrite:
        raise FileExistsError(
            f"output exists: {config.output_dir}; pass overwrite=True"
        )
    if config.steps < 1 or config.batch_size < 1 or config.checkpoint_every < 1:
        raise ValueError("steps, batch_size, and checkpoint_every must be >= 1")
    if config.eval_every is not None and config.eval_every < 1:
        raise ValueError("eval_every must be >= 1")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if not 0.0 <= config.corridor_entry_threshold <= 1.0:
        raise ValueError("corridor_entry_threshold must be within [0, 1]")
    if config.corridor_retention_tolerance < 0.0:
        raise ValueError("corridor_retention_tolerance must be >= 0")


def _tree_delta_norm(before, after):
    return float(
        jnp.sqrt(
            sum(
                jnp.sum(jnp.square(jnp.asarray(right) - jnp.asarray(left)))
                for left, right in zip(
                    jax.tree_util.tree_leaves(before),
                    jax.tree_util.tree_leaves(after),
                    strict=True,
                )
            )
        )
    )


def _optional_int(value):
    return None if value is None else int(value)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
