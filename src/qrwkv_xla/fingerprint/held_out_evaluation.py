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
    FingerprintTargetRecord,
    load_fingerprint_exemplars,
    load_fingerprint_targets,
    summarize_fingerprint_artifact,
)
from qrwkv_xla.artifacts._json import read_json_object, write_json
from qrwkv_xla.checkpointing import load_checkpoint
from qrwkv_xla.contracts import VocabContract
from qrwkv_xla.fingerprint.provenance import (
    build_artifact_source_lineage,
    file_sha256,
    hash_checkpoint_bundle,
    parameter_fingerprint,
    stable_hash,
)
from qrwkv_xla.students import create_student_backend
from qrwkv_xla.training.fingerprint_stats import (
    compute_fingerprint_distribution_stats_at_positions,
)

PROVENANCE_NAME = "fingerprint_provenance.json"
PRIMARY_METRIC = "held_out_corridor_loss_total"
SECONDARY_METRICS = (
    "inside_all_rate",
    "mean_distance_outside_corridor",
    "entropy_absolute_error",
    "top1_margin_absolute_error",
    "top8_mass_absolute_error",
    "top32_mass_absolute_error",
    "tail_mass_absolute_error",
    "teacher_student_kl",
)


@dataclass(frozen=True)
class HeldOutFingerprintEvaluationConfig:
    baseline_checkpoint: Path
    fingerprint_checkpoint: Path
    held_out_fingerprint_artifact: Path
    train_fingerprint_artifact: Path
    output_dir: Path
    p151_report: Path | None = None
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 0
    tie_tolerance: float = 1e-12
    overwrite: bool = False


@dataclass(frozen=True)
class HeldOutFingerprintEvaluationResult:
    status: str
    winner: str
    output_dir: Path
    report_path: Path
    metrics_path: Path
    summary_path: Path


def write_fingerprint_provenance(
    artifact_dir: Path,
    *,
    source_file: Path,
    artifact_role: str,
    allow_legacy_positional_source_join: bool = False,
) -> Path:
    if artifact_role not in {"training", "held_out_evaluation"}:
        raise ValueError("artifact_role must be training or held_out_evaluation")
    lineage = build_artifact_source_lineage(
        artifact_dir,
        source_file,
        allow_legacy_positional_source_join=(allow_legacy_positional_source_join),
    )
    manifest = read_json_object(artifact_dir / "manifest.json")
    capture_summary = _optional_json(artifact_dir / "capture_summary.json")
    split_payload = {
        "artifact_role": artifact_role,
        "ordered_example_ids": lineage["ordered_example_ids"],
        "source_text_hashes": lineage["source_text_hashes"],
        "token_sequence_hashes": lineage["token_sequence_hashes"],
    }
    payload = {
        "schema_version": "qrwkv_xla.fingerprint_provenance.v1",
        "artifact_type": "behavioral_fingerprint",
        "artifact_role": artifact_role,
        "artifact_version": str(manifest.get("artifact_version", "")),
        "source_file": str(source_file),
        "source_file_sha256": lineage["source_file_sha256"],
        "ordered_example_ids": split_payload["ordered_example_ids"],
        "ordered_example_ids_sha256": stable_hash(split_payload["ordered_example_ids"]),
        "source_text_hashes": lineage["source_text_hashes"],
        "ordered_source_text_sha256": lineage["ordered_source_text_sha256"],
        "token_sequence_hashes": split_payload["token_sequence_hashes"],
        "tokenized_inputs_sha256": lineage["tokenized_inputs_sha256"],
        "capture_config_sha256": lineage["capture_config_sha256"],
        "teacher_identity_sha256": lineage["teacher_identity_sha256"],
        "split_manifest_sha256": stable_hash(split_payload),
        "artifact_manifest_sha256": lineage["artifact_manifest_sha256"],
        "source_example_set_sha256": lineage["source_example_set_sha256"],
        "source_join_kind": lineage["source_join_kind"],
        "source_join_complete": lineage["source_join_complete"],
        "lineage_confidence": lineage["lineage_confidence"],
        "publication_grade_lineage": lineage["publication_grade_lineage"],
        "warnings": lineage["warnings"],
        "tokenizer_identity": manifest.get("teacher", {}).get("tokenizer_name"),
        "teacher_model_id": manifest.get("teacher", {}).get("model_name"),
        "teacher_revision": manifest.get("teacher", {}).get("revision"),
        "tokenizer_revision": manifest.get("teacher", {}).get("tokenizer_revision"),
        "capture_command": capture_summary.get("capture_command"),
        "capture_timestamp": capture_summary.get("capture_timestamp"),
        "software_commit": capture_summary.get("software_commit"),
    }
    path = artifact_dir / PROVENANCE_NAME
    write_json(path, payload)
    return path


def validate_fingerprint_provenance(
    artifact_dir: Path,
    *,
    expected_role: str,
) -> dict[str, Any]:
    path = artifact_dir / PROVENANCE_NAME
    payload = read_json_object(path)
    lineage = build_artifact_source_lineage(
        artifact_dir,
        Path(str(payload.get("source_file", ""))),
        allow_legacy_positional_source_join=(
            payload.get("source_join_kind") == "legacy_positional"
        ),
    )
    expected = {
        "source_file_sha256": lineage["source_file_sha256"],
        "ordered_example_ids_sha256": lineage["ordered_example_ids_sha256"],
        "source_example_set_sha256": lineage["source_example_set_sha256"],
        "tokenized_inputs_sha256": lineage["tokenized_inputs_sha256"],
        "capture_config_sha256": lineage["capture_config_sha256"],
        "teacher_identity_sha256": lineage["teacher_identity_sha256"],
        "artifact_manifest_sha256": lineage["artifact_manifest_sha256"],
        "ordered_source_text_sha256": lineage["ordered_source_text_sha256"],
        "split_manifest_sha256": stable_hash(
            {
                "artifact_role": payload.get("artifact_role"),
                "ordered_example_ids": payload.get("ordered_example_ids"),
                "source_text_hashes": payload.get("source_text_hashes"),
                "token_sequence_hashes": payload.get("token_sequence_hashes"),
            }
        ),
    }
    blockers = []
    if payload.get("artifact_role") != expected_role:
        blockers.append(
            "artifact role mismatch: "
            f"{payload.get('artifact_role')!r} != {expected_role!r}"
        )
    required = (
        "source_file_sha256",
        "ordered_example_ids_sha256",
        "ordered_source_text_sha256",
        "tokenized_inputs_sha256",
        "capture_config_sha256",
        "teacher_identity_sha256",
        "split_manifest_sha256",
        "artifact_manifest_sha256",
    )
    blockers.extend(
        f"missing required provenance hash: {key}"
        for key in required
        if not payload.get(key)
    )
    blockers.extend(
        f"provenance hash mismatch: {key}"
        for key, value in expected.items()
        if payload.get(key) != value
    )
    if payload.get("source_text_hashes") != lineage["source_text_hashes"]:
        blockers.append("provenance hash mismatch: source_text_hashes")
    return {
        "status": "pass" if not blockers else "fail",
        "valid": not blockers,
        "blockers": blockers,
        "path": str(path),
        "provenance": payload,
    }


def paired_bootstrap_interval(
    deltas: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired bootstrap requires a non-empty rank-1 array")
    if samples <= 0:
        raise ValueError("bootstrap_samples must be > 0")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = np.mean(rng.choice(values, size=values.size, replace=True))
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def select_held_out_winner(
    mean_delta: float,
    ci95: tuple[float, float],
    *,
    tolerance: float,
) -> str:
    if abs(mean_delta) <= tolerance or ci95[0] <= 0.0 <= ci95[1]:
        return "inconclusive"
    return "fingerprint" if mean_delta > 0.0 else "baseline"


def run_held_out_fingerprint_evaluation(
    config: HeldOutFingerprintEvaluationConfig,
) -> HeldOutFingerprintEvaluationResult:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_provenance = validate_fingerprint_provenance(
        config.train_fingerprint_artifact,
        expected_role="training",
    )
    held_out_provenance = validate_fingerprint_provenance(
        config.held_out_fingerprint_artifact,
        expected_role="held_out_evaluation",
    )
    split_validation = _validate_split(
        config,
        train_provenance=train_provenance,
        held_out_provenance=held_out_provenance,
    )
    if not split_validation["valid"]:
        raise ValueError(
            "held-out split validation failed: "
            + "; ".join(split_validation["blockers"])
        )
    p151_report_path = config.p151_report or _discover_p151_report(
        config.baseline_checkpoint
    )
    p151_report = read_json_object(p151_report_path)
    if not p151_report.get("fairness", {}).get("comparison_valid", False):
        raise ValueError("P151 fairness contract is missing or invalid")
    if Path(str(p151_report.get("artifact_dir", ""))).resolve() != (
        config.train_fingerprint_artifact.resolve()
    ):
        raise ValueError("P151 report training artifact does not match P152 input")
    checkpoint_validation = _validate_checkpoints(
        config,
        p151_report,
        p151_report_path=p151_report_path,
        train_provenance=train_provenance,
    )
    write_json(
        config.output_dir / "checkpoint_lineage_validation.json",
        checkpoint_validation,
    )
    if not checkpoint_validation["valid"]:
        raise ValueError(
            "checkpoint validation failed: "
            + "; ".join(checkpoint_validation["blockers"])
        )

    held_out_records = _target_records(config.held_out_fingerprint_artifact)
    exemplars = _exemplar_map(config.held_out_fingerprint_artifact)
    baseline_eval = _evaluate_checkpoint(
        config.baseline_checkpoint,
        config.held_out_fingerprint_artifact,
        held_out_records,
        exemplars,
    )
    fingerprint_eval = _evaluate_checkpoint(
        config.fingerprint_checkpoint,
        config.held_out_fingerprint_artifact,
        held_out_records,
        exemplars,
    )
    if baseline_eval["record_keys"] != fingerprint_eval["record_keys"]:
        raise ValueError("checkpoints were not evaluated in identical record order")
    if not _evaluation_metrics_finite(baseline_eval, fingerprint_eval):
        raise ValueError("held-out evaluation produced non-finite metrics")
    baseline_eval["aggregate"]["primary_bootstrap_ci95"] = list(
        paired_bootstrap_interval(
            np.asarray([row["primary_score"] for row in baseline_eval["records"]]),
            samples=config.bootstrap_samples,
            seed=config.bootstrap_seed,
        )
    )
    fingerprint_eval["aggregate"]["primary_bootstrap_ci95"] = list(
        paired_bootstrap_interval(
            np.asarray([row["primary_score"] for row in fingerprint_eval["records"]]),
            samples=config.bootstrap_samples,
            seed=config.bootstrap_seed,
        )
    )
    per_records, deltas = _paired_records(baseline_eval, fingerprint_eval)
    delta_values = np.asarray(deltas, dtype=np.float64)
    delta_ci = paired_bootstrap_interval(
        delta_values,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed,
    )
    mean_delta = float(np.mean(delta_values))
    winner = select_held_out_winner(
        mean_delta,
        delta_ci,
        tolerance=config.tie_tolerance,
    )
    paired = _paired_statistics(delta_values, delta_ci)
    comparison = _comparison(baseline_eval, fingerprint_eval, paired, winner)
    report = {
        "phase": "P152.1",
        "status": "pass",
        "comparison_kind": "held_out_fingerprint_evaluation",
        "primary_metric_name": PRIMARY_METRIC,
        "primary_metric_direction": "lower_is_better",
        "secondary_metric_names": list(SECONDARY_METRICS),
        "metric_selection_predeclared": True,
        "comparison_valid": True,
        "comparison_lineage_valid": True,
        "held_out_evaluation_allowed": True,
        "source_join_kind": train_provenance["provenance"]["source_join_kind"],
        "source_join_complete": train_provenance["provenance"]["source_join_complete"],
        "lineage_confidence": train_provenance["provenance"]["lineage_confidence"],
        "publication_grade_lineage": train_provenance["provenance"][
            "publication_grade_lineage"
        ],
        "winner": winner,
        "winner_declared": winner != "inconclusive",
        "winner_scope": "held_out_fingerprint_primary_metric_only",
        "split_validation": split_validation,
        "checkpoint_validation": checkpoint_validation,
        "arms": {
            "baseline": baseline_eval["aggregate"],
            "fingerprint": fingerprint_eval["aggregate"],
        },
        "comparison": comparison,
        "paired_statistics": paired,
        "teacher_distribution_metrics": {
            "availability": baseline_eval["teacher_metric_availability"],
            "baseline": baseline_eval["teacher_metrics"],
            "fingerprint": fingerprint_eval["teacher_metrics"],
        },
        "general_quality_claim_made": False,
        "quality_per_byte_claim_made": False,
        "scale_claim_made": False,
        "radlads_parity_claim_made": False,
        "training_performed": False,
        "exemplar_training_enabled": False,
        "limitations": [
            "The winner scope is limited to this held-out fingerprint primary metric.",
            "No downstream language-quality or scale claim is made.",
        ],
    }
    paths = _write_outputs(
        config,
        report=report,
        per_records=per_records,
        train_provenance=train_provenance,
        held_out_provenance=held_out_provenance,
    )
    return HeldOutFingerprintEvaluationResult(
        status="pass",
        winner=winner,
        output_dir=config.output_dir,
        report_path=paths["report"],
        metrics_path=paths["metrics"],
        summary_path=paths["summary"],
    )


def _validate_config(config: HeldOutFingerprintEvaluationConfig) -> None:
    if config.bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be > 0")
    if config.tie_tolerance < 0:
        raise ValueError("tie_tolerance must be >= 0")
    if config.output_dir.exists() and not config.overwrite:
        raise FileExistsError(
            f"output exists: {config.output_dir}; pass overwrite=True"
        )


def _validate_split(
    config: HeldOutFingerprintEvaluationConfig,
    *,
    train_provenance: dict[str, Any],
    held_out_provenance: dict[str, Any],
) -> dict[str, Any]:
    blockers = [
        *train_provenance["blockers"],
        *held_out_provenance["blockers"],
    ]
    train = train_provenance["provenance"]
    held_out = held_out_provenance["provenance"]
    publication_grade = bool(
        train.get("publication_grade_lineage", False)
        and held_out.get("publication_grade_lineage", False)
    )
    if not publication_grade:
        blockers.append("publication_grade_lineage_required")
    train_ids = set(train.get("ordered_example_ids", ()))
    held_out_ids = set(held_out.get("ordered_example_ids", ()))
    train_tokens = set(train.get("token_sequence_hashes", ()))
    held_out_tokens = set(held_out.get("token_sequence_hashes", ()))
    train_texts = set(train.get("source_text_hashes", ()))
    held_out_texts = set(held_out.get("source_text_hashes", ()))
    id_overlap = sorted(train_ids & held_out_ids)
    token_overlap = sorted(train_tokens & held_out_tokens)
    text_overlap = sorted(train_texts & held_out_texts)
    if id_overlap:
        blockers.append("training and held-out example IDs overlap")
    if token_overlap:
        blockers.append("training and held-out tokenized inputs overlap")
    if text_overlap:
        blockers.append("training and held-out source texts overlap")
    train_summary = summarize_fingerprint_artifact(config.train_fingerprint_artifact)
    held_out_summary = summarize_fingerprint_artifact(
        config.held_out_fingerprint_artifact
    )
    contracts = {
        "same_vocab_size": train_summary.vocab_size == held_out_summary.vocab_size,
        "same_tokenizer": train_summary.tokenizer_name
        == held_out_summary.tokenizer_name,
        "same_teacher_identity": train["teacher_identity_sha256"]
        == held_out["teacher_identity_sha256"],
        "same_capture_config": train["capture_config_sha256"]
        == held_out["capture_config_sha256"],
        "same_tracked_stats": train_summary.tracked_stats
        == held_out_summary.tracked_stats,
        "same_sequence_length": train_summary.max_seq_len
        == held_out_summary.max_seq_len,
    }
    blockers.extend(
        key.replace("same_", "") + " mismatch"
        for key, valid in contracts.items()
        if not valid
    )
    return {
        "status": "pass" if not blockers else "fail",
        "valid": not blockers,
        "blockers": blockers,
        "training_example_count": len(train_ids),
        "held_out_example_count": len(held_out_ids),
        "id_overlap_count": len(id_overlap),
        "token_sequence_overlap_count": len(token_overlap),
        "source_text_overlap_count": len(text_overlap),
        "source_join_kind": held_out.get("source_join_kind"),
        "source_join_complete": held_out.get("source_join_complete", False),
        "lineage_confidence": ("full" if publication_grade else "reduced"),
        "publication_grade_lineage": publication_grade,
        **contracts,
    }


def _validate_checkpoints(
    config: HeldOutFingerprintEvaluationConfig,
    p151_report: dict[str, Any],
    *,
    p151_report_path: Path,
    train_provenance: dict[str, Any],
) -> dict[str, Any]:
    lineage = p151_report.get("lineage")
    if not isinstance(lineage, dict):
        return {
            "status": "fail",
            "comparison_lineage_valid": False,
            "held_out_evaluation_allowed": False,
            "blockers": ["missing_required_p151_lineage_fields"],
            "p151_report_sha256": file_sha256(p151_report_path),
        }
    declared_arms = lineage.get("arms")
    shared = lineage.get("shared_initialization")
    training_artifact = lineage.get("training_artifact")
    if not all(
        isinstance(value, dict) for value in (declared_arms, shared, training_artifact)
    ):
        return {
            "status": "fail",
            "comparison_lineage_valid": False,
            "held_out_evaluation_allowed": False,
            "blockers": ["missing_required_p151_lineage_fields"],
            "p151_report_sha256": file_sha256(p151_report_path),
        }
    if not all(name in declared_arms for name in ("baseline", "fingerprint")):
        return {
            "status": "fail",
            "comparison_lineage_valid": False,
            "held_out_evaluation_allowed": False,
            "blockers": ["missing_required_p151_lineage_fields"],
            "p151_report_sha256": file_sha256(p151_report_path),
        }

    blockers: list[str] = []
    rows: dict[str, Any] = {}
    loaded: dict[str, Any] = {}
    held_out = summarize_fingerprint_artifact(config.held_out_fingerprint_artifact)
    expected_roles = {
        "baseline": "conventional_causal_lm",
        "fingerprint": "fingerprint_corridor",
    }
    for arm_name, path in (
        ("baseline", config.baseline_checkpoint),
        ("fingerprint", config.fingerprint_checkpoint),
    ):
        started = time.perf_counter()
        checkpoint = load_checkpoint(path)
        loaded[arm_name] = checkpoint
        actual_hashes = hash_checkpoint_bundle(path)
        declared = declared_arms.get(arm_name, {})
        actual_fingerprint = parameter_fingerprint(checkpoint.params)
        checks = {
            "metadata_sha256_match": actual_hashes["checkpoint_metadata_sha256"]
            == declared.get("checkpoint_metadata_sha256"),
            "params_sha256_match": actual_hashes["params_sha256"]
            == declared.get("params_sha256"),
            "bundle_sha256_match": actual_hashes["checkpoint_bundle_sha256"]
            == declared.get("checkpoint_bundle_sha256"),
            "parameter_fingerprint_match": actual_fingerprint
            == declared.get("final_parameter_fingerprint"),
            "optimizer_steps_match": checkpoint.manifest.step
            == declared.get("optimizer_steps_completed"),
            "requested_steps_match": declared.get("requested_steps")
            == p151_report.get(arm_name, {}).get("requested_steps"),
            "arm_role_match": declared.get("training_arm") == expected_roles[arm_name],
            "student_architecture_match": checkpoint.manifest.student_architecture
            == declared.get("student_architecture"),
            "student_backend_match": checkpoint.manifest.student_architecture
            == declared.get("student_backend"),
            "student_config_match": stable_hash(checkpoint.manifest.student_config)
            == declared.get("student_config_sha256"),
            "training_artifact_match": declared.get("training_artifact_manifest_sha256")
            == training_artifact.get("manifest_sha256")
            == train_provenance["provenance"].get("artifact_manifest_sha256"),
            "source_example_set_match": declared.get(
                "training_source_example_set_sha256"
            )
            == training_artifact.get("source_example_set_sha256")
            == train_provenance["provenance"].get("source_example_set_sha256"),
            "shared_initialization_match": declared.get(
                "shared_initial_parameter_fingerprint"
            )
            == shared.get("parameter_fingerprint"),
            "software_commit_match": declared.get("software_commit")
            == lineage.get("software_commit"),
        }
        if not all(checks.values()):
            blockers.append(f"{arm_name}_checkpoint_lineage_mismatch")
        rows[arm_name] = {
            "checkpoint_dir": str(path),
            **actual_hashes,
            "parameter_fingerprint": actual_fingerprint,
            "source_phase": "P151",
            "training_arm": expected_roles[arm_name],
            "optimizer_steps_completed": checkpoint.manifest.step,
            "checkpoint_load_seconds": time.perf_counter() - started,
            "student_architecture": checkpoint.manifest.student_architecture,
            "student_config": checkpoint.manifest.student_config,
            **checks,
        }

    if (
        int(rows["baseline"]["student_config"].get("vocab_size", -1))
        != held_out.vocab_size
    ):
        blockers.append("student_config_lineage_mismatch")
    if jax.tree_util.tree_structure(
        loaded["baseline"].params
    ) != jax.tree_util.tree_structure(loaded["fingerprint"].params):
        blockers.append("student_config_lineage_mismatch")
    cross_arm = {
        "shared_initialization_match": declared_arms["baseline"].get(
            "shared_initial_parameter_fingerprint"
        )
        == declared_arms["fingerprint"].get("shared_initial_parameter_fingerprint")
        == shared.get("parameter_fingerprint"),
        "step_budget_match": declared_arms["baseline"].get("requested_steps")
        == declared_arms["fingerprint"].get("requested_steps")
        and declared_arms["baseline"].get("optimizer_steps_completed")
        == declared_arms["fingerprint"].get("optimizer_steps_completed"),
        "training_artifact_match": declared_arms["baseline"].get(
            "training_artifact_manifest_sha256"
        )
        == declared_arms["fingerprint"].get("training_artifact_manifest_sha256"),
        "source_example_set_match": declared_arms["baseline"].get(
            "training_source_example_set_sha256"
        )
        == declared_arms["fingerprint"].get("training_source_example_set_sha256"),
    }
    provenance = train_provenance["provenance"]
    training_provenance_match = all(
        (
            training_artifact.get("manifest_sha256")
            == provenance.get("artifact_manifest_sha256"),
            training_artifact.get("source_file_sha256")
            == provenance.get("source_file_sha256"),
            training_artifact.get("ordered_example_ids_sha256")
            == provenance.get("ordered_example_ids_sha256"),
            training_artifact.get("source_example_set_sha256")
            == provenance.get("source_example_set_sha256"),
            training_artifact.get("tokenized_inputs_sha256")
            == provenance.get("tokenized_inputs_sha256"),
            training_artifact.get("capture_config_sha256")
            == provenance.get("capture_config_sha256"),
        )
    )
    if not cross_arm["shared_initialization_match"]:
        blockers.append("shared_initialization_lineage_mismatch")
    if not cross_arm["step_budget_match"]:
        blockers.append("checkpoint_step_budget_mismatch")
    if not cross_arm["training_artifact_match"]:
        blockers.append("training_artifact_lineage_mismatch")
    if not cross_arm["source_example_set_match"]:
        blockers.append("source_example_lineage_mismatch")
    if not training_provenance_match:
        blockers.append("training_artifact_lineage_mismatch")
    valid = not blockers
    return {
        "status": "pass" if valid else "fail",
        "valid": valid,
        "comparison_lineage_valid": valid,
        "held_out_evaluation_allowed": valid,
        "blockers": blockers,
        "p151_report_sha256": file_sha256(p151_report_path),
        "baseline_checkpoint": rows["baseline"],
        "fingerprint_checkpoint": rows["fingerprint"],
        "cross_arm": cross_arm,
        "training_provenance_match": training_provenance_match,
    }


def _evaluate_checkpoint(
    checkpoint_path: Path,
    artifact_dir: Path,
    records: tuple[FingerprintTargetRecord, ...],
    exemplars: dict[tuple[str, int], Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint_hash_before = stable_hash(
        [
            file_sha256(checkpoint_path / name)
            for name in ("checkpoint.json", "params.npz")
        ]
    )
    load_started = time.perf_counter()
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_load_seconds = time.perf_counter() - load_started
    params_before = parameter_fingerprint(checkpoint.params)
    summary = summarize_fingerprint_artifact(artifact_dir)
    contract = VocabContract(
        tokenizer_id=summary.tokenizer_name or "fingerprint-artifact",
        tokenizer_hash=summary.tokenizer_name or None,
        vocab_size=summary.vocab_size,
        model_id=summary.teacher_model_name or None,
        model_family="behavioral_fingerprint",
    )
    backend = create_student_backend(
        vocab_contract=contract,
        architecture_id=checkpoint.manifest.student_architecture,
    )
    rows = []
    teacher_rows = []
    for record in records:
        rows.append(_evaluate_record(backend, checkpoint.params, record))
        exemplar = exemplars.get((record.example_id, record.position))
        if exemplar is not None:
            teacher_rows.append(
                _teacher_metrics(backend, checkpoint.params, record, exemplar)
            )
    jax.block_until_ready(checkpoint.params)
    params_after = parameter_fingerprint(checkpoint.params)
    checkpoint_hash_after = stable_hash(
        [
            file_sha256(checkpoint_path / name)
            for name in ("checkpoint.json", "params.npz")
        ]
    )
    if params_before != params_after or checkpoint_hash_before != checkpoint_hash_after:
        raise ValueError("evaluation mutated checkpoint parameters or files")
    elapsed = time.perf_counter() - started
    aggregate = _aggregate_corridor(rows)
    tokens = sum(len(record.input_ids) for record in records)
    aggregate.update(
        {
            "wall_clock_seconds": elapsed,
            "records_per_second": len(records) / max(elapsed, 1e-12),
            "tokens_per_second": tokens / max(elapsed, 1e-12),
            "peak_memory_if_available": None,
            "checkpoint_load_seconds": checkpoint_load_seconds,
            "parameters_unchanged": True,
        }
    )
    return {
        "records": rows,
        "record_keys": [row["record_key"] for row in rows],
        "aggregate": aggregate,
        "teacher_metrics": _mean_metrics(teacher_rows),
        "teacher_metric_availability": {
            "available": bool(teacher_rows),
            "records_with_dense_teacher_probs": len(teacher_rows),
            "records_evaluated": len(records),
            "full_coverage": len(teacher_rows) == len(records),
            "unavailable_reason": None
            if teacher_rows
            else "held-out artifact has no matching dense exemplar targets",
        },
    }


def _evaluate_record(
    backend: Any, params: Any, record: FingerprintTargetRecord
) -> dict[str, Any]:
    input_ids = jnp.asarray([record.input_ids], dtype=jnp.int32)
    position = jnp.asarray([record.position], dtype=jnp.int32)
    output, _ = backend.forward_full(params, input_ids)
    stats = compute_fingerprint_distribution_stats_at_positions(
        backend.logits(output), position
    )
    values = {
        "entropy": float(stats.entropy[0]),
        "top1_margin": float(stats.top1_margin[0]),
        "top8_mass": float(stats.top8_mass[0]),
        "top32_mass": float(stats.top32_mass[0]),
        "tail_mass": float(stats.tail_mass[0]),
    }
    bounds = {
        "entropy": (record.entropy_min, record.entropy_max),
        "top1_margin": (record.top1_margin_min, record.top1_margin_max),
        "top8_mass": (record.top8_mass_min, record.top8_mass_max),
        "top32_mass": (record.top32_mass_min, record.top32_mass_max),
        "tail_mass": (record.tail_mass_min, record.tail_mass_max),
    }
    distances = {
        name: max(lower - values[name], 0.0, values[name] - upper)
        for name, (lower, upper) in bounds.items()
    }
    inside = {name: distance == 0.0 for name, distance in distances.items()}
    return {
        "record_key": f"{record.example_id}:{record.position}",
        "example_id": record.example_id,
        "position": record.position,
        "tokens_evaluated": len(record.input_ids),
        "primary_score": sum(distance * distance for distance in distances.values()),
        "distance_outside_corridor": sum(distances.values()),
        "inside_all": all(inside.values()),
        **{f"inside_{name}": value for name, value in inside.items()},
    }


def _teacher_metrics(
    backend: Any, params: Any, record: Any, exemplar: Any
) -> dict[str, float]:
    output, _ = backend.forward_full(
        params,
        jnp.asarray([record.input_ids], dtype=jnp.int32),
    )
    logits = np.asarray(backend.logits(output)[0, record.position], dtype=np.float64)
    student = np.exp(logits - np.max(logits))
    student /= np.sum(student)
    teacher = np.asarray(exemplar.teacher_probs, dtype=np.float64)
    eps = 1e-12
    student_entropy = -float(np.sum(student * np.log(student + eps)))
    teacher_entropy = -float(np.sum(teacher * np.log(teacher + eps)))
    teacher_stats = _probability_stats(teacher)
    student_stats = _probability_stats(student)
    k = min(8, len(student))
    return {
        "teacher_student_kl": float(
            np.sum(teacher * (np.log(teacher + eps) - np.log(student + eps)))
        ),
        "teacher_student_cross_entropy": -float(
            np.sum(teacher * np.log(student + eps))
        ),
        "student_entropy": student_entropy,
        "teacher_entropy": teacher_entropy,
        "entropy_absolute_error": abs(student_entropy - teacher_entropy),
        "top1_agreement": float(np.argmax(student) == np.argmax(teacher)),
        "topk_overlap": len(
            set(np.argsort(student)[-k:]) & set(np.argsort(teacher)[-k:])
        )
        / k,
        "top1_margin_absolute_error": abs(
            student_stats["top1_margin"] - teacher_stats["top1_margin"]
        ),
        "top8_mass_absolute_error": abs(
            student_stats["top8_mass"] - teacher_stats["top8_mass"]
        ),
        "top32_mass_absolute_error": abs(
            student_stats["top32_mass"] - teacher_stats["top32_mass"]
        ),
        "tail_mass_absolute_error": abs(
            student_stats["tail_mass"] - teacher_stats["tail_mass"]
        ),
    }


def _probability_stats(probs: np.ndarray) -> dict[str, float]:
    ordered = np.sort(probs)[::-1]
    top8 = float(np.sum(ordered[: min(8, len(ordered))]))
    top32 = float(np.sum(ordered[: min(32, len(ordered))]))
    return {
        "top1_margin": float(ordered[0] - ordered[1])
        if len(ordered) > 1
        else float(ordered[0]),
        "top8_mass": top8,
        "top32_mass": top32,
        "tail_mass": max(0.0, 1.0 - top32),
    }


def _aggregate_corridor(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = np.asarray([row["primary_score"] for row in rows], dtype=np.float64)
    distances = np.asarray([row["distance_outside_corridor"] for row in rows])
    result = {
        "corridor_loss_total": float(np.mean(scores)),
        "inside_all_rate": float(np.mean([row["inside_all"] for row in rows])),
        "mean_distance_outside_corridor": float(np.mean(distances)),
        "max_distance_outside_corridor": float(np.max(distances)),
        "records_evaluated": len(rows),
        "tokens_evaluated": sum(int(row["tokens_evaluated"]) for row in rows),
        "per_record_primary_metric": scores.tolist(),
        "primary_mean": float(np.mean(scores)),
        "primary_median": float(np.median(scores)),
        "primary_standard_deviation": float(np.std(scores)),
        "primary_minimum": float(np.min(scores)),
        "primary_maximum": float(np.max(scores)),
    }
    for name in ("entropy", "top1_margin", "top8_mass", "top32_mass", "tail_mass"):
        result[f"inside_{name}_rate"] = float(
            np.mean([row[f"inside_{name}"] for row in rows])
        )
    return result


def _paired_records(
    baseline: dict[str, Any], fingerprint: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[float]]:
    rows = []
    deltas = []
    for base, finger in zip(baseline["records"], fingerprint["records"], strict=True):
        delta = float(base["primary_score"] - finger["primary_score"])
        deltas.append(delta)
        rows.append(
            {
                "record_key": base["record_key"],
                "baseline_primary_score": base["primary_score"],
                "fingerprint_primary_score": finger["primary_score"],
                "paired_delta_baseline_minus_fingerprint": delta,
                "baseline": base,
                "fingerprint": finger,
            }
        )
    return rows, deltas


def _paired_statistics(values: np.ndarray, ci95: tuple[float, float]) -> dict[str, Any]:
    tolerance = 1e-12
    return {
        "mean_paired_delta": float(np.mean(values)),
        "median_paired_delta": float(np.median(values)),
        "paired_delta_ci95": list(ci95),
        "fraction_records_won_by_baseline": float(np.mean(values < -tolerance)),
        "fraction_records_won_by_fingerprint": float(np.mean(values > tolerance)),
        "fraction_tied": float(np.mean(np.abs(values) <= tolerance)),
    }


def _comparison(
    baseline: dict[str, Any],
    fingerprint: dict[str, Any],
    paired: dict[str, Any],
    winner: str,
) -> dict[str, Any]:
    base = float(baseline["aggregate"]["corridor_loss_total"])
    finger = float(fingerprint["aggregate"]["corridor_loss_total"])
    return {
        "baseline_primary_score": base,
        "fingerprint_primary_score": finger,
        "primary_score_delta_baseline_minus_fingerprint": base - finger,
        "primary_score_relative_change": (base - finger) / max(abs(base), 1e-12),
        "baseline_inside_all_rate": baseline["aggregate"]["inside_all_rate"],
        "fingerprint_inside_all_rate": fingerprint["aggregate"]["inside_all_rate"],
        "inside_all_rate_delta": fingerprint["aggregate"]["inside_all_rate"]
        - baseline["aggregate"]["inside_all_rate"],
        "winner": winner,
        **paired,
    }


def _write_outputs(
    config: HeldOutFingerprintEvaluationConfig,
    *,
    report: dict[str, Any],
    per_records: list[dict[str, Any]],
    train_provenance: dict[str, Any],
    held_out_provenance: dict[str, Any],
) -> dict[str, Path]:
    paths = {
        "report": config.output_dir / "held_out_evaluation_report.json",
        "metrics": config.output_dir / "held_out_evaluation_metrics.json",
        "summary": config.output_dir / "held_out_evaluation_summary.md",
    }
    write_json(paths["report"], report)
    write_json(
        paths["metrics"],
        {
            "arms": report["arms"],
            "comparison": report["comparison"],
            "paired_statistics": report["paired_statistics"],
        },
    )
    write_json(
        config.output_dir / "held_out_split_validation.json", report["split_validation"]
    )
    write_json(
        config.output_dir / "checkpoint_validation.json",
        report["checkpoint_validation"],
    )
    write_json(
        config.output_dir / "provenance_manifest.json",
        {"training": train_provenance, "held_out": held_out_provenance},
    )
    write_json(
        config.output_dir / "evaluation_receipt.json",
        {
            "phase": "P152.1",
            "training_performed": False,
            "record_order_sha256": stable_hash(
                [row["record_key"] for row in per_records]
            ),
        },
    )
    _write_jsonl(config.output_dir / "per_record_metrics.jsonl", per_records)
    _write_jsonl(
        config.output_dir / "paired_deltas.jsonl",
        [
            {
                "record_key": row["record_key"],
                "baseline_primary_score": row["baseline_primary_score"],
                "fingerprint_primary_score": row["fingerprint_primary_score"],
                "paired_delta_baseline_minus_fingerprint": row[
                    "paired_delta_baseline_minus_fingerprint"
                ],
            }
            for row in per_records
        ],
    )
    paths["summary"].write_text(_render_summary(report), encoding="utf-8")
    return paths


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _render_summary(report: dict[str, Any]) -> str:
    return "\n".join(
        (
            "# P152 Held-Out Fingerprint Evaluation",
            "",
            f"- Status: {report['status']}",
            f"- Winner: {report['winner']}",
            f"- Primary metric: {report['primary_metric_name']} (lower is better)",
            f"- Baseline score: {report['comparison']['baseline_primary_score']}",
            f"- Fingerprint score: {report['comparison']['fingerprint_primary_score']}",
            "- Winner scope: held-out fingerprint primary metric only",
            "- General quality claim: false",
            "",
        )
    )


def _target_records(path: Path) -> tuple[FingerprintTargetRecord, ...]:
    records = tuple(load_fingerprint_targets(path, batch_size=1).iter_records())
    if not records:
        raise ValueError("held-out fingerprint artifact contains zero target records")
    return records


def _exemplar_map(path: Path) -> dict[tuple[str, int], Any]:
    dataset = load_fingerprint_exemplars(path, batch_size=1, require_exemplars=False)
    return {
        (record.example_id, record.position): record
        for record in dataset.iter_records()
    }


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float | None]:
    if not rows:
        return {
            name: None
            for name in (
                "teacher_student_kl",
                "teacher_student_cross_entropy",
                "student_entropy",
                "teacher_entropy",
                "entropy_absolute_error",
                "top1_agreement",
                "topk_overlap",
                "top1_margin_absolute_error",
                "top8_mass_absolute_error",
                "top32_mass_absolute_error",
                "tail_mass_absolute_error",
            )
        }
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _evaluation_metrics_finite(*arms: dict[str, Any]) -> bool:
    return all(
        math.isfinite(float(row["primary_score"]))
        and math.isfinite(float(row["distance_outside_corridor"]))
        for arm in arms
        for row in arm["records"]
    )


def _optional_json(path: Path) -> dict[str, Any]:
    return read_json_object(path) if path.is_file() else {}


def _discover_p151_report(checkpoint: Path) -> Path:
    for parent in checkpoint.parents:
        candidate = parent / "trained_baseline_comparison_report.json"
        if candidate.is_file():
            return candidate
    raise ValueError(
        "P151 comparison report was not provided and could not be discovered"
    )
