from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from qrwkv_xla.artifacts import (
    summarize_fingerprint_artifact,
)
from qrwkv_xla.artifacts._json import read_json_object, write_json
from qrwkv_xla.checkpointing import load_checkpoint
from qrwkv_xla.contracts import VocabContract
from qrwkv_xla.fingerprint.aggressiveness_profiles import (
    resolve_aggressiveness_profile,
)
from qrwkv_xla.fingerprint.corridor_measurement import (
    CorridorMeasurementConfig,
    run_corridor_measurement,
)
from qrwkv_xla.fingerprint.exemplar_pass import (
    ExemplarPassConfig,
    run_exemplar_pass,
)
from qrwkv_xla.fingerprint.held_out_evaluation import (
    _evaluate_checkpoint,
    _exemplar_map,
    _target_records,
    paired_bootstrap_interval,
    validate_fingerprint_provenance,
)
from qrwkv_xla.fingerprint.provenance import (
    file_sha256,
    hash_checkpoint_bundle,
    parameter_fingerprint,
    stable_hash,
)
from qrwkv_xla.fingerprint.trained_baseline import (
    FingerprintTrainedBaselineConfig,
    run_fingerprint_trained_baseline_comparison,
)
from qrwkv_xla.students import create_student_backend
from qrwkv_xla.tracking import get_git_metadata

ARM_NAMES = (
    "conventional_baseline",
    "corridor_only",
    "exemplar_only",
    "two_cycle",
)


@dataclass(frozen=True)
class TwoCycleExperimentConfig:
    training_fingerprint_artifact: Path
    calibration_fingerprint_artifact: Path
    final_test_fingerprint_artifact: Path
    source_texts: Path
    selected_profile_receipt: Path
    output_dir: Path
    student_backend: str = "current_qrwkv"
    student_architecture: str | None = None
    baseline_steps: int = 3
    corridor_steps: int = 3
    exemplar_steps: int = 3
    corridor_only_steps: int | None = None
    exemplar_only_steps: int | None = None
    budget_comparison_mode: str = "stage_matched"
    batch_size: int = 1
    optimizer: str = "adamw"
    baseline_learning_rate: float = 1e-4
    exemplar_learning_rate: float = 5e-5
    exemplar_max_grad_norm: float | None = 1.0
    corridor_eval_every: int | None = 1
    exemplar_eval_every: int | None = 1
    checkpoint_every: int = 3
    exemplar_sampling_policy: str = "sequential"
    exemplar_max_records: int | None = None
    seed: int = 0
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 0
    tie_tolerance: float = 1e-12
    overwrite: bool = False


@dataclass(frozen=True)
class TwoCycleExperimentResult:
    status: str
    primary_result: str | None
    output_dir: Path
    report_path: Path


def run_two_cycle_experiment(
    config: TwoCycleExperimentConfig,
) -> TwoCycleExperimentResult:
    _validate_config(config)
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    split_validation = validate_three_way_split(config)
    write_json(config.output_dir / "three_way_split_validation.json", split_validation)
    selection, selected_config = _load_selected_profile(config)
    profile_name = str(selection["selected_profile"])
    resolve_aggressiveness_profile(profile_name)
    expected_student_config_sha256 = _expected_student_config_sha256(config)
    calibration_lineage = _validate_calibration_lineage(
        config,
        selection=selection,
        selected_config=selected_config,
        split_validation=split_validation,
        expected_student_config_sha256=expected_student_config_sha256,
    )
    write_json(
        config.output_dir / "calibration_lineage_validation.json",
        calibration_lineage,
    )

    baseline_result = run_fingerprint_trained_baseline_comparison(
        FingerprintTrainedBaselineConfig(
            fingerprint_artifact=config.training_fingerprint_artifact,
            source_texts=config.source_texts,
            output_dir=config.output_dir / "arms" / "conventional_baseline",
            steps=config.baseline_steps,
            batch_size=config.batch_size,
            optimizer=config.optimizer,
            learning_rate=config.baseline_learning_rate,
            seed=config.seed,
            student_backend=config.student_backend,
            overwrite=config.overwrite,
        )
    )
    p151 = read_json_object(baseline_result.report_path)
    source_initial = Path(p151["shared_initial_checkpoint_path"])
    shared_checkpoint = (
        config.output_dir
        / "shared_initialization_checkpoint"
        / "checkpoints"
        / "initial"
    )
    shared_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if shared_checkpoint.exists() and config.overwrite:
        shutil.rmtree(shared_checkpoint)
    shutil.copytree(source_initial, shared_checkpoint)
    shared_loaded = load_checkpoint(shared_checkpoint)
    shared_hashes = hash_checkpoint_bundle(shared_checkpoint)
    shared_fingerprint = parameter_fingerprint(shared_loaded.params)
    shared_valid = bool(
        shared_fingerprint == p151["shared_initial_parameter_fingerprint"]
        and shared_hashes["checkpoint_bundle_sha256"]
        == hash_checkpoint_bundle(source_initial)["checkpoint_bundle_sha256"]
    )
    if not shared_valid:
        raise ValueError("shared_initialization_mismatch")
    shared_receipt = {
        "status": "pass",
        "initialization_seed": config.seed,
        "student_backend": config.student_backend,
        "student_config_sha256": stable_hash(shared_loaded.manifest.student_config),
        "parameter_fingerprint": shared_fingerprint,
        "source_checkpoint": str(source_initial),
        "materialized_checkpoint": str(shared_checkpoint),
        "exact_identity_verified": True,
        **shared_hashes,
    }
    write_json(config.output_dir / "shared_initialization_receipt.json", shared_receipt)

    corridor_only_common = _corridor_config_fields(
        config,
        selected_config,
        steps=config.corridor_only_steps or config.corridor_steps,
    )
    two_cycle_corridor_common = _corridor_config_fields(
        config, selected_config, steps=config.corridor_steps
    )
    corridor_only = run_corridor_measurement(
        CorridorMeasurementConfig(
            output_dir=config.output_dir / "arms" / "corridor_only",
            initial_checkpoint=shared_checkpoint,
            p151_report=baseline_result.report_path,
            selected_aggressiveness_profile=profile_name,
            selected_profile_config_sha256=selection["selected_profile_config_sha256"],
            **corridor_only_common,
        )
    )
    two_cycle_corridor = run_corridor_measurement(
        CorridorMeasurementConfig(
            output_dir=config.output_dir / "arms" / "two_cycle" / "corridor",
            initial_checkpoint=shared_checkpoint,
            p151_report=baseline_result.report_path,
            selected_aggressiveness_profile=profile_name,
            selected_profile_config_sha256=selection["selected_profile_config_sha256"],
            **two_cycle_corridor_common,
        )
    )
    corridor_only_report = read_json_object(corridor_only.report_path)
    two_cycle_corridor_report = read_json_object(two_cycle_corridor.report_path)
    corridor_stage_configs_match = (
        corridor_only_report["corridor_aggressiveness"]
        == two_cycle_corridor_report["corridor_aggressiveness"]
    )
    corridor_configs_match = corridor_stage_configs_match and (
        config.budget_comparison_mode == "equal_total_budget"
        or (
            corridor_only_report["requested_steps"]
            == two_cycle_corridor_report["requested_steps"]
            and parameter_fingerprint(
                load_checkpoint(corridor_only.checkpoint_dir).params
            )
            == parameter_fingerprint(
                load_checkpoint(two_cycle_corridor.checkpoint_dir).params
            )
        )
    )
    if not corridor_configs_match:
        raise ValueError("corridor_arm_configuration_mismatch")

    derived_selection = _write_derived_selection_receipt(
        config,
        selection=selection,
        selected_config=selected_config,
        corridor_checkpoint=two_cycle_corridor.checkpoint_dir,
    )
    exemplar_only_common = _exemplar_config_fields(
        config, steps=config.exemplar_only_steps or config.exemplar_steps
    )
    two_cycle_exemplar_common = _exemplar_config_fields(
        config, steps=config.exemplar_steps
    )
    exemplar_only = run_exemplar_pass(
        ExemplarPassConfig(
            corridor_checkpoint=shared_checkpoint,
            output_dir=config.output_dir / "arms" / "exemplar_only",
            allow_shared_initialization_parent_for_control=True,
            **exemplar_only_common,
        )
    )
    two_cycle_exemplar = run_exemplar_pass(
        ExemplarPassConfig(
            corridor_checkpoint=two_cycle_corridor.checkpoint_dir,
            output_dir=config.output_dir / "arms" / "two_cycle" / "exemplar",
            p153_report=two_cycle_corridor.report_path,
            selected_profile=derived_selection,
            **two_cycle_exemplar_common,
        )
    )
    exemplar_only_report = read_json_object(exemplar_only.report_path)
    two_cycle_exemplar_report = read_json_object(two_cycle_exemplar.report_path)
    exemplar_only_view = _exemplar_fairness_view(
        exemplar_only_report, exemplar_only.output_dir
    )
    two_cycle_exemplar_view = _exemplar_fairness_view(
        two_cycle_exemplar_report, two_cycle_exemplar.output_dir
    )
    exemplar_configs_match = _without_requested_steps(
        exemplar_only_view
    ) == _without_requested_steps(two_cycle_exemplar_view) and (
        config.budget_comparison_mode == "equal_total_budget"
        or exemplar_only_view == two_cycle_exemplar_view
    )
    if not exemplar_configs_match:
        raise ValueError("exemplar_arm_configuration_mismatch")
    all_arm_initialization_valid = _all_arm_initialization_valid(
        shared_fingerprint=shared_fingerprint,
        shared_hash=shared_hashes["checkpoint_bundle_sha256"],
        p151=p151,
        corridor_only_report=corridor_only_report,
        two_cycle_corridor_report=two_cycle_corridor_report,
        exemplar_only=exemplar_only,
    )
    if not all_arm_initialization_valid:
        raise ValueError("arm_initialization_mismatch")

    cycle_boundary = _cycle_boundary_receipt(
        two_cycle_corridor,
        two_cycle_corridor_report,
        two_cycle_exemplar,
        two_cycle_exemplar_report,
    )
    if not cycle_boundary["cycle_boundary_valid"]:
        raise ValueError("cycle_boundary_invalid")
    write_json(config.output_dir / "cycle_boundary_receipt.json", cycle_boundary)

    checkpoints = {
        "shared_initialization": shared_checkpoint,
        "conventional_baseline": Path(p151["baseline"]["checkpoint_dir"]),
        "corridor_only": corridor_only.checkpoint_dir,
        "exemplar_only": exemplar_only.final_checkpoint,
        "two_cycle_corridor": two_cycle_corridor.checkpoint_dir,
        "two_cycle_final": two_cycle_exemplar.final_checkpoint,
    }
    freeze = build_configuration_freeze(
        config,
        shared_initialization_hash=shared_hashes["checkpoint_bundle_sha256"],
        selected_profile=profile_name,
        selected_profile_config_sha256=selection["selected_profile_config_sha256"],
        exemplar_sampling_receipt=read_json_object(
            exemplar_only.output_dir / "sampling_receipt.json"
        ),
        calibration_artifact_sha256=split_validation["calibration"][
            "artifact_manifest_sha256"
        ],
        training_artifact_sha256=split_validation["training"][
            "artifact_manifest_sha256"
        ],
    )
    freeze_path = config.output_dir / "experiment_configuration_freeze.json"
    write_json(freeze_path, freeze)
    final_test_access = _final_test_access_receipt(
        config,
        freeze_path=freeze_path,
        freeze=freeze,
        split_validation=split_validation,
    )
    write_json(config.output_dir / "final_test_access_receipt.json", final_test_access)
    evaluations = _evaluate_all(config, checkpoints)
    evaluation_metrics = _evaluation_metrics(evaluations)
    per_record_rows = _aligned_per_record_metrics(evaluations)
    comparisons = _paired_comparisons(config, per_record_rows)
    primary = comparisons["two_cycle_final_vs_exemplar_only_final"]
    resources = _resource_accounting(
        config,
        p151=p151,
        corridor_only=corridor_only_report,
        exemplar_only=exemplar_only,
        two_cycle_corridor=two_cycle_corridor_report,
        two_cycle_exemplar=two_cycle_exemplar,
    )
    fairness = {
        "status": "pass",
        "shared_initialization_valid": shared_valid,
        "all_arms_share_initialization": all_arm_initialization_valid,
        "corridor_configs_match": corridor_configs_match,
        "corridor_stage_configs_match": corridor_stage_configs_match,
        "exemplar_configs_match": exemplar_configs_match,
        "budget_comparison_mode": config.budget_comparison_mode,
        "held_out_record_order_identical": True,
        "selected_profile": profile_name,
        "selected_profile_config_sha256": selection["selected_profile_config_sha256"],
        "calibration_report_sha256": _required_sibling_hash(
            config.selected_profile_receipt,
            "aggressiveness_calibration_report.json",
        ),
        "selection_receipt_sha256": file_sha256(config.selected_profile_receipt),
        "publication_grade_calibration": _publication_grade_calibration(
            config.selected_profile_receipt
        ),
        "primary_metric": "held_out_teacher_student_kl",
        "primary_metric_direction": "lower_is_better",
        "secondary_metrics": [
            "held_out_corridor_loss",
            "inside_all_rate",
            "mean_distance_outside_corridor",
        ],
        "tie_tolerance": config.tie_tolerance,
        "bootstrap_samples": config.bootstrap_samples,
        "bootstrap_seed": config.bootstrap_seed,
        "three_way_split_valid": split_validation["three_way_split_valid"],
        "calibration_lineage_valid": calibration_lineage["calibration_lineage_valid"],
        "final_test_independent": True,
        "evaluation_artifact_role": "final_held_out_test",
        "evaluation_artifact_sha256": split_validation["final_test"][
            "artifact_manifest_sha256"
        ],
        "calibration_artifact_sha256": split_validation["calibration"][
            "artifact_manifest_sha256"
        ],
        "evaluation_calibration_artifacts_distinct": True,
    }
    write_json(config.output_dir / "experiment_fairness_contract.json", fairness)
    write_json(config.output_dir / "paired_comparison_metrics.json", comparisons)
    write_json(config.output_dir / "resource_accounting.json", resources)
    _write_jsonl(config.output_dir / "per_record_arm_metrics.jsonl", per_record_rows)
    arm_lineage = _arm_lineage(config, checkpoints, evaluations, resources)
    status = (
        "pass"
        if all(
            (
                baseline_result.status == "pass",
                corridor_only.status == "pass",
                exemplar_only.status == "pass",
                two_cycle_corridor.status == "pass",
                two_cycle_exemplar.status == "pass",
                shared_valid,
                all_arm_initialization_valid,
                corridor_configs_match,
                exemplar_configs_match,
                cycle_boundary["cycle_boundary_valid"],
                split_validation["three_way_split_valid"],
                calibration_lineage["calibration_lineage_valid"],
                final_test_access["final_test_access_valid"],
            )
        )
        else "fail"
    )
    report = {
        "phase": "P155.1",
        "status": status,
        "experiment_kind": "sequential_corridor_then_exemplar",
        "arms": list(ARM_NAMES),
        "shared_initialization_valid": shared_valid,
        "all_arms_share_initialization": all_arm_initialization_valid,
        "cycle_boundary_valid": cycle_boundary["cycle_boundary_valid"],
        "mixed_objective_enabled": False,
        "fresh_exemplar_optimizer_state": True,
        "primary_comparison": "two_cycle_final_vs_exemplar_only_final",
        "primary_metric": "held_out_teacher_student_kl",
        "primary_metric_direction": "lower_is_better",
        "primary_result": _normalize_primary_result(primary["result"])
        if status == "pass"
        else None,
        "winner_declared": status == "pass" and primary["result"] != "inconclusive",
        "fairness": fairness,
        "cycle_boundary": cycle_boundary,
        "comparisons": comparisons,
        "resources": resources,
        "evaluation_metrics": evaluation_metrics,
        "arm_lineage": arm_lineage,
        "split_integrity": {
            "training_artifact_sha256": split_validation["training"][
                "artifact_manifest_sha256"
            ],
            "calibration_artifact_sha256": split_validation["calibration"][
                "artifact_manifest_sha256"
            ],
            "final_test_artifact_sha256": split_validation["final_test"][
                "artifact_manifest_sha256"
            ],
            "three_way_split_valid": split_validation["three_way_split_valid"],
            "final_test_independent": True,
            "configuration_frozen_before_final_test_access": final_test_access[
                "configuration_frozen_before_final_test_access"
            ],
        },
        "evaluation_artifact_role": "final_held_out_test",
        "evaluation_artifact_sha256": split_validation["final_test"][
            "artifact_manifest_sha256"
        ],
        "calibration_artifact_sha256": split_validation["calibration"][
            "artifact_manifest_sha256"
        ],
        "evaluation_calibration_artifacts_distinct": True,
        "final_test_unseen_by_calibration": True,
        "final_test_used_for_profile_selection": False,
        "final_test_used_for_hyperparameter_selection": False,
        "final_test_opened_after_configuration_freeze": True,
        "final_test_independent": True,
        "result_scope": "independent_final_test",
        "publication_grade_final_test": bool(
            status == "pass"
            and fairness["publication_grade_calibration"] is True
            and split_validation["three_way_split_valid"]
        ),
        "total_wall_clock_seconds": time.perf_counter() - started,
        "general_quality_claim_made": False,
        "quality_per_byte_claim_made": False,
        "scale_claim_made": False,
        "radlads_parity_claim_made": False,
    }
    report_path = config.output_dir / "two_cycle_experiment_report.json"
    write_json(report_path, report)
    (config.output_dir / "two_cycle_experiment_summary.md").write_text(
        "# P155.1 Sequential Two-Cycle Split Integrity\n\n"
        f"- Status: {status}\n"
        f"- Selected corridor profile: {profile_name}\n"
        f"- Primary metric: held_out_teacher_student_kl (lower is better)\n"
        f"- Primary result: {report['primary_result']}\n"
        "- Cycle boundary valid: "
        f"{str(cycle_boundary['cycle_boundary_valid']).lower()}\n"
        "- Mixed objective enabled: false\n"
        "- Quality-per-byte claim: false\n",
        encoding="utf-8",
    )
    return TwoCycleExperimentResult(
        status=status,
        primary_result=report["primary_result"],
        output_dir=config.output_dir,
        report_path=report_path,
    )


def paired_lower_is_better_comparison(
    left_scores: dict[str, float],
    right_scores: dict[str, float],
    *,
    left_name: str,
    right_name: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    tie_tolerance: float,
) -> dict[str, Any]:
    if set(left_scores) != set(right_scores) or not left_scores:
        raise ValueError("paired comparison requires identical aligned record keys")
    keys = sorted(left_scores)
    deltas = np.asarray(
        [left_scores[key] - right_scores[key] for key in keys], dtype=np.float64
    )
    if not np.all(np.isfinite(deltas)):
        raise ValueError("paired comparison contains non-finite metrics")
    ci95 = paired_bootstrap_interval(
        deltas, samples=bootstrap_samples, seed=bootstrap_seed
    )
    if ci95[1] < -tie_tolerance:
        result = f"{left_name}_better"
    elif ci95[0] > tie_tolerance:
        result = f"{right_name}_better"
    else:
        result = "inconclusive"
    return {
        "left_arm": left_name,
        "right_arm": right_name,
        "metric": "held_out_teacher_student_kl",
        "direction": "lower_is_better",
        "aligned_record_count": len(keys),
        "record_order_sha256": stable_hash(keys),
        "mean_paired_delta_left_minus_right": float(np.mean(deltas)),
        "median_paired_delta_left_minus_right": float(np.median(deltas)),
        "paired_delta_ci95": list(ci95),
        "fraction_left_won": float(np.mean(deltas < -tie_tolerance)),
        "fraction_tied": float(np.mean(np.abs(deltas) <= tie_tolerance)),
        "fraction_left_lost": float(np.mean(deltas > tie_tolerance)),
        "result": result,
    }


def sum_stage_resources(*stages: dict[str, float | int]) -> dict[str, float | int]:
    keys = (
        "optimizer_steps",
        "records_consumed",
        "tokens_consumed",
        "artifact_bytes_logically_consumed",
        "training_seconds",
        "evaluation_seconds",
        "checkpoint_seconds",
        "total_wall_clock_seconds",
    )
    return {key: sum(stage[key] for stage in stages) for key in keys}


def required_arms_present(arms: dict[str, Any]) -> bool:
    return set(arms) >= set(ARM_NAMES)


def validate_three_way_split(config: TwoCycleExperimentConfig) -> dict[str, Any]:
    paths = {
        "training": config.training_fingerprint_artifact.resolve(),
        "calibration": config.calibration_fingerprint_artifact.resolve(),
        "final_test": config.final_test_fingerprint_artifact.resolve(),
    }
    if paths["training"] == paths["calibration"]:
        raise ValueError("training_calibration_split_overlap")
    if paths["training"] == paths["final_test"]:
        raise ValueError("training_final_test_split_overlap")
    if paths["calibration"] == paths["final_test"]:
        raise ValueError("calibration_final_test_split_overlap")
    specifications = (
        ("training", config.training_fingerprint_artifact, "training"),
        (
            "calibration",
            config.calibration_fingerprint_artifact,
            "calibration_validation",
        ),
        (
            "final_test",
            config.final_test_fingerprint_artifact,
            "final_held_out_test",
        ),
    )
    rows: dict[str, dict[str, Any]] = {}
    sets: dict[str, dict[str, set[str]]] = {}
    for name, path, role in specifications:
        if name == "final_test":
            provenance = _validate_final_test_provenance_metadata(path, role)
        else:
            validated = validate_fingerprint_provenance(path, expected_role=role)
            if not validated["valid"]:
                raise ValueError(f"missing_or_invalid_{name}_split_provenance")
            provenance = validated["provenance"]
        manifest = read_json_object(path / "manifest.json")
        rows[name] = {
            "artifact_dir": str(path),
            "artifact_role": role,
            "artifact_manifest_sha256": provenance["artifact_manifest_sha256"],
            "source_file_sha256": provenance["source_file_sha256"],
            "ordered_example_ids_sha256": provenance["ordered_example_ids_sha256"],
            "source_example_set_sha256": provenance["source_example_set_sha256"],
            "ordered_source_text_sha256": provenance["ordered_source_text_sha256"],
            "tokenized_inputs_sha256": provenance["tokenized_inputs_sha256"],
            "teacher_identity_sha256": provenance["teacher_identity_sha256"],
            "capture_config_sha256": provenance["capture_config_sha256"],
            "tokenizer_identity": provenance["tokenizer_identity"],
            "vocab_size": int(manifest["teacher"]["vocab_size"]),
            "tracked_statistics": list(manifest["stats"]["tracked"]),
            "sequence_length_policy": {
                "kind": "fixed",
                "max_seq_len": int(manifest["sequence"]["max_seq_len"]),
            },
        }
        sets[name] = {
            "ids": set(provenance["ordered_example_ids"]),
            "tokens": set(provenance["token_sequence_hashes"]),
            "texts": set(provenance["source_text_hashes"]),
        }
    pair_specs = (
        ("training", "calibration", "training_calibration_split_overlap"),
        ("training", "final_test", "training_final_test_split_overlap"),
        (
            "calibration",
            "final_test",
            "calibration_final_test_split_overlap",
        ),
    )
    disjointness: dict[str, bool] = {}
    for left, right, error in pair_specs:
        pair_name = f"{left}_vs_{right}"
        overlaps = {
            kind: sets[left][kind] & sets[right][kind]
            for kind in ("ids", "tokens", "texts")
        }
        for kind, overlap in overlaps.items():
            disjointness[f"{pair_name}_{kind}"] = not overlap
        if any(overlaps.values()):
            raise ValueError(error)
    return {
        "status": "pass",
        **rows,
        "pairwise_disjointness": disjointness,
        "three_way_split_valid": all(disjointness.values()),
        "final_test_unseen_by_calibration": True,
    }


def _validate_final_test_provenance_metadata(path, expected_role):
    provenance_path = path / "fingerprint_provenance.json"
    provenance = read_json_object(provenance_path)
    required = (
        "artifact_manifest_sha256",
        "source_file_sha256",
        "ordered_example_ids_sha256",
        "source_example_set_sha256",
        "ordered_source_text_sha256",
        "tokenized_inputs_sha256",
        "teacher_identity_sha256",
        "capture_config_sha256",
        "ordered_example_ids",
        "token_sequence_hashes",
        "source_text_hashes",
    )
    valid = bool(
        provenance.get("artifact_role") == expected_role
        and all(provenance.get(key) for key in required)
        and provenance.get("artifact_manifest_sha256")
        == file_sha256(path / "manifest.json")
        and provenance.get("source_file_sha256")
        == file_sha256(Path(provenance["source_file"]))
        and provenance.get("split_manifest_sha256")
        == stable_hash(
            {
                "artifact_role": provenance.get("artifact_role"),
                "ordered_example_ids": provenance.get("ordered_example_ids"),
                "source_text_hashes": provenance.get("source_text_hashes"),
                "token_sequence_hashes": provenance.get("token_sequence_hashes"),
            }
        )
    )
    if not valid:
        raise ValueError("missing_or_invalid_final_test_split_provenance")
    return provenance


def build_configuration_freeze(
    config: TwoCycleExperimentConfig,
    *,
    shared_initialization_hash: str,
    selected_profile: str,
    selected_profile_config_sha256: str,
    exemplar_sampling_receipt: dict[str, Any],
    calibration_artifact_sha256: str,
    training_artifact_sha256: str,
) -> dict[str, Any]:
    frozen = {
        "student_architecture": config.student_architecture,
        "student_backend": config.student_backend,
        "shared_initialization_hash": shared_initialization_hash,
        "selected_corridor_profile": selected_profile,
        "selected_corridor_config_sha256": selected_profile_config_sha256,
        "corridor_step_budget": config.corridor_steps,
        "exemplar_step_budget": config.exemplar_steps,
        "baseline_step_budget": config.baseline_steps,
        "corridor_only_step_budget": config.corridor_only_steps,
        "exemplar_only_step_budget": config.exemplar_only_steps,
        "budget_comparison_mode": config.budget_comparison_mode,
        "batch_size": config.batch_size,
        "initialization_and_sampling_seed": config.seed,
        "corridor_eval_every": config.corridor_eval_every,
        "exemplar_eval_every": config.exemplar_eval_every,
        "checkpoint_every": config.checkpoint_every,
        "optimizer_configuration": {
            "type": config.optimizer,
            "baseline_learning_rate": config.baseline_learning_rate,
            "exemplar_learning_rate": config.exemplar_learning_rate,
            "exemplar_max_grad_norm": config.exemplar_max_grad_norm,
        },
        "exemplar_sampling_policy": config.exemplar_sampling_policy,
        "exemplar_max_records": config.exemplar_max_records,
        "exemplar_records_selected": exemplar_sampling_receipt["records_selected"],
        "exemplar_record_order_sha256": exemplar_sampling_receipt[
            "record_order_sha256"
        ],
        "primary_metric": "held_out_teacher_student_kl",
        "secondary_metrics": [
            "held_out_corridor_loss",
            "inside_all_rate",
            "mean_distance_outside_corridor",
        ],
        "bootstrap_samples": config.bootstrap_samples,
        "bootstrap_seed": config.bootstrap_seed,
        "tie_tolerance": config.tie_tolerance,
        "training_artifact_sha256": training_artifact_sha256,
        "calibration_artifact_sha256": calibration_artifact_sha256,
        "software_commit": get_git_metadata(Path(__file__).resolve().parents[3]).get(
            "commit"
        ),
    }
    return {
        "phase": "P155.1",
        "status": "frozen",
        "freeze_timestamp": datetime.now(UTC).isoformat(),
        "frozen_configuration": frozen,
        "configuration_freeze_sha256": stable_hash(frozen),
        "final_test_artifact_influenced_freeze": False,
    }


def _load_selected_profile(config):
    receipt = read_json_object(config.selected_profile_receipt)
    selected = receipt.get("selected_profile")
    if not (
        receipt.get("status") == "pass"
        and receipt.get("selection_allowed") is True
        and receipt.get("winner_declared") is True
        and selected
        and receipt.get("selected_profile_config_sha256")
    ):
        raise ValueError("selected_profile_receipt_invalid")
    config_path = config.selected_profile_receipt.with_name(
        "selected_profile_config.json"
    )
    if not config_path.is_file():
        raise ValueError("selected_profile_config_missing")
    calibration_report = config.selected_profile_receipt.with_name(
        "aggressiveness_calibration_report.json"
    )
    publication_receipt = config.selected_profile_receipt.with_name(
        "publication_grade_receipt.json"
    )
    if not calibration_report.is_file() or not publication_receipt.is_file():
        raise ValueError("calibration_evidence_missing")
    calibration = read_json_object(calibration_report)
    if (
        calibration.get("status") != "pass"
        or calibration.get("selected_profile") != selected
    ):
        raise ValueError("calibration_report_mismatch")
    selected_config = read_json_object(config_path)
    if (
        selected_config.get("profile_name") != selected
        or stable_hash(selected_config) != receipt["selected_profile_config_sha256"]
    ):
        raise ValueError("selected_profile_config_mismatch")
    return receipt, selected_config


def _expected_student_config_sha256(config):
    artifact = summarize_fingerprint_artifact(config.training_fingerprint_artifact)
    contract = VocabContract(
        tokenizer_id=artifact.tokenizer_name or "fingerprint-artifact",
        tokenizer_hash=artifact.tokenizer_name or None,
        vocab_size=artifact.vocab_size,
        model_id=artifact.teacher_model_name or None,
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
        "vocab_size": int(getattr(raw, "vocab_size", artifact.vocab_size)),
        "hidden_size": int(getattr(raw, "hidden_size", 0)),
        "num_layers": int(getattr(raw, "num_layers", 0)),
        "num_heads": None
        if getattr(raw, "num_heads", None) is None
        else int(raw.num_heads),
        "num_kv_heads": None
        if getattr(raw, "num_kv_heads", None) is None
        else int(raw.num_kv_heads),
        "emit_logits": bool(getattr(raw, "emit_logits", True)),
        "tie_embeddings": bool(getattr(raw, "tie_embeddings", False)),
        "emit_mixer_outputs": bool(getattr(raw, "emit_mixer_outputs", False)),
    }
    return stable_hash(student_config)


def _validate_calibration_lineage(
    config,
    *,
    selection,
    selected_config,
    split_validation,
    expected_student_config_sha256,
):
    required = (
        "calibration_training_artifact_sha256",
        "calibration_validation_artifact_sha256",
        "calibration_student_config_sha256",
        "selected_profile_config_sha256",
        "calibration_report_sha256",
        "publication_grade_receipt_sha256",
    )
    if any(not selection.get(key) for key in required):
        raise ValueError("calibration_artifact_hashes_missing")
    if (
        selection["calibration_training_artifact_sha256"]
        != split_validation["training"]["artifact_manifest_sha256"]
    ):
        raise ValueError("calibration_training_artifact_mismatch")
    if (
        selection["calibration_validation_artifact_sha256"]
        != split_validation["calibration"]["artifact_manifest_sha256"]
    ):
        raise ValueError("calibration_validation_artifact_mismatch")
    if selection["calibration_student_config_sha256"] != expected_student_config_sha256:
        raise ValueError("calibration_student_config_mismatch")
    if selection["selected_profile_config_sha256"] != stable_hash(selected_config):
        raise ValueError("selected_profile_config_mismatch")
    calibration_report_path = config.selected_profile_receipt.with_name(
        "aggressiveness_calibration_report.json"
    )
    publication_receipt_path = config.selected_profile_receipt.with_name(
        "publication_grade_receipt.json"
    )
    if selection["calibration_report_sha256"] != file_sha256(
        calibration_report_path
    ) or selection["publication_grade_receipt_sha256"] != file_sha256(
        publication_receipt_path
    ):
        raise ValueError("calibration_report_lineage_mismatch")
    return {
        "status": "pass",
        "calibration_lineage_valid": True,
        "calibration_training_artifact_sha256": selection[
            "calibration_training_artifact_sha256"
        ],
        "calibration_validation_artifact_sha256": selection[
            "calibration_validation_artifact_sha256"
        ],
        "calibration_student_config_sha256": selection[
            "calibration_student_config_sha256"
        ],
        "selected_profile_name": selection["selected_profile"],
        "selected_profile_config_sha256": stable_hash(selected_config),
        "calibration_report_sha256": selection["calibration_report_sha256"],
        "selection_receipt_sha256": file_sha256(config.selected_profile_receipt),
        "publication_grade_receipt_sha256": selection[
            "publication_grade_receipt_sha256"
        ],
    }


def _corridor_config_fields(config, profile, *, steps):
    return {
        "fingerprint_artifact": config.training_fingerprint_artifact,
        "held_out_fingerprint_artifact": config.calibration_fingerprint_artifact,
        "held_out_artifact_role": "calibration_validation",
        "source_texts": config.source_texts,
        "steps": steps,
        "eval_every": config.corridor_eval_every,
        "checkpoint_every": config.checkpoint_every,
        "batch_size": config.batch_size,
        "optimizer": config.optimizer,
        "learning_rate": profile["learning_rate"],
        "max_grad_norm": profile["max_grad_norm"],
        "corridor_loss_weight": profile["corridor_loss_weight"],
        "penalty_kind": profile["penalty_kind"],
        "penalty_power": profile["penalty_power"],
        "entropy_weight": profile["per_stat_weights"]["entropy"],
        "top1_margin_weight": profile["per_stat_weights"]["top1_margin"],
        "top8_mass_weight": profile["per_stat_weights"]["top8_mass"],
        "top32_mass_weight": profile["per_stat_weights"]["top32_mass"],
        "tail_mass_weight": profile["per_stat_weights"]["tail_mass"],
        "worst_stat_boost": profile["worst_stat_boost"],
        "distance_normalization": profile["distance_normalization"],
        "stability_abort_enabled": profile["stability_abort_enabled"],
        "parameter_norm_limit": profile["parameter_norm_limit"],
        "gradient_norm_hard_limit": profile["gradient_norm_hard_limit"],
        "held_out_loss_abort_multiple": profile["held_out_loss_abort_multiple"],
        "seed": config.seed,
        "student_backend": config.student_backend,
        "stop_on_stable_entry": False,
        "overwrite": config.overwrite,
    }


def _exemplar_config_fields(config, *, steps):
    return {
        "fingerprint_artifact": config.training_fingerprint_artifact,
        "held_out_fingerprint_artifact": config.calibration_fingerprint_artifact,
        "student_backend": config.student_backend,
        "student_architecture": config.student_architecture,
        "steps": steps,
        "batch_size": config.batch_size,
        "optimizer": config.optimizer,
        "learning_rate": config.exemplar_learning_rate,
        "max_grad_norm": config.exemplar_max_grad_norm,
        "seed": config.seed,
        "checkpoint_every": config.checkpoint_every,
        "eval_every": config.exemplar_eval_every,
        "exemplar_max_records": config.exemplar_max_records,
        "exemplar_sampling_policy": config.exemplar_sampling_policy,
        "overwrite": config.overwrite,
    }


def _write_derived_selection_receipt(
    config, *, selection, selected_config, corridor_checkpoint
):
    path = config.output_dir / "arms" / "two_cycle" / "corridor_selection_binding.json"
    loaded = load_checkpoint(corridor_checkpoint)
    payload = {
        "status": "pass",
        "selection_allowed": True,
        "winner_declared": True,
        "selected_profile": selection["selected_profile"],
        "selected_profile_config_sha256": stable_hash(selected_config),
        "selected_corridor_checkpoint_bundle_sha256": hash_checkpoint_bundle(
            corridor_checkpoint
        )["checkpoint_bundle_sha256"],
        "selected_corridor_parameter_fingerprint": parameter_fingerprint(loaded.params),
        "source_selection_receipt_sha256": file_sha256(config.selected_profile_receipt),
    }
    write_json(path, payload)
    return path


def _exemplar_fairness_view(report, output_dir):
    sampling = read_json_object(output_dir / "sampling_receipt.json")
    return {
        "requested_steps": report["requested_steps"],
        "exemplar_loss_type": report["exemplar_loss_type"],
        "exemplar_loss_weight": report["exemplar_loss_weight"],
        "corridor_loss_enabled": report["corridor_loss_enabled"],
        "mixed_objective_enabled": report["mixed_objective_enabled"],
        "batch_size": report["batch_size"],
        "optimizer": report["optimizer"],
        "learning_rate": report["learning_rate"],
        "max_grad_norm": report["max_grad_norm"],
        "evaluation_interval_steps": report["evaluation_interval_steps"],
        "checkpoint_interval_steps": report["checkpoint_interval_steps"],
        "sampling_policy": sampling["sampling_policy"],
        "sampling_seed": sampling["sampling_seed"],
        "record_order_sha256": sampling["record_order_sha256"],
        "records_selected": sampling["records_selected"],
        "exemplar_max_records": sampling["exemplar_max_records"],
        "exemplar_artifact_sha256": sampling["exemplar_artifact_sha256"],
    }


def _without_requested_steps(view):
    return {key: value for key, value in view.items() if key != "requested_steps"}


def _all_arm_initialization_valid(
    *,
    shared_fingerprint,
    shared_hash,
    p151,
    corridor_only_report,
    two_cycle_corridor_report,
    exemplar_only,
):
    exemplar_manifest = load_checkpoint(exemplar_only.final_checkpoint).manifest
    fingerprints = (
        p151["baseline"]["initial_parameter_fingerprint"],
        corridor_only_report["lineage"]["initialization"]["parameter_fingerprint"],
        two_cycle_corridor_report["lineage"]["initialization"]["parameter_fingerprint"],
    )
    return bool(
        all(value == shared_fingerprint for value in fingerprints)
        and exemplar_manifest.target_manifest.get("parent_checkpoint_bundle_sha256")
        == shared_hash
    )


def _cycle_boundary_receipt(
    corridor_result, corridor_report, exemplar_result, exemplar_report
):
    corridor_hashes = hash_checkpoint_bundle(corridor_result.checkpoint_dir)
    corridor_fingerprint = parameter_fingerprint(
        load_checkpoint(corridor_result.checkpoint_dir).params
    )
    exemplar_manifest = load_checkpoint(exemplar_result.final_checkpoint).manifest
    checks = {
        "corridor_process_completed": corridor_result.status == "pass",
        "corridor_checkpoint_written": corridor_result.checkpoint_dir.is_dir(),
        "exemplar_process_started_after_corridor_completion": True,
        "exemplar_parent_checkpoint_match": exemplar_manifest.target_manifest.get(
            "parent_checkpoint_bundle_sha256"
        )
        == corridor_hashes["checkpoint_bundle_sha256"],
        "corridor_optimizer_state_loaded_by_exemplar": exemplar_report[
            "input_checkpoint_optimizer_state_loaded"
        ],
        "fresh_exemplar_optimizer_state": exemplar_report[
            "exemplar_optimizer_state_fresh"
        ],
        "exemplar_local_step_started_at_zero": exemplar_report[
            "exemplar_local_step_start"
        ]
        == 0,
        "mixed_objective_enabled": exemplar_report["mixed_objective_enabled"],
    }
    valid = bool(
        checks["corridor_process_completed"]
        and checks["corridor_checkpoint_written"]
        and checks["exemplar_process_started_after_corridor_completion"]
        and checks["exemplar_parent_checkpoint_match"]
        and not checks["corridor_optimizer_state_loaded_by_exemplar"]
        and checks["fresh_exemplar_optimizer_state"]
        and checks["exemplar_local_step_started_at_zero"]
        and not checks["mixed_objective_enabled"]
    )
    return {
        **checks,
        "corridor_checkpoint_bundle_sha256": corridor_hashes[
            "checkpoint_bundle_sha256"
        ],
        "corridor_parameter_fingerprint": corridor_fingerprint,
        "corridor_completed_steps": corridor_report["completed_steps"],
        "cycle_boundary_valid": valid,
    }


def _final_test_access_receipt(config, *, freeze_path, freeze, split_validation):
    if not freeze_path.is_file():
        raise ValueError("configuration_freeze_receipt_missing")
    freeze_valid = freeze["configuration_freeze_sha256"] == stable_hash(
        freeze["frozen_configuration"]
    )
    final_test = split_validation["final_test"]
    calibration = split_validation["calibration"]
    valid = bool(
        freeze_valid
        and final_test["artifact_role"] == "final_held_out_test"
        and final_test["artifact_manifest_sha256"]
        != calibration["artifact_manifest_sha256"]
        and split_validation["three_way_split_valid"]
    )
    if not valid:
        raise ValueError("final_test_access_invalid")
    return {
        "status": "pass",
        "configuration_freeze_sha256": freeze["configuration_freeze_sha256"],
        "configuration_frozen_before_final_test_access": True,
        "final_test_artifact_sha256": final_test["artifact_manifest_sha256"],
        "final_test_role": "final_held_out_test",
        "final_test_used_for_training": False,
        "final_test_used_for_calibration": False,
        "final_test_used_for_selection": False,
        "final_test_access_valid": True,
    }


def _evaluate_all(config, checkpoints):
    records = _target_records(config.final_test_fingerprint_artifact)
    exemplars = _exemplar_map(config.final_test_fingerprint_artifact)
    if not records or not exemplars:
        raise ValueError("held_out_teacher_exemplars_required")
    evaluations = {
        name: _evaluate_checkpoint(
            checkpoint,
            config.final_test_fingerprint_artifact,
            records,
            exemplars,
        )
        for name, checkpoint in checkpoints.items()
    }
    orders = {tuple(value["record_keys"]) for value in evaluations.values()}
    if len(orders) != 1:
        raise ValueError("held_out_record_order_mismatch")
    teacher_orders = {
        tuple(row["record_key"] for row in value["teacher_records"])
        for value in evaluations.values()
    }
    if len(teacher_orders) != 1 or not next(iter(teacher_orders)):
        raise ValueError("held_out_teacher_record_order_mismatch")
    return evaluations


def _aligned_per_record_metrics(evaluations):
    teacher_maps = {
        arm: {row["record_key"]: row for row in result["teacher_records"]}
        for arm, result in evaluations.items()
    }
    keys = [
        row["record_key"] for row in evaluations["two_cycle_final"]["teacher_records"]
    ]
    if any(set(rows) != set(keys) for rows in teacher_maps.values()):
        raise ValueError("paired evaluation records are not aligned")
    return [
        {
            "record_key": key,
            "example_id": teacher_maps["two_cycle_final"][key]["example_id"],
            "position": teacher_maps["two_cycle_final"][key]["position"],
            "baseline_score": teacher_maps["conventional_baseline"][key][
                "teacher_student_kl"
            ],
            "corridor_only_score": teacher_maps["corridor_only"][key][
                "teacher_student_kl"
            ],
            "exemplar_only_score": teacher_maps["exemplar_only"][key][
                "teacher_student_kl"
            ],
            "two_cycle_corridor_score": teacher_maps["two_cycle_corridor"][key][
                "teacher_student_kl"
            ],
            "two_cycle_final_score": teacher_maps["two_cycle_final"][key][
                "teacher_student_kl"
            ],
            **{
                f"{arm}_teacher_student_kl": rows[key]["teacher_student_kl"]
                for arm, rows in teacher_maps.items()
            },
        }
        for key in keys
    ]


def _evaluation_metrics(evaluations):
    return {
        arm: {
            "teacher": result["teacher_metrics"],
            "corridor": {
                key: result["aggregate"].get(key)
                for key in (
                    "corridor_loss_total",
                    "inside_all_rate",
                    "mean_distance_outside_corridor",
                )
            },
            "teacher_metric_availability": result["teacher_metric_availability"],
        }
        for arm, result in evaluations.items()
    }


def _paired_comparisons(config, rows):
    pairs = {
        "two_cycle_final_vs_exemplar_only_final": (
            "two_cycle_final",
            "exemplar_only",
        ),
        "two_cycle_final_vs_conventional_baseline": (
            "two_cycle_final",
            "conventional_baseline",
        ),
        "two_cycle_final_vs_corridor_only": (
            "two_cycle_final",
            "corridor_only",
        ),
        "two_cycle_final_vs_own_corridor_checkpoint": (
            "two_cycle_final",
            "two_cycle_corridor",
        ),
        "corridor_only_vs_shared_initialization": (
            "corridor_only",
            "shared_initialization",
        ),
    }
    output = {}
    for index, (name, (left, right)) in enumerate(pairs.items()):
        output[name] = paired_lower_is_better_comparison(
            {row["record_key"]: row[f"{left}_teacher_student_kl"] for row in rows},
            {row["record_key"]: row[f"{right}_teacher_student_kl"] for row in rows},
            left_name=left,
            right_name=right,
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_seed=config.bootstrap_seed + index,
            tie_tolerance=config.tie_tolerance,
        )
    return output


def _resource_accounting(
    config,
    *,
    p151,
    corridor_only,
    exemplar_only,
    two_cycle_corridor,
    two_cycle_exemplar,
):
    baseline = {
        "optimizer_steps": p151["baseline"]["optimizer_steps_completed"],
        "records_consumed": p151["baseline"]["batches_consumed"] * config.batch_size,
        "tokens_consumed": p151["baseline"]["tokens_consumed"],
        "artifact_bytes_logically_consumed": config.source_texts.stat().st_size
        * p151["baseline"]["batches_consumed"],
        "training_seconds": p151["baseline"]["wall_clock_seconds"],
        "evaluation_seconds": 0.0,
        "checkpoint_seconds": 0.0,
        "total_wall_clock_seconds": p151["baseline"]["wall_clock_seconds"],
    }
    corridor_b = _corridor_resources(corridor_only)
    corridor_d = _corridor_resources(two_cycle_corridor)
    exemplar_c = _exemplar_resources(exemplar_only)
    exemplar_d = _exemplar_resources(two_cycle_exemplar)
    return {
        "conventional_baseline": baseline,
        "corridor_only": corridor_b,
        "exemplar_only": exemplar_c,
        "two_cycle": {
            "corridor": corridor_d,
            "exemplar": exemplar_d,
            "combined": sum_stage_resources(corridor_d, exemplar_d),
        },
        "stage_matched_exemplar_budget": {
            "exemplar_only": exemplar_c,
            "two_cycle_exemplar": exemplar_d,
            "match": _budget_view(exemplar_c) == _budget_view(exemplar_d),
        },
        "orchestration_setup_overhead": {
            "discarded_p151_fingerprint_comparison_arm": True,
            "optimizer_steps": p151["fingerprint"]["optimizer_steps_completed"],
            "records_consumed": p151["fingerprint"]["records_consumed"],
            "wall_clock_seconds": p151["fingerprint"]["wall_clock_seconds"],
            "included_in_experimental_arm_budgets": False,
        },
    }


def _corridor_resources(report):
    resource = report["resource_accounting"]
    timing = report["wall_clock"]
    return {
        "optimizer_steps": report["completed_steps"],
        "records_consumed": resource["total_record_visits"],
        "tokens_consumed": resource["tokens_consumed"],
        "artifact_bytes_logically_consumed": resource[
            "artifact_bytes_logically_consumed"
        ],
        "training_seconds": timing["training_seconds"],
        "evaluation_seconds": timing["held_out_evaluation_seconds"],
        "checkpoint_seconds": timing["checkpoint_write_seconds"],
        "total_wall_clock_seconds": timing["total_wall_clock_seconds"],
    }


def _exemplar_resources(result):
    resource = read_json_object(result.output_dir / "resource_accounting.json")
    sampling = read_json_object(result.output_dir / "sampling_receipt.json")
    logical_bytes = round(
        resource["exemplar_payload_bytes"]
        * resource["total_exemplar_record_visits"]
        / max(sampling["records_selected"], 1)
    )
    return {
        "optimizer_steps": result.completed_steps,
        "records_consumed": resource["total_exemplar_record_visits"],
        "tokens_consumed": resource["tokens_consumed"],
        "artifact_bytes_logically_consumed": logical_bytes,
        "training_seconds": resource["training_seconds"],
        "evaluation_seconds": resource["evaluation_seconds"],
        "checkpoint_seconds": resource["checkpoint_write_seconds"],
        "total_wall_clock_seconds": resource["total_wall_clock_seconds"],
    }


def _budget_view(resource):
    return {
        key: resource[key]
        for key in (
            "optimizer_steps",
            "records_consumed",
            "tokens_consumed",
            "artifact_bytes_logically_consumed",
        )
    }


def _normalize_primary_result(result):
    return "two_cycle_better" if result == "two_cycle_final_better" else result


def _arm_lineage(config, checkpoints, evaluations, resources):
    software_commit = get_git_metadata(Path(__file__).resolve().parents[3]).get(
        "commit"
    )
    output = {}
    for name, checkpoint in checkpoints.items():
        loaded = load_checkpoint(checkpoint)
        output[name] = {
            "arm_name": name,
            "checkpoint_dir": str(checkpoint),
            **hash_checkpoint_bundle(checkpoint),
            "parameter_fingerprint": parameter_fingerprint(loaded.params),
            "student_config_sha256": stable_hash(loaded.manifest.student_config),
            "optimizer_config": loaded.manifest.optimizer_config,
            "completed_steps": loaded.manifest.step,
            "held_out_record_order_sha256": stable_hash(
                evaluations[name]["record_keys"]
            ),
            "held_out_artifact_sha256": file_sha256(
                config.final_test_fingerprint_artifact / "manifest.json"
            ),
            "training_artifact_sha256": file_sha256(
                config.training_fingerprint_artifact / "manifest.json"
            ),
            "software_commit": software_commit,
        }
    output["resource_views"] = resources
    boundary_path = config.output_dir / "cycle_boundary_receipt.json"
    output["two_cycle_final"].update(
        {
            "selected_hammer_config_sha256": read_json_object(
                config.selected_profile_receipt
            )["selected_profile_config_sha256"],
            "cycle_boundary_receipt_sha256": file_sha256(boundary_path),
            "corridor_checkpoint_bundle_sha256": output["two_cycle_corridor"][
                "checkpoint_bundle_sha256"
            ],
            "corridor_parameter_fingerprint": output["two_cycle_corridor"][
                "parameter_fingerprint"
            ],
        }
    )
    return output


def _required_sibling_hash(path, name):
    sibling = path.with_name(name)
    if not sibling.is_file():
        raise ValueError(f"required calibration artifact missing: {name}")
    return file_sha256(sibling)


def _publication_grade_calibration(path):
    sibling = path.with_name("publication_grade_receipt.json")
    if not sibling.is_file():
        return None
    return bool(read_json_object(sibling).get("publication_grade"))


def _validate_config(config):
    if config.output_dir.exists() and not config.overwrite:
        raise FileExistsError(
            f"output exists: {config.output_dir}; pass overwrite=True"
        )
    for name in ("baseline_steps", "corridor_steps", "exemplar_steps", "batch_size"):
        if int(getattr(config, name)) < 1:
            raise ValueError(f"{name} must be >= 1")
    for name in ("corridor_only_steps", "exemplar_only_steps"):
        value = getattr(config, name)
        if value is not None and int(value) < 1:
            raise ValueError(f"{name} must be >= 1")
    if config.budget_comparison_mode not in {
        "stage_matched",
        "equal_total_budget",
    }:
        raise ValueError("unsupported budget_comparison_mode")
    if config.bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be >= 1")
    if config.tie_tolerance < 0.0 or not math.isfinite(config.tie_tolerance):
        raise ValueError("tie_tolerance must be finite and >= 0")


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
