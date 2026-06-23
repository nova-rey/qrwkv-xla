from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qrwkv_xla.checkpointing import load_checkpoint
from qrwkv_xla.fingerprint.adaptive_corridor_scheduler import (
    AdaptiveCorridorSchedulerConfig,
)
from qrwkv_xla.fingerprint.full_distillation_crossover import (
    CrossoverExecutionBackend,
    FullDistillationCrossoverConfig,
    run_full_distillation_crossover,
)
from qrwkv_xla.fingerprint.mode_plateau_controller import ModePlateauConfig
from qrwkv_xla.fingerprint.provenance import parameter_fingerprint
from qrwkv_xla.fingerprint.radjax_crossover_backend import (
    RadjaxCrossoverBackend,
    RadjaxCrossoverBackendConfig,
    _strict_resources,
    _validate_trajectory_steps,
    classify_artifact_bytes,
    classify_artifact_file,
    teacher_bytes_to_target,
    validate_byte_accounting,
)
from qrwkv_xla.teachers import HFTeacherBackend
from scripts.run_full_distillation_crossover import TinyCpuCrossoverBackend


def test_radjax_backend_implements_protocol_without_synthetic_inheritance() -> None:
    assert isinstance(
        RadjaxCrossoverBackend.__new__(RadjaxCrossoverBackend),
        CrossoverExecutionBackend,
    )
    assert not issubclass(RadjaxCrossoverBackend, TinyCpuCrossoverBackend)


def test_incomplete_backend_does_not_implement_protocol() -> None:
    assert not isinstance(SimpleNamespace(), CrossoverExecutionBackend)


def test_production_backend_requires_continuous_strict_execution() -> None:
    assert RadjaxCrossoverBackend.checkpoint_execution_mode == "continuous_trajectory"
    assert RadjaxCrossoverBackend.strict_resource_accounting is True


@pytest.mark.parametrize("steps", [(1, 1, 2), (2, 1)])
def test_invalid_continuous_checkpoint_steps_fail(steps: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _validate_trajectory_steps(steps)


def test_missing_resource_is_null_with_reason_not_zero() -> None:
    resource = _strict_resources(
        {"cpu_seconds": None, "gpu_seconds": 0, "training_records": 3}, {}
    )
    assert resource["cumulative_cpu_seconds"] is None
    assert resource["resource_provenance"]["cumulative_cpu_seconds"] == {
        "value": None,
        "status": "unavailable",
        "reason": "underlying runner does not expose this measurement",
    }
    assert resource["cumulative_gpu_seconds"] == 0
    assert resource["cumulative_training_records"] == 3
    assert resource["silent_zero_defaults_present"] is False


def test_interval_resources_reconcile_with_cumulative() -> None:
    resource = _strict_resources(
        {"training_records": 5, "training_wall_seconds": 1.5},
        {"training_records": 2, "training_wall_seconds": 0.5},
    )
    assert resource["interval_training_records"] == 3
    assert resource["interval_training_wall_seconds"] == 1.0


def _accounting(arm: str, corridor: int, exemplar: int, shared: int) -> dict:
    total = corridor + exemplar + shared
    return {
        "arm": arm,
        "artifact_bytes_available": total,
        "artifact_bytes_selected": total,
        "artifact_bytes_logically_consumed": total,
        "corridor_payload_bytes": corridor,
        "exemplar_payload_bytes": exemplar,
        "shared_teacher_metadata_bytes": shared,
        "teacher_artifact_bytes_total": total,
        "source_text_bytes_consumed": 0,
    }


@pytest.mark.parametrize(
    "accounting",
    [
        _accounting("adaptive_two_cycle", 10, 20, 5),
        _accounting("exemplar_only", 0, 20, 5),
    ],
)
def test_valid_byte_accounting_contract(accounting: dict) -> None:
    validate_byte_accounting(accounting, arm=accounting["arm"])
    assert (
        teacher_bytes_to_target(accounting)
        == accounting["teacher_artifact_bytes_total"]
    )


def test_byte_components_may_not_exceed_total() -> None:
    accounting = _accounting("adaptive_two_cycle", 15, 10, 0)
    accounting["teacher_artifact_bytes_total"] = 20
    with pytest.raises(ValueError, match="disjoint"):
        validate_byte_accounting(accounting, arm=accounting["arm"])


def test_vanilla_teacher_bytes_are_zero() -> None:
    valid = _accounting("vanilla", 0, 0, 0)
    valid["source_text_bytes_consumed"] = 100
    validate_byte_accounting(valid, arm="vanilla")
    invalid = {**valid, "teacher_artifact_bytes_total": 1}
    with pytest.raises(ValueError, match="disjoint"):
        validate_byte_accounting(invalid, arm="vanilla")


def test_old_additive_bytes_to_target_formula_is_rejected() -> None:
    accounting = _accounting("adaptive_two_cycle", 10, 20, 0)
    old_double_counted = accounting["teacher_artifact_bytes_total"] + 10 + 20
    assert old_double_counted == 60
    assert teacher_bytes_to_target(accounting) == 30


def test_disjoint_100_20_30_artifact_fixture(tmp_path: Path) -> None:
    (tmp_path / "targets").mkdir()
    (tmp_path / "targets" / "corridor.bin").write_bytes(b"c" * 100)
    (tmp_path / "manifest.json").write_bytes(b"m" * 20)
    (tmp_path / "exemplars").mkdir()
    (tmp_path / "exemplars" / "exemplar.bin").write_bytes(b"e" * 30)
    result = classify_artifact_bytes(tmp_path)
    totals = result["category_bytes"]
    assert totals["corridor_payload"] == 100
    assert totals["exemplar_payload"] == 30
    assert totals["shared_teacher_metadata"] == 20
    assert result["artifact_bytes_available"] == 150
    assert result["artifact_bytes_available"] != 180
    assert {row["category"] for row in result["files"]} == {
        "corridor_payload",
        "exemplar_payload",
        "shared_teacher_metadata",
    }


def test_unknown_artifact_file_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "unknown.bin"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="exactly one"):
        classify_artifact_file(tmp_path, path)


class TinyTeacherModel:
    def __call__(self, *, input_ids, attention_mask=None):
        del attention_mask
        values = np.asarray(input_ids)
        vocab = np.arange(16, dtype=np.float32)
        logits = values[..., None].astype(np.float32) * 0.01 + vocab * 0.001
        return SimpleNamespace(logits=logits)


def test_real_radjax_three_arm_cpu_smoke(tmp_path: Path) -> None:
    source = (
        Path(__file__).parent
        / "fixtures"
        / "behavioral_fingerprint"
        / "v0_1_with_exemplars_tiny"
    )
    artifacts = {}
    for role in ("training", "calibration", "final_test"):
        destination = tmp_path / role
        shutil.copytree(source, destination)
        exemplar_path = destination / "exemplars" / "exemplars-00000.jsonl"
        exemplar_rows = [
            json.loads(line) for line in exemplar_path.read_text().splitlines() if line
        ]
        for index, row in enumerate(exemplar_rows):
            row["example_id"] = f"p137t{index:03d}"
        exemplar_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in exemplar_rows),
            encoding="utf-8",
        )
        artifacts[role] = destination
    source_texts = tmp_path / "source.jsonl"
    source_texts.write_text('{"text":"tiny real source"}\n', encoding="utf-8")
    student_config = tmp_path / "student.json"
    student_config.write_text('{"student_backend":"tiny_debug"}\n', encoding="utf-8")
    profile = tmp_path / "profile.json"
    profile.write_text('{"status":"pass"}\n', encoding="utf-8")
    mode_payload = json.loads((artifacts["training"] / "modes.json").read_text())
    mode_ids = tuple(str(mode["mode_id"]) for mode in mode_payload["modes"])
    controller = ModePlateauConfig(
        required_modes=mode_ids,
        primary_progress_metric="inside_corridor_rate",
        entry_inside_rate_threshold=0.0,
        entry_mean_distance_threshold=1e6,
        entry_worst_violation_threshold=1e6,
        regression_inside_rate_floor=-0.1,
        regression_mean_distance_ceiling=2e6,
        regression_worst_violation_ceiling=2e6,
        minimum_observations=2,
        progress_window_observations=2,
        plateau_patience_observations=1,
        maximum_corridor_steps=4,
    )
    scheduler = AdaptiveCorridorSchedulerConfig(
        controller=controller,
        mode_weights={mode_id: 1.0 for mode_id in mode_ids},
        global_freeze_confirmation_observations=1,
    )
    teacher = HFTeacherBackend(
        model_id="local-tiny-causal-teacher",
        tokenizer=SimpleNamespace(
            vocab_size=16,
            name_or_path="local-tiny-tokenizer",
            pad_token_id=0,
            eos_token_id=0,
        ),
        model=TinyTeacherModel(),
    )
    output = tmp_path / "output"
    backend = RadjaxCrossoverBackend(
        RadjaxCrossoverBackendConfig(
            training_artifact=artifacts["training"],
            calibration_artifact=artifacts["calibration"],
            final_test_artifact=artifacts["final_test"],
            source_texts=source_texts,
            selected_profile_receipt=profile,
            receipt_root=output,
            adaptive_scheduler=scheduler,
            teacher_backend=teacher,
            student_backend="tiny_debug",
            batch_size=2,
            evaluation_prompt_limit=1,
            prefix_length=1,
        )
    )
    report = run_full_distillation_crossover(
        FullDistillationCrossoverConfig(
            training_artifact=artifacts["training"],
            calibration_artifact=artifacts["calibration"],
            final_test_artifact=artifacts["final_test"],
            source_texts=source_texts,
            student_config=student_config,
            selected_profile_receipt=profile,
            output_dir=output,
            seeds=(0,),
            checkpoint_fractions=(0.0, 1.0, 2.0),
            target_quality_thresholds={"teacher_student_kl": 100.0},
            bootstrap_samples=20,
            maximum_steps=8,
        ),
        backend=backend,
    )
    assert report["status"] == "pass"
    assert report["checkpoint_execution_mode"] == "continuous_trajectory"
    assert report["strict_resource_accounting"] is True
    binding = json.loads((output / "radjax_backend_binding_receipt.json").read_text())
    assert binding["status"] == "pass"
    assert binding["backend"] == "radjax"
    assert binding["real_student_used"] is True
    assert binding["real_teacher_used"] is True
    assert binding["real_adaptive_runner_used"] is True
    assert binding["real_vanilla_runner_used"] is True
    assert binding["real_exemplar_runner_used"] is True
    assert binding["fresh_optimizer_exact_match"] is True
    assert binding["byte_accounting_valid"] is True
    assert binding["publication_grade"] is False
    assert binding["full_distillation_run_started"] is False
    assert binding["ready_for_P156_5"] is False
    state = json.loads((output / "crossover_experiment_state.json").read_text())
    adaptive_rows = state["arm_checkpoint_results"]["0:adaptive_two_cycle"]
    assert len(adaptive_rows) == 3
    boundary_resource = adaptive_rows[0]["resource_accounting"]
    assert boundary_resource["cycle_two_optimizer_instantiated"] is False
    assert boundary_resource["actual_initial_exemplar_optimizer_hash"] is None
    assert boundary_resource["fresh_optimizer_exact_match"] is None
    assert boundary_resource["fresh_optimizer_proof_status"] == "not_applicable"
    proof_resource = adaptive_rows[1]["resource_accounting"]
    assert proof_resource["cycle_two_optimizer_instantiated"] is True
    assert proof_resource["fresh_optimizer_proof_status"] == "proven"
    assert proof_resource["fresh_optimizer_exact_match"] is True
    assert (
        proof_resource["expected_fresh_exemplar_optimizer_hash"]
        == proof_resource["actual_initial_exemplar_optimizer_hash"]
    )
    assert proof_resource["cumulative_training_records"] == (
        boundary_resource["cumulative_training_records"]
        + proof_resource["cycle_two_cumulative_resources"]["training_records"]
    )
    assert proof_resource["cumulative_training_tokens"] == (
        boundary_resource["cumulative_training_tokens"]
        + proof_resource["cycle_two_cumulative_resources"]["training_tokens"]
    )
    assert proof_resource["cumulative_training_wall_seconds"] == (
        boundary_resource["cumulative_training_wall_seconds"]
        + proof_resource["cycle_two_cumulative_resources"]["training_wall_seconds"]
    )
    for arm in ("vanilla", "exemplar_only", "adaptive_two_cycle"):
        rows = state["arm_checkpoint_results"][f"0:{arm}"]
        assert [row["total_step"] for row in rows] == sorted(
            row["total_step"] for row in rows
        )
        for previous, current in zip(rows, rows[1:], strict=False):
            assert current["parent_checkpoint_hash"] == previous["checkpoint_hash"]
        for row in rows:
            resource = row["resource_accounting"]
            if (
                row["total_step"] > 0
                and arm != "adaptive_two_cycle"
                or row["exemplar_steps"] > 0
            ):
                assert resource["cumulative_training_records"] > 0
                assert resource["cumulative_training_tokens"] > 0
                assert resource["cumulative_training_wall_seconds"] > 0
            assert resource.get("continuous_trajectory_confirmed") is True
            assert resource.get("silent_zero_defaults_present") is False
            if resource.get("interval_total_wall_seconds") is not None:
                assert resource["interval_total_wall_seconds"] >= 0
    byte_receipt = json.loads(
        (output / "real_backend_byte_accounting_receipt.json").read_text()
    )
    assert byte_receipt["teacher_artifact_bytes_total"] == sum(
        byte_receipt[key]
        for key in (
            "corridor_payload_bytes",
            "exemplar_payload_bytes",
            "shared_teacher_metadata_bytes",
        )
    )
    assert (
        byte_receipt["artifact_bytes_logically_consumed"]
        == byte_receipt["teacher_artifact_bytes_total"]
    )
    boundary = json.loads(
        (output / "seed_0" / "cycle_boundary_receipt.json").read_text()
    )
    assert boundary["fresh_optimizer_exact_match"] is True
    assert boundary["fresh_optimizer_proof_status"] == "proven"
    assert boundary["freshness_proof_total_step"] > boundary["boundary_total_step"]
    assert (
        boundary["expected_fresh_exemplar_optimizer_hash"]
        == boundary["actual_cycle_two_initial_optimizer_hash"]
    )
    assert (
        boundary["actual_cycle_two_initial_optimizer_hash"]
        != boundary["corridor_optimizer_state_hash"]
    )
    resume_shared = backend.create_shared_initialization(
        seed=11, output_dir=tmp_path / "resume_shared"
    )
    interrupted_dir = tmp_path / "resume_interrupted"
    first_segment = backend._train_vanilla_trajectory(
        11, resume_shared, (1, 2), interrupted_dir
    )
    resumed_segment = backend._train_vanilla_trajectory(
        11, resume_shared, (3,), interrupted_dir
    )
    uninterrupted = backend._train_vanilla_trajectory(
        11, resume_shared, (1, 2, 3), tmp_path / "resume_uninterrupted"
    )
    assert first_segment[-1].total_step == 2
    assert (
        resumed_segment[0].parent_checkpoint_hash == first_segment[-1].checkpoint_hash
    )
    assert resumed_segment[-1].checkpoint_hash == uninterrupted[-1].checkpoint_hash
    assert (
        resumed_segment[-1].optimizer_final_state_hash
        == uninterrupted[-1].optimizer_final_state_hash
    )
    resume_discovery = backend.discover_adaptive_cycle_one(
        seed=11,
        shared_initialization=resume_shared,
        output_dir=tmp_path / "resume_cycle_one",
    )
    exemplar_partial = backend._train_exemplar_trajectory(
        "exemplar_only",
        11,
        resume_shared,
        resume_discovery,
        (1, 2),
        tmp_path / "exemplar_resumed",
    )
    exemplar_resumed = backend._train_exemplar_trajectory(
        "exemplar_only",
        11,
        resume_shared,
        resume_discovery,
        (3,),
        tmp_path / "exemplar_resumed",
    )
    exemplar_uninterrupted = backend._train_exemplar_trajectory(
        "exemplar_only",
        11,
        resume_shared,
        resume_discovery,
        (1, 2, 3),
        tmp_path / "exemplar_uninterrupted",
    )
    s = int(resume_discovery.optimizer_steps_completed or 0)
    adaptive_partial = backend._train_exemplar_trajectory(
        "adaptive_two_cycle",
        11,
        resume_shared,
        resume_discovery,
        (s, s + 1),
        tmp_path / "adaptive_resumed",
    )
    adaptive_resumed = backend._train_exemplar_trajectory(
        "adaptive_two_cycle",
        11,
        resume_shared,
        resume_discovery,
        (s + 2,),
        tmp_path / "adaptive_resumed",
    )
    adaptive_uninterrupted = backend._train_exemplar_trajectory(
        "adaptive_two_cycle",
        11,
        resume_shared,
        resume_discovery,
        (s, s + 1, s + 2),
        tmp_path / "adaptive_uninterrupted",
    )

    def resume_proof(interrupted, resumed, full):
        resumed_final = resumed[-1]
        full_final = full[-1]
        parameter_match = parameter_fingerprint(
            load_checkpoint(resumed_final.checkpoint).params
        ) == parameter_fingerprint(load_checkpoint(full_final.checkpoint).params)
        optimizer_match = (
            resumed_final.optimizer_final_state_hash
            == full_final.optimizer_final_state_hash
        )
        counters_match = all(
            resumed_final.resource_accounting[name]
            == full_final.resource_accounting[name]
            for name in (
                "cumulative_optimizer_steps",
                "cumulative_training_records",
                "cumulative_training_tokens",
            )
        )
        return {
            "interrupted_at_step": interrupted[-1].total_step,
            "uninterrupted_final_checkpoint_hash": full_final.checkpoint_hash,
            "resumed_final_checkpoint_hash": resumed_final.checkpoint_hash,
            "uninterrupted_final_optimizer_hash": full_final.optimizer_final_state_hash,
            "resumed_final_optimizer_hash": resumed_final.optimizer_final_state_hash,
            "parameter_hash_match": parameter_match,
            "optimizer_hash_match": optimizer_match,
            "counters_match": counters_match,
            "completed_steps_replayed": False,
            "pass": parameter_match and optimizer_match and counters_match,
        }

    resume_evidence = {
        "vanilla": resume_proof(first_segment, resumed_segment, uninterrupted),
        "exemplar_only": resume_proof(
            exemplar_partial, exemplar_resumed, exemplar_uninterrupted
        ),
        "adaptive_two_cycle": resume_proof(
            adaptive_partial, adaptive_resumed, adaptive_uninterrupted
        ),
    }
    backend.record_resume_evidence(resume_evidence)
    final_binding = json.loads(
        (output / "radjax_backend_binding_receipt.json").read_text()
    )
    assert final_binding["ready_for_P156_5"] is True
    evidence_dir = os.environ.get("P156_4_1_1_SMOKE_RECEIPT_DIR")
    if evidence_dir:
        destination = Path(evidence_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for source_path in (
            output / "real_backend_integration_smoke_report.json",
            output / "radjax_backend_binding_receipt.json",
            output / "real_backend_optimizer_freshness_receipt.json",
            output / "real_backend_byte_accounting_receipt.json",
            output / "seed_0" / "cycle_boundary_receipt.json",
            output / "full_distillation_crossover_summary.md",
        ):
            shutil.copy2(source_path, destination / source_path.name)
    p156_4_2_evidence = os.environ.get("P156_4_2_SMOKE_RECEIPT_DIR")
    if p156_4_2_evidence:
        destination = Path(p156_4_2_evidence)
        destination.mkdir(parents=True, exist_ok=True)
        for name in (
            "continuous_trajectory_receipt.json",
            "resource_field_mapping_receipt.json",
            "strict_resource_accounting_receipt.json",
            "continuous_resume_receipt.json",
            "p156_4_2_real_cpu_smoke_report.json",
        ):
            shutil.copy2(output / name, destination / name)
        (destination / "summary.md").write_text(
            "# P156.4.2 Validation\n\n"
            "- Status: pass\n"
            "- Backend: radjax\n"
            "- Checkpoint execution: continuous_trajectory\n"
            "- Strict resource accounting: true\n"
            "- Publication grade: false\n"
            "- Full distillation run started: false\n",
            encoding="utf-8",
        )
    p156_4_2_1_evidence = os.environ.get("P156_4_2_1_SMOKE_RECEIPT_DIR")
    if p156_4_2_1_evidence:
        destination = Path(p156_4_2_1_evidence)
        destination.mkdir(parents=True, exist_ok=True)
        for name in (
            "cycle_one_resource_baseline_receipt.json",
            "adaptive_full_cost_accounting_receipt.json",
            "continuous_resume_receipt.json",
            "strict_resource_accounting_receipt.json",
            "p156_4_2_1_real_cpu_smoke_report.json",
        ):
            shutil.copy2(output / name, destination / name)
        (destination / "summary.md").write_text(
            "# P156.4.2.1 Validation\n\n"
            "- Status: pass\n"
            "- Adaptive full-cost accounting: valid\n"
            "- Three-arm resume equivalence: valid\n"
            "- Completed steps replayed: false\n"
            "- Publication grade: false\n"
            "- Full distillation run started: false\n"
            "- Ready for P156.5: true\n",
            encoding="utf-8",
        )
