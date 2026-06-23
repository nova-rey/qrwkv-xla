from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from qrwkv_xla.fingerprint.full_distillation_crossover import (
    FullDistillationCrossoverConfig,
    build_crossover_plan,
    derive_checkpoint_schedule,
    first_observed_target_crossing,
    paired_bootstrap_statistics,
    run_full_distillation_crossover,
)
from scripts.run_full_distillation_crossover import (
    TinyCpuCrossoverBackend,
    build_smoke_inputs,
)


def _config(tmp_path: Path, **overrides: object) -> FullDistillationCrossoverConfig:
    inputs = build_smoke_inputs(tmp_path / "inputs")
    values = {
        "training_artifact": inputs["training"],
        "calibration_artifact": inputs["calibration"],
        "final_test_artifact": inputs["final_test"],
        "source_texts": inputs["source_texts"],
        "student_config": inputs["student_config"],
        "selected_profile_receipt": inputs["selected_profile_receipt"],
        "output_dir": tmp_path / "output",
        "seeds": (7,),
        "checkpoint_fractions": (0.0, 0.5),
        "bootstrap_samples": 100,
        "target_quality_thresholds": {"teacher_student_kl": 1.0},
        "maximum_steps": 3,
    }
    values.update(overrides)
    return FullDistillationCrossoverConfig(**values)


def test_checkpoint_schedule_is_derived_deduplicated_and_strict() -> None:
    schedule = derive_checkpoint_schedule(2, (0.0, 0.1, 0.25, 0.5, 1.0))
    assert schedule == (2, 3, 4)
    assert all(
        right > left for left, right in zip(schedule, schedule[1:], strict=False)
    )
    assert schedule[-1] == 2 + 2


def test_plan_only_calculation_does_not_create_output(tmp_path: Path) -> None:
    config = _config(tmp_path)
    plan = build_crossover_plan(config)
    assert plan.seeds == (7,)
    assert plan.arms == ("vanilla", "exemplar_only", "adaptive_two_cycle")
    assert plan.adaptive_discoveries_required == 1
    assert plan.estimated_maximum_training_runs == 4
    assert not config.output_dir.exists()


def test_max_new_training_runs_fails_before_execution(tmp_path: Path) -> None:
    config = _config(tmp_path, max_new_training_runs=3)
    with pytest.raises(ValueError, match="exceeding"):
        build_crossover_plan(config)
    assert not config.output_dir.exists()


def test_paired_bootstrap_is_deterministic_and_inconclusive_at_zero() -> None:
    left = [1.0, 2.0, 3.0, 4.0]
    right = [2.0, 1.0, 4.0, 3.0]
    first = paired_bootstrap_statistics(left, right, samples=500, seed=9)
    second = paired_bootstrap_statistics(left, right, samples=500, seed=9)
    assert first == second
    assert first["result"] == "inconclusive"


def test_target_crossing_uses_first_observed_checkpoint_only() -> None:
    rows = [
        {"total_step": 20, "teacher_student_kl": 0.4},
        {"total_step": 10, "teacher_student_kl": 0.8},
        {"total_step": 15, "teacher_student_kl": 0.6},
    ]
    crossing = first_observed_target_crossing(
        rows,
        metric="teacher_student_kl",
        threshold=0.65,
        direction="lower",
    )
    assert crossing is not None
    assert crossing["total_step"] == 15


def test_initialization_mismatch_fails(tmp_path: Path) -> None:
    class MismatchBackend(TinyCpuCrossoverBackend):
        def train_arm(self, **kwargs):
            checkpoints = super().train_arm(**kwargs)
            return [
                replace(checkpoint, initial_parameter_hash="wrong")
                for checkpoint in checkpoints
            ]

    with pytest.raises(ValueError, match="initialization hash mismatch"):
        run_full_distillation_crossover(_config(tmp_path), backend=MismatchBackend())


def test_tiny_three_arm_implementation_smoke(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = run_full_distillation_crossover(config, backend=TinyCpuCrossoverBackend())
    assert report["phase"] == "P156.4"
    assert report["status"] == "pass"
    assert report["implementation_smoke_complete"] is True
    assert report["full_distillation_run_started"] is False
    assert report["publication_grade"] is False
    assert report["ready_for_P156_5"] is True
    assert report["checkpoint_cells_completed"] == 6

    output = config.output_dir
    for name in (
        "full_distillation_crossover_report.json",
        "full_distillation_crossover_summary.md",
        "crossover_experiment_state.json",
        "matched_checkpoint_schedule.jsonl",
        "crossover_checkpoint_comparisons.jsonl",
        "teacher_forced_checkpoint_metrics.jsonl",
        "student_prefix_checkpoint_metrics.jsonl",
        "free_running_checkpoint_metrics.jsonl",
        "checkpoint_resource_accounting.jsonl",
        "publication_claims_receipt.json",
    ):
        assert (output / name).is_file()
    teacher = _jsonl(output / "teacher_forced_checkpoint_metrics.jsonl")
    prefixes = _jsonl(output / "student_prefix_checkpoint_metrics.jsonl")
    generations = _jsonl(output / "free_running_checkpoint_metrics.jsonl")
    comparisons = _jsonl(output / "crossover_checkpoint_comparisons.jsonl")
    assert len(teacher) == len(prefixes) == len(generations) == 6
    assert all(row["contexts_generated_by_student"] for row in prefixes)
    assert all(row["exact_text_match_required"] is False for row in generations)
    assert {tuple(row["arm_pair"]) for row in comparisons} == {
        ("adaptive_two_cycle", "exemplar_only"),
        ("adaptive_two_cycle", "vanilla"),
        ("exemplar_only", "vanilla"),
    }
    shared = json.loads(
        (output / "seed_7" / "shared_initialization_receipt.json").read_text()
    )
    assert shared["used_by_arms"] == [
        "vanilla",
        "exemplar_only",
        "adaptive_two_cycle",
    ]
    adaptive = json.loads(
        (output / "seed_7" / "adaptive_completion_receipt.json").read_text()
    )
    assert adaptive["S"] == 2
    assert adaptive["confirmation_only_evaluations"] == 2
    schedule = json.loads(
        (output / "seed_7" / "matched_checkpoint_schedule.json").read_text()
    )
    assert schedule["total_step_checkpoints"] == [2, 3]
    boundary = json.loads(
        (output / "seed_7" / "cycle_boundary_receipt.json").read_text()
    )
    assert boundary["fresh_optimizer_confirmed"] is True


def test_completed_resume_does_not_rerun_backend(tmp_path: Path) -> None:
    config = _config(tmp_path)
    backend = TinyCpuCrossoverBackend()
    first = run_full_distillation_crossover(config, backend=backend)

    class FailingBackend(TinyCpuCrossoverBackend):
        def create_shared_initialization(self, **kwargs):
            raise AssertionError("completed resume reran backend")

    resumed = run_full_distillation_crossover(
        replace(config, resume=True), backend=FailingBackend()
    )
    assert resumed == first


def test_changed_config_invalidates_resume(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_full_distillation_crossover(config, backend=TinyCpuCrossoverBackend())
    with pytest.raises(ValueError, match="resume config hash mismatch"):
        run_full_distillation_crossover(
            replace(config, resume=True, bootstrap_seed=99),
            backend=TinyCpuCrossoverBackend(),
        )


def test_partial_resume_skips_discovery_and_completed_evaluation(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    class CrashDuringEvaluation(TinyCpuCrossoverBackend):
        def __init__(self):
            self.calls = 0

        def evaluate_checkpoint(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("synthetic evaluation interruption")
            return super().evaluate_checkpoint(**kwargs)

    with pytest.raises(RuntimeError, match="interruption"):
        run_full_distillation_crossover(config, backend=CrashDuringEvaluation())

    class ResumeBackend(TinyCpuCrossoverBackend):
        def create_shared_initialization(self, **kwargs):
            raise AssertionError("resume reran shared initialization")

        def discover_adaptive_cycle_one(self, **kwargs):
            raise AssertionError("resume reran adaptive discovery")

        def train_arm(self, **kwargs):
            raise AssertionError("resume reran completed arm training")

        def evaluate_checkpoint(self, **kwargs):
            checkpoint = kwargs["checkpoint"]
            if checkpoint.arm == "vanilla" and checkpoint.total_step == 2:
                raise AssertionError("resume reran completed evaluation cell")
            return super().evaluate_checkpoint(**kwargs)

    report = run_full_distillation_crossover(
        replace(config, resume=True), backend=ResumeBackend()
    )
    assert report["status"] == "pass"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]
