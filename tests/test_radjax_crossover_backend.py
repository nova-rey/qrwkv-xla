from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qrwkv_xla.fingerprint.adaptive_corridor_scheduler import (
    AdaptiveCorridorSchedulerConfig,
)
from qrwkv_xla.fingerprint.full_distillation_crossover import (
    CrossoverExecutionBackend,
    FullDistillationCrossoverConfig,
    run_full_distillation_crossover,
)
from qrwkv_xla.fingerprint.mode_plateau_controller import ModePlateauConfig
from qrwkv_xla.fingerprint.radjax_crossover_backend import (
    RadjaxCrossoverBackend,
    RadjaxCrossoverBackendConfig,
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


@pytest.mark.parametrize(
    "accounting",
    [
        {
            "arm": "adaptive_two_cycle",
            "teacher_artifact_bytes_consumed": 30,
            "corridor_artifact_bytes_consumed": 10,
            "exemplar_artifact_bytes_consumed": 20,
            "source_text_bytes_consumed": 0,
        },
        {
            "arm": "exemplar_only",
            "teacher_artifact_bytes_consumed": 30,
            "corridor_artifact_bytes_consumed": 0,
            "exemplar_artifact_bytes_consumed": 20,
            "source_text_bytes_consumed": 0,
        },
    ],
)
def test_valid_byte_accounting_contract(accounting: dict) -> None:
    validate_byte_accounting(accounting, arm=accounting["arm"])
    assert teacher_bytes_to_target(accounting) == 30


def test_byte_components_may_not_exceed_total() -> None:
    accounting = {
        "arm": "adaptive_two_cycle",
        "teacher_artifact_bytes_consumed": 20,
        "corridor_artifact_bytes_consumed": 15,
        "exemplar_artifact_bytes_consumed": 10,
        "source_text_bytes_consumed": 0,
    }
    with pytest.raises(ValueError, match="exceed"):
        validate_byte_accounting(accounting, arm=accounting["arm"])


def test_vanilla_teacher_bytes_are_zero() -> None:
    valid = {
        "arm": "vanilla",
        "teacher_artifact_bytes_consumed": 0,
        "corridor_artifact_bytes_consumed": 0,
        "exemplar_artifact_bytes_consumed": 0,
        "source_text_bytes_consumed": 100,
    }
    validate_byte_accounting(valid, arm="vanilla")
    invalid = {**valid, "teacher_artifact_bytes_consumed": 1}
    with pytest.raises(ValueError, match="vanilla"):
        validate_byte_accounting(invalid, arm="vanilla")


def test_old_additive_bytes_to_target_formula_is_rejected() -> None:
    accounting = {
        "arm": "adaptive_two_cycle",
        "teacher_artifact_bytes_consumed": 30,
        "corridor_artifact_bytes_consumed": 10,
        "exemplar_artifact_bytes_consumed": 20,
        "source_text_bytes_consumed": 0,
    }
    old_double_counted = sum(
        accounting[key]
        for key in (
            "teacher_artifact_bytes_consumed",
            "corridor_artifact_bytes_consumed",
            "exemplar_artifact_bytes_consumed",
        )
    )
    assert old_double_counted == 60
    assert teacher_bytes_to_target(accounting) == 30


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
            checkpoint_fractions=(0.0, 1.0),
            target_quality_thresholds={"teacher_student_kl": 100.0},
            bootstrap_samples=20,
            maximum_steps=8,
        ),
        backend=backend,
    )
    assert report["status"] == "pass"
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
    assert binding["ready_for_P156_5"] is True
    boundary = json.loads(
        (output / "seed_0" / "cycle_boundary_receipt.json").read_text()
    )
    assert boundary["fresh_optimizer_exact_match"] is True
    assert (
        boundary["expected_fresh_exemplar_optimizer_hash"]
        == boundary["actual_cycle_two_initial_optimizer_hash"]
    )
    assert (
        boundary["actual_cycle_two_initial_optimizer_hash"]
        != boundary["corridor_optimizer_state_hash"]
    )
