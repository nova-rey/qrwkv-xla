from __future__ import annotations

import subprocess
import sys

from qrwkv_xla.validation.pipeline import (
    PipelineValidationResult,
    ValidationStepResult,
    build_validation_commands,
    format_pipeline_validation_result,
    run_pipeline_validation,
)


def _joined(commands: list[tuple[str, ...]]) -> list[str]:
    return [" ".join(command) for command in commands]


def test_default_command_list_includes_expected_safe_commands() -> None:
    commands = build_validation_commands()
    joined = _joined(commands)

    assert commands[0] == (sys.executable, "scripts/print_env.py")
    assert any("scripts/xla_inspect.py" in command for command in joined)
    assert any("scripts/smoke_cpu.py" in command for command in joined)
    assert any("scripts/smoke_tpu.py" in command for command in joined)
    assert any("scripts/resolve_qwen_policy.py" in command for command in joined)
    assert any("teacher_export_qwen_dryrun.yaml" in command for command in joined)
    assert any("scripts/inspect_prompt_corpus.py" in command for command in joined)
    assert any("scripts/create_prompt_manifest.py" in command for command in joined)
    assert any(
        "teacher_export_qwen_dryrun_corpus.yaml" in command for command in joined
    )
    assert any("teacher_export_stub.yaml" in command for command in joined)
    assert any("scripts/inspect_targets.py" in command for command in joined)
    assert any("tiny_student" in command for command in joined)
    assert any("rwkv7_reference" in command for command in joined)
    assert any("scripts/run_distill_stage.py" in command for command in joined)
    assert any("--optimizer adamw" in command for command in joined)
    assert any(
        "distill_stage0_adamw_clipped_stub.yaml" in command for command in joined
    )
    assert any("--track-run" in command for command in joined)
    assert any("--run-root runs/pipeline_smoke" in command for command in joined)
    assert any(
        "--checkpoint-out checkpoints/pipeline_smoke/stage0" in command
        for command in joined
    )
    assert any(
        "--resume-from checkpoints/pipeline_smoke/stage0" in command
        for command in joined
    )
    assert any("scripts/generate_from_checkpoint.py" in command for command in joined)
    assert any(
        "--output-dir eval_outputs/pipeline_generation_smoke" in command
        for command in joined
    )
    assert any("scripts/evaluate_checkpoint.py" in command for command in joined)
    assert any("configs/eval_regression_smoke.yaml" in command for command in joined)
    assert any(
        "--output-dir eval_outputs/pipeline_eval_smoke" in command for command in joined
    )
    assert any("scripts/tpu_distill_smoke.py" in command for command in joined)


def test_default_command_list_excludes_hf_export() -> None:
    joined = "\n".join(_joined(build_validation_commands()))

    assert "teacher_export_hf_tiny.yaml" not in joined
    assert "teacher_export_hf_tiny_corpus.yaml" not in joined
    assert "artifacts/teacher_targets/hf_tiny" not in joined


def test_include_hf_adds_hf_export_inspect_and_distill() -> None:
    joined = _joined(build_validation_commands(include_hf=True))

    assert any("teacher_export_hf_tiny.yaml" in command for command in joined)
    assert any("teacher_export_hf_tiny_corpus.yaml" in command for command in joined)
    assert any(
        "scripts/inspect_targets.py artifacts/teacher_targets/hf_tiny" in command
        for command in joined
    )
    assert any(
        "scripts/run_distill_stage.py" in command
        and "--targets artifacts/teacher_targets/hf_tiny" in command
        for command in joined
    )


def test_require_tpu_adds_require_tpu_to_tpu_distill_smoke() -> None:
    tpu_commands = [
        command
        for command in build_validation_commands(require_tpu=True)
        if "scripts/tpu_distill_smoke.py" in command
    ]

    assert len(tpu_commands) == 1
    assert "--require-tpu" in tpu_commands[0]


def test_failed_steps_property_works() -> None:
    failed = ValidationStepResult(
        name="bad",
        command=("bad",),
        returncode=1,
        passed=False,
    )
    passed = ValidationStepResult(
        name="good",
        command=("good",),
        returncode=0,
        passed=True,
    )
    result = PipelineValidationResult(passed=False, steps=(passed, failed))

    assert result.failed_steps == (failed,)


def test_stop_on_failure_stops_after_first_failure(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, stdout="no", stderr="bad")

    monkeypatch.setattr(
        "qrwkv_xla.validation.pipeline.run_validation_command", fake_run
    )

    result = run_pipeline_validation(stop_on_failure=True)

    assert not result.passed
    assert len(result.steps) == 1
    assert calls == [build_validation_commands()[0]]


def test_continue_on_failure_runs_all_commands(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="")

    monkeypatch.setattr(
        "qrwkv_xla.validation.pipeline.run_validation_command", fake_run
    )

    result = run_pipeline_validation(stop_on_failure=False)

    assert not result.passed
    assert len(result.steps) == len(build_validation_commands())
    assert calls == build_validation_commands()


def test_formatting_includes_step_names_and_pass_fail_state() -> None:
    result = PipelineValidationResult(
        passed=False,
        steps=(
            ValidationStepResult("first", ("ok",), 0, True),
            ValidationStepResult("second", ("bad",), 2, False),
        ),
    )

    formatted = format_pipeline_validation_result(result)

    assert "PASS first" in formatted
    assert "FAIL second" in formatted
