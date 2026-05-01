from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FAKE_TARGETS = "artifacts/teacher_targets/fake_export"
FAKE_LOGITS_TARGETS = "artifacts/teacher_targets/fake_export_logits"
HF_TINY_TARGETS = "artifacts/teacher_targets/hf_tiny"
HF_TINY_CORPUS_TARGETS = "artifacts/teacher_targets/hf_tiny_corpus"
CHECKPOINT_SMOKE = "checkpoints/pipeline_smoke/stage0"
CHECKPOINT_SMOKE_RESUME = "checkpoints/pipeline_smoke/stage0_resume"
CHECKPOINT_ADAMW_SMOKE = "checkpoints/pipeline_smoke/adamw"
CHECKPOINT_SCHEDULED_ADAMW_SMOKE = "checkpoints/pipeline_smoke/adamw_schedule"
CHECKPOINT_HIDDEN_FOR_LOGITS = "checkpoints/pipeline_hidden_only_for_logits"
CHECKPOINT_HIDDEN_PLUS_LOGITS = "checkpoints/pipeline_hidden_plus_logits"
EVAL_GENERATION_SMOKE = "eval_outputs/pipeline_generation_smoke"
EVAL_REGRESSION_SMOKE = "eval_outputs/pipeline_eval_smoke"
TRACKING_SMOKE_ROOT = "runs/pipeline_smoke"


@dataclass(frozen=True)
class ValidationStepResult:
    name: str
    command: tuple[str, ...]
    returncode: int
    passed: bool
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class PipelineValidationResult:
    passed: bool
    steps: tuple[ValidationStepResult, ...] = ()

    @property
    def failed_steps(self) -> tuple[ValidationStepResult, ...]:
        return tuple(step for step in self.steps if not step.passed)


def build_validation_commands(
    include_hf: bool = False,
    require_tpu: bool = False,
    max_steps: int = 2,
) -> list[tuple[str, ...]]:
    max_steps_value = str(max_steps)
    tpu_distill_command = [
        sys.executable,
        "scripts/tpu_distill_smoke.py",
        "--targets",
        FAKE_TARGETS,
        "--max-steps",
        max_steps_value,
    ]
    if require_tpu:
        tpu_distill_command.append("--require-tpu")

    commands: list[tuple[str, ...]] = [
        (sys.executable, "scripts/print_env.py"),
        (sys.executable, "scripts/xla_inspect.py"),
        (sys.executable, "scripts/smoke_cpu.py"),
        (sys.executable, "scripts/smoke_tpu.py"),
        (
            sys.executable,
            "scripts/resolve_qwen_policy.py",
            "Qwen3.latest",
            "--allow-unresolved",
        ),
        (
            sys.executable,
            "scripts/export_teacher_targets.py",
            "--config",
            "configs/teacher_export_qwen_dryrun.yaml",
            "--dry-run",
            "--resolve-qwen-policy",
            "--allow-unresolved-policy",
        ),
        (
            sys.executable,
            "scripts/inspect_prompt_corpus.py",
            "corpora/smoke_prompts.jsonl",
        ),
        (
            sys.executable,
            "scripts/create_prompt_manifest.py",
            "corpora/smoke_prompts.jsonl",
            "--out",
            "artifacts/prompt_manifests/smoke_prompts.manifest.json",
            "--description",
            "Pipeline smoke corpus.",
            "--overwrite",
        ),
        (
            sys.executable,
            "scripts/export_teacher_targets.py",
            "--config",
            "configs/teacher_export_qwen_dryrun_corpus.yaml",
            "--dry-run",
            "--resolve-qwen-policy",
            "--allow-unresolved-policy",
        ),
        (
            sys.executable,
            "scripts/export_teacher_targets.py",
            "--config",
            "configs/teacher_export_stub.yaml",
        ),
        (sys.executable, "scripts/inspect_targets.py", FAKE_TARGETS),
        (
            sys.executable,
            "scripts/export_teacher_targets.py",
            "--config",
            "configs/teacher_export_stub_logits.yaml",
        ),
        (sys.executable, "scripts/inspect_targets.py", FAKE_LOGITS_TARGETS),
        (
            sys.executable,
            "scripts/run_distill_stage.py",
            "--config",
            "configs/distill_stage0_adamw_schedule_stub.yaml",
            "--max-steps",
            max_steps_value,
            "--checkpoint-out",
            CHECKPOINT_SCHEDULED_ADAMW_SMOKE,
            "--checkpoint-overwrite",
        ),
        (
            sys.executable,
            "scripts/train_student_smoke.py",
            "--targets",
            FAKE_TARGETS,
            "--student-architecture",
            "tiny_student",
            "--max-steps",
            max_steps_value,
        ),
        (
            sys.executable,
            "scripts/train_student_smoke.py",
            "--targets",
            FAKE_TARGETS,
            "--student-architecture",
            "rwkv7_reference",
            "--max-steps",
            max_steps_value,
        ),
        (
            sys.executable,
            "scripts/run_distill_stage.py",
            "--config",
            "configs/distill_stage0_logits_stub.yaml",
            "--max-steps",
            max_steps_value,
        ),
        (
            sys.executable,
            "scripts/run_distill_stage.py",
            "--config",
            "configs/distill_stage0_stub.yaml",
            "--max-steps",
            max_steps_value,
        ),
        (
            sys.executable,
            "scripts/run_distill_stage.py",
            "--config",
            "configs/distill_stage0_stub.yaml",
            "--max-steps",
            "1",
            "--track-run",
            "--run-root",
            TRACKING_SMOKE_ROOT,
            "--run-name",
            "pipeline-smoke",
            "--run-overwrite",
        ),
        (
            sys.executable,
            "scripts/run_distill_stage.py",
            "--config",
            "configs/distill_stage0_stub.yaml",
            "--max-steps",
            "1",
            "--checkpoint-out",
            CHECKPOINT_SMOKE,
            "--checkpoint-overwrite",
        ),
        (
            sys.executable,
            "scripts/run_distill_stage.py",
            "--config",
            "configs/distill_stage0_stub.yaml",
            "--max-steps",
            "1",
            "--optimizer",
            "adamw",
            "--learning-rate",
            "0.001",
            "--weight-decay",
            "0.01",
            "--checkpoint-out",
            CHECKPOINT_ADAMW_SMOKE,
            "--checkpoint-overwrite",
        ),
        (
            sys.executable,
            "scripts/run_distill_stage.py",
            "--config",
            "configs/distill_stage0_stub.yaml",
            "--max-steps",
            "1",
            "--resume-from",
            CHECKPOINT_SMOKE,
            "--checkpoint-out",
            CHECKPOINT_SMOKE_RESUME,
            "--checkpoint-overwrite",
        ),
        (
            sys.executable,
            "scripts/run_distill_stage.py",
            "--config",
            "configs/distill_stage0_stub.yaml",
            "--targets",
            FAKE_TARGETS,
            "--max-steps",
            "1",
            "--checkpoint-out",
            CHECKPOINT_HIDDEN_FOR_LOGITS,
            "--checkpoint-overwrite",
        ),
        (
            sys.executable,
            "scripts/run_distill_stage.py",
            "--config",
            "configs/distill_stage0_logits_stub.yaml",
            "--targets",
            FAKE_LOGITS_TARGETS,
            "--max-steps",
            "1",
            "--resume-from",
            CHECKPOINT_HIDDEN_FOR_LOGITS,
            "--checkpoint-out",
            CHECKPOINT_HIDDEN_PLUS_LOGITS,
            "--checkpoint-overwrite",
        ),
        (
            sys.executable,
            "scripts/generate_from_checkpoint.py",
            "--checkpoint",
            CHECKPOINT_HIDDEN_PLUS_LOGITS,
            "--prompt",
            "Hello QRWKV",
            "--max-new-tokens",
            "4",
            "--output-dir",
            EVAL_GENERATION_SMOKE,
        ),
        (
            sys.executable,
            "scripts/evaluate_checkpoint.py",
            "--checkpoint",
            CHECKPOINT_HIDDEN_PLUS_LOGITS,
            "--config",
            "configs/eval_regression_smoke.yaml",
            "--output-dir",
            EVAL_REGRESSION_SMOKE,
        ),
        tuple(tpu_distill_command),
    ]

    if include_hf:
        commands.extend(
            [
                (
                    sys.executable,
                    "scripts/export_teacher_targets.py",
                    "--config",
                    "configs/teacher_export_hf_tiny.yaml",
                ),
                (sys.executable, "scripts/inspect_targets.py", HF_TINY_TARGETS),
                (
                    sys.executable,
                    "scripts/export_teacher_targets.py",
                    "--config",
                    "configs/teacher_export_hf_tiny_corpus.yaml",
                ),
                (
                    sys.executable,
                    "scripts/inspect_targets.py",
                    HF_TINY_CORPUS_TARGETS,
                ),
                (
                    sys.executable,
                    "scripts/run_distill_stage.py",
                    "--config",
                    "configs/distill_stage0_stub.yaml",
                    "--targets",
                    HF_TINY_TARGETS,
                    "--max-steps",
                    max_steps_value,
                ),
            ]
        )

    return commands


def run_validation_command(
    command: tuple[str, ...],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def run_pipeline_validation(
    include_hf: bool = False,
    require_tpu: bool = False,
    max_steps: int = 2,
    stop_on_failure: bool = True,
) -> PipelineValidationResult:
    results: list[ValidationStepResult] = []
    commands = build_validation_commands(
        include_hf=include_hf,
        require_tpu=require_tpu,
        max_steps=max_steps,
    )
    for command in commands:
        completed = run_validation_command(command)
        passed = completed.returncode == 0
        results.append(
            ValidationStepResult(
                name=build_step_name(command),
                command=command,
                returncode=completed.returncode,
                passed=passed,
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
            )
        )
        if not passed and stop_on_failure:
            break

    steps = tuple(results)
    return PipelineValidationResult(
        passed=all(step.passed for step in steps),
        steps=steps,
    )


def format_pipeline_validation_result(result: PipelineValidationResult) -> str:
    lines = [
        f"Pipeline validation {'passed' if result.passed else 'failed'}: "
        f"{_passed_count(result)}/{len(result.steps)} steps passed"
    ]
    for index, step in enumerate(result.steps, start=1):
        status = "PASS" if step.passed else "FAIL"
        lines.append(f"{index}. {status} {step.name}")
        lines.append(f"   command: {format_command(step.command)}")
        if not step.passed:
            lines.append(f"   returncode: {step.returncode}")
    return "\n".join(lines)


def format_command(command: tuple[str, ...]) -> str:
    return shlex.join(command)


def build_step_name(command: tuple[str, ...]) -> str:
    script = _script_name(command)
    if script == "resolve_qwen_policy":
        label = _value_after(command, "scripts/resolve_qwen_policy.py")
        return f"resolve_qwen_policy {label}" if label else script
    if script == "export_teacher_targets":
        config = _config_stem(command)
        suffixes = [config] if config else []
        if "--dry-run" in command:
            suffixes.append("dry-run")
        return " ".join([script, *suffixes])
    if script == "inspect_targets":
        target = command[-1] if command else ""
        return f"inspect_targets {Path(target).name}" if target else script
    if script == "inspect_prompt_corpus":
        target = command[-1] if command else ""
        return f"inspect_prompt_corpus {Path(target).name}" if target else script
    if script == "create_prompt_manifest":
        target = _value_after(command, "scripts/create_prompt_manifest.py")
        return f"create_prompt_manifest {Path(target).name}" if target else script
    if script == "train_student_smoke":
        architecture = (
            _option_value(command, "--student-architecture") or "tiny_student"
        )
        return f"train_student_smoke {architecture}"
    if script == "run_distill_stage":
        optimizer = _option_value(command, "--optimizer")
        if optimizer:
            return f"run_distill_stage {optimizer}"
        if "--track-run" in command:
            return "run_distill_stage tracked"
        if "--resume-from" in command:
            return "run_distill_stage checkpoint-resume"
        if "--checkpoint-out" in command:
            return "run_distill_stage checkpoint-save"
        targets = _option_value(command, "--targets")
        return f"run_distill_stage {Path(targets).name}" if targets else script
    if script == "generate_from_checkpoint":
        return "generate_from_checkpoint smoke"
    if script == "evaluate_checkpoint":
        return "evaluate_checkpoint regression_smoke"
    if script == "tpu_distill_smoke" and "--require-tpu" in command:
        return "tpu_distill_smoke require-tpu"
    return script


def _passed_count(result: PipelineValidationResult) -> int:
    return sum(1 for step in result.steps if step.passed)


def _script_name(command: tuple[str, ...]) -> str:
    for part in command:
        if part.startswith("scripts/") and part.endswith(".py"):
            return Path(part).stem
    return Path(command[0]).name if command else "<empty>"


def _config_stem(command: tuple[str, ...]) -> str | None:
    config = _option_value(command, "--config")
    return Path(config).stem if config else None


def _option_value(command: tuple[str, ...], option: str) -> str | None:
    try:
        index = command.index(option)
    except ValueError:
        return None
    value_index = index + 1
    if value_index >= len(command):
        return None
    return command[value_index]


def _value_after(command: tuple[str, ...], value: str) -> str | None:
    try:
        index = command.index(value)
    except ValueError:
        return None
    value_index = index + 1
    if value_index >= len(command):
        return None
    return command[value_index]
