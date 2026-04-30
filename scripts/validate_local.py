from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMANDS = [
    [sys.executable, "-m", "compileall", "src", "scripts", "tests"],
    [sys.executable, "scripts/print_env.py"],
    [sys.executable, "scripts/xla_inspect.py"],
    [sys.executable, "scripts/smoke_cpu.py"],
    [sys.executable, "scripts/smoke_tpu.py"],
    [
        sys.executable,
        "scripts/export_teacher_targets.py",
        "--config",
        "configs/teacher_export_stub.yaml",
    ],
    [
        sys.executable,
        "scripts/resolve_qwen_policy.py",
        "Qwen3.latest",
        "--allow-unresolved",
    ],
    [
        sys.executable,
        "scripts/export_teacher_targets.py",
        "--config",
        "configs/teacher_export_qwen_dryrun.yaml",
        "--dry-run",
        "--resolve-qwen-policy",
        "--allow-unresolved-policy",
    ],
    [
        sys.executable,
        "scripts/inspect_targets.py",
        "artifacts/teacher_targets/fake_export",
    ],
    [
        sys.executable,
        "scripts/train_student_smoke.py",
        "--targets",
        "artifacts/teacher_targets/fake_export",
        "--max-steps",
        "2",
    ],
    [
        sys.executable,
        "scripts/train_student_smoke.py",
        "--targets",
        "artifacts/teacher_targets/fake_export",
        "--student-architecture",
        "rwkv7_reference",
        "--max-steps",
        "2",
    ],
    [
        sys.executable,
        "scripts/run_distill_stage.py",
        "--config",
        "configs/distill_stage0_stub.yaml",
        "--max-steps",
        "2",
    ],
    [
        sys.executable,
        "scripts/tpu_distill_smoke.py",
        "--targets",
        "artifacts/teacher_targets/fake_export",
        "--max-steps",
        "2",
    ],
    [sys.executable, "-m", "pytest"],
    [sys.executable, "-m", "ruff", "check", "."],
    [sys.executable, "-m", "ruff", "format", "--check", "."],
]


def _format_command(command: list[str]) -> str:
    return " ".join(command)


def main() -> None:
    for command in COMMANDS:
        print(f"+ {_format_command(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
