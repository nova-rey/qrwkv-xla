from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMANDS = [
    [sys.executable, "-m", "compileall", "src", "scripts", "tests"],
    [sys.executable, "scripts/print_env.py"],
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
        "scripts/inspect_targets.py",
        "artifacts/teacher_targets/fake_export",
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
