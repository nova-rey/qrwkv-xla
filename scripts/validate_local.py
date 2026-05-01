from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMANDS = [
    [sys.executable, "-m", "compileall", "src", "scripts", "tests"],
    [
        sys.executable,
        "scripts/export_teacher_targets.py",
        "--config",
        "configs/teacher_export_stub_logits.yaml",
    ],
    [
        sys.executable,
        "scripts/run_distill_stage.py",
        "--config",
        "configs/distill_stage0_logits_stub.yaml",
        "--max-steps",
        "1",
        "--checkpoint-out",
        "checkpoints/generation_smoke",
        "--checkpoint-overwrite",
    ],
    [
        sys.executable,
        "scripts/generate_from_checkpoint.py",
        "--checkpoint",
        "checkpoints/generation_smoke",
        "--prompt",
        "Hello QRWKV",
        "--max-new-tokens",
        "4",
        "--output-dir",
        "eval_outputs/generation_smoke",
    ],
    [sys.executable, "scripts/validate_pipeline.py"],
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
