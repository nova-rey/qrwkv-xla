from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_run_distill_attention_cli() -> None:
    export = subprocess.run(
        [
            sys.executable,
            "scripts/export_teacher_targets.py",
            "--config",
            "configs/teacher_export_stub_attention.yaml",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "fake_attention_export" in export.stdout

    run = subprocess.run(
        [
            sys.executable,
            "scripts/run_distill_stage.py",
            "--config",
            "configs/distill_stage1_attention_stub.yaml",
            "--max-steps",
            "2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "final_attention_or_mixer" in run.stdout
