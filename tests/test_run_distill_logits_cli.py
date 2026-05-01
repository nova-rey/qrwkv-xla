from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_run_distill_logits_cli_config_and_flags() -> None:
    export = subprocess.run(
        [
            sys.executable,
            "scripts/export_teacher_targets.py",
            "--config",
            "configs/teacher_export_stub_logits.yaml",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert export.returncode == 0, export.stderr

    config_run = subprocess.run(
        [
            sys.executable,
            "scripts/run_distill_stage.py",
            "--config",
            "configs/distill_stage0_logits_stub.yaml",
            "--max-steps",
            "2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert config_run.returncode == 0, config_run.stderr
    assert "final_logits_kl:" in config_run.stdout
    assert "logits_kl_enabled: True" in config_run.stdout

    flags_run = subprocess.run(
        [
            sys.executable,
            "scripts/run_distill_stage.py",
            "--config",
            "configs/distill_stage0_logits_stub.yaml",
            "--max-steps",
            "1",
            "--emit-logits",
            "--enable-logits-kl",
            "--hidden-mse-weight",
            "0.5",
            "--logits-kl-weight",
            "0.5",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert flags_run.returncode == 0, flags_run.stderr
    assert "final_logits_kl:" in flags_run.stdout
