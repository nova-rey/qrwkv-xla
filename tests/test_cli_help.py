from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script_name",
    [
        "create_fake_targets.py",
        "inspect_targets.py",
        "export_teacher_targets.py",
        "train_student_smoke.py",
        "run_distill_stage.py",
        "xla_inspect.py",
        "tpu_distill_smoke.py",
        "smoke_tpu.py",
        "validate_pipeline.py",
        "inspect_prompt_corpus.py",
        "create_prompt_manifest.py",
        "split_prompt_corpus.py",
    ],
)
def test_script_help(script_name: str) -> None:
    script = ROOT / "scripts" / script_name

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
