from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from qrwkv_xla.teacher_export import (
    ExportRequest,
    FakeTeacherExporter,
    TeacherExportConfig,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_distill_stage.py"


def test_run_distill_stage_cli_accepts_adamw_flags(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(ROOT / "configs" / "distill_stage0_stub.yaml"),
            "--targets",
            str(bundle_dir),
            "--student-architecture",
            "tiny_student",
            "--max-steps",
            "1",
            "--optimizer",
            "adamw",
            "--learning-rate",
            "0.01",
            "--weight-decay",
            "0.1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "optimizer: adamw" in result.stdout
    assert "learning_rate: 0.01" in result.stdout
    assert "steps: 1" in result.stdout


def _fake_bundle(tmp_path: Path) -> Path:
    config = TeacherExportConfig()
    config = replace(
        config,
        targets=replace(
            config.targets,
            sequence_length=8,
            hidden_size=4,
            num_layers=2,
            vocab_size=32,
        ),
        runtime=replace(
            config.runtime,
            output_dir=tmp_path / "bundle",
            batch_size=2,
            num_shards=1,
        ),
    )
    FakeTeacherExporter().export(
        ExportRequest(config=config, output_dir=config.runtime.output_dir)
    )
    return config.runtime.output_dir
