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
SCRIPT = ROOT / "scripts" / "train_student_smoke.py"


def test_cli_trains_student_on_existing_bundle(tmp_path: Path) -> None:
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
            num_shards=2,
        ),
    )
    FakeTeacherExporter().export(
        ExportRequest(config=config, output_dir=config.runtime.output_dir)
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--targets",
            str(config.runtime.output_dir),
            "--max-steps",
            "2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "initial_loss:" in result.stdout
    assert "final_loss:" in result.stdout
    assert "steps: 2" in result.stdout


def test_cli_trains_rwkv7_reference_student_on_fake_bundle(tmp_path: Path) -> None:
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
            output_dir=tmp_path / "rwkv7_bundle",
            batch_size=2,
            num_shards=2,
        ),
    )
    FakeTeacherExporter().export(
        ExportRequest(config=config, output_dir=config.runtime.output_dir)
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--targets",
            str(config.runtime.output_dir),
            "--student-architecture",
            "rwkv7_reference",
            "--max-steps",
            "2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "rwkv7_reference" in result.stdout
    assert "initial_loss:" in result.stdout
    assert "final_loss:" in result.stdout
    assert "steps: 2" in result.stdout
