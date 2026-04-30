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


def test_run_distill_stage_checkpoint_cli_resume(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path)
    first_dir = tmp_path / "checkpoints" / "cli_first"
    second_dir = tmp_path / "checkpoints" / "cli_second"

    first = subprocess.run(
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
            "--checkpoint-out",
            str(first_dir),
            "--checkpoint-overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    assert "checkpoint_out:" in first.stdout
    assert "end_step: 1" in first.stdout

    second = subprocess.run(
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
            "2",
            "--resume-from",
            str(first_dir),
            "--checkpoint-out",
            str(second_dir),
            "--checkpoint-overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert second.returncode == 0, second.stderr
    assert "resume_from:" in second.stdout
    assert "start_step: 1" in second.stdout
    assert "end_step: 3" in second.stdout


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
