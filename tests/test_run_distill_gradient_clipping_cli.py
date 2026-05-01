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


def test_run_distill_stage_cli_accepts_gradient_clipping_flags(
    tmp_path: Path,
) -> None:
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
            "--max-grad-norm",
            "0.1",
            "--clip-epsilon",
            "0.00001",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "max_grad_norm: 0.1" in result.stdout
    assert "clip_epsilon: 1e-05" in result.stdout
    assert "final_grad_clip_scale:" in result.stdout


def test_run_distill_stage_cli_disable_gradient_clipping_override(
    tmp_path: Path,
) -> None:
    bundle_dir = _fake_logits_bundle(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(ROOT / "configs" / "distill_stage0_adamw_clipped_stub.yaml"),
            "--targets",
            str(bundle_dir),
            "--student-architecture",
            "tiny_student",
            "--max-steps",
            "1",
            "--disable-grad-clipping",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "max_grad_norm: None" in result.stdout
    assert "final_grad_clip_scale: 1.00000000" in result.stdout


def test_run_distill_stage_cli_rejects_conflicting_clipping_flags() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--max-grad-norm",
            "1.0",
            "--disable-grad-clipping",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "conflicts" in result.stderr


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


def _fake_logits_bundle(tmp_path: Path) -> Path:
    config = TeacherExportConfig()
    config = replace(
        config,
        targets=replace(
            config.targets,
            sequence_length=8,
            hidden_size=4,
            num_layers=2,
            vocab_size=512,
            include_logits=True,
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
