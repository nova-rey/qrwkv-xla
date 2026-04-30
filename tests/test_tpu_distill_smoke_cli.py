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
TPU_SMOKE_SCRIPT = ROOT / "scripts" / "tpu_distill_smoke.py"
XLA_INSPECT_SCRIPT = ROOT / "scripts" / "xla_inspect.py"


def test_tpu_distill_smoke_cli_runs_cpu_safe(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(TPU_SMOKE_SCRIPT),
            "--targets",
            str(bundle_dir),
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
    assert "backend:" in result.stdout
    assert "has_tpu:" in result.stdout
    assert "initial_loss:" in result.stdout
    assert "final_loss:" in result.stdout


def test_xla_inspect_cli_runs() -> None:
    result = subprocess.run(
        [sys.executable, str(XLA_INSPECT_SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "jax_available:" in result.stdout


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
