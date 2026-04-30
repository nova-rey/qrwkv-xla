from __future__ import annotations

import json
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


def test_run_distill_stage_cli_tracking_flags(tmp_path: Path) -> None:
    bundle_dir = _fake_bundle(tmp_path)
    run_root = tmp_path / "runs"
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
            "--track-run",
            "--run-root",
            str(run_root),
            "--run-name",
            "CLI Tracking",
            "--run-tag",
            "cli",
            "--run-note",
            "tracked smoke",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "run_dir:" in result.stdout
    assert "metrics_path:" in result.stdout
    assert "summary_path:" in result.stdout
    run_dir_line = next(
        line for line in result.stdout.splitlines() if line.startswith("run_dir:")
    )
    run_dir = Path(run_dir_line.split(":", 1)[1].strip())

    assert run_dir.is_dir()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "checkpoints" / "final" / "checkpoint.json").is_file()
    run_payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_payload["run_name"] == "CLI Tracking"
    assert run_payload["tags"] == ["cli"]


def test_run_distill_stage_help_includes_tracking_flags() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--track-run" in result.stdout
    assert "--run-root" in result.stdout
    assert "--run-tag" in result.stdout


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
