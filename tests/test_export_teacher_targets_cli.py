from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from qrwkv_xla.targets import inspect_target_bundle
from qrwkv_xla.targets.shards import read_shard
from qrwkv_xla.targets.store import shard_path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_teacher_targets.py"
CONFIG = ROOT / "configs" / "teacher_export_stub.yaml"


def test_cli_exports_bundle(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_bundle"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(CONFIG),
            "--out",
            str(out_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = inspect_target_bundle(out_dir)
    assert summary["shard_count"] == 2
    assert "output_dir:" in result.stdout
    assert "teacher_policy_label: Qwen3.latest" in result.stdout


def test_cli_unknown_backend_fails(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_bad_backend"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(CONFIG),
            "--out",
            str(out_dir),
            "--backend",
            "unknown",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert (
        "Unknown teacher exporter backend" in result.stderr
        or "runtime.exporter_backend" in result.stderr
    )


def test_cli_include_logits_writes_logits(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_logits"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(CONFIG),
            "--out",
            str(out_dir),
            "--include-logits",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    shard = read_shard(shard_path(out_dir, 0))
    assert "logits" in shard
