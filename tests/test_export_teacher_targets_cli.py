from __future__ import annotations

import os
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
    assert "loss_mask" in shard


def test_cli_out_override_remains_relative_to_cwd(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "teacher.yaml"
    config.write_text(
        """
targets:
  sequence_length: 4
  hidden_size: 4
  num_layers: 1
  vocab_size: 16
runtime:
  batch_size: 1
  num_shards: 1
  output_dir: config_relative_bundle
""",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config),
            "--out",
            "cli_relative_bundle",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "cli_relative_bundle" / "manifest.json").is_file()
    assert not (config_dir / "cli_relative_bundle").exists()
    assert not (config_dir / "config_relative_bundle").exists()
