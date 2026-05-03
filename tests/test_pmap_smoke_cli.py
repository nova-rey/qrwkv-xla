from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import jax
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_pmap_distill_smoke_skips_when_devices_unavailable() -> None:
    result = _run(
        "scripts/pmap_distill_smoke.py",
        "--config",
        "configs/distill_stage1_attention_pmap_smoke.yaml",
        "--min-device-count",
        "999",
    )

    assert result.returncode == 0, result.stderr
    assert "SKIPPED:" in result.stdout


def test_pmap_distill_smoke_hard_fails_when_required() -> None:
    result = _run(
        "scripts/pmap_distill_smoke.py",
        "--config",
        "configs/distill_stage1_attention_pmap_smoke.yaml",
        "--min-device-count",
        "999",
        "--require-multiple-devices",
    )

    assert result.returncode != 0
    assert "SKIPPED:" in result.stderr or "SKIPPED:" in result.stdout


def test_pmap_lm_smoke_skips_when_devices_unavailable() -> None:
    result = _run(
        "scripts/pmap_lm_smoke.py",
        "--config",
        "configs/lm_stage3_pmap_smoke.yaml",
        "--min-device-count",
        "999",
    )

    assert result.returncode == 0, result.stderr
    assert "SKIPPED:" in result.stdout


@pytest.mark.skipif(
    jax.local_device_count() < 2, reason="requires >=2 local JAX devices"
)
def test_pmap_distill_smoke_runs_when_multiple_devices_are_available() -> None:
    export = _run(
        "scripts/export_teacher_targets.py",
        "--config",
        "configs/teacher_export_stub_attention.yaml",
    )
    assert export.returncode == 0, export.stderr

    result = _run(
        "scripts/pmap_distill_smoke.py",
        "--config",
        "configs/distill_stage1_attention_pmap_smoke.yaml",
        "--require-multiple-devices",
        "--min-device-count",
        "2",
    )

    assert result.returncode == 0, result.stderr
    final_loss_line = next(
        line for line in result.stdout.splitlines() if line.startswith("final_loss:")
    )
    final_loss = float(final_loss_line.split(":", 1)[1].strip())
    assert math.isfinite(final_loss)
