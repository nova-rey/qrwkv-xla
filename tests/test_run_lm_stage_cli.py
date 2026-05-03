from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_run_lm_stage_cli_smoke(tmp_path: Path) -> None:
    checkpoint_out = tmp_path / "checkpoints" / "lm_cli"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_lm_stage.py",
            "--config",
            "configs/lm_stage3_smoke.yaml",
            "--max-steps",
            "1",
            "--checkpoint-out",
            str(checkpoint_out),
            "--checkpoint-overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "mode: lm_stage3_ce" in completed.stdout
    assert "final_ce_loss:" in completed.stdout
    assert (checkpoint_out / "checkpoint.json").is_file()
