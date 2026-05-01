from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.generation_test_utils import write_generation_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_generation_smoke.py"


def test_eval_generation_smoke_cli_writes_artifacts(tmp_path: Path) -> None:
    checkpoint_dir = write_generation_checkpoint(tmp_path, emit_logits=True)
    output_dir = tmp_path / "eval_outputs" / "generation_smoke"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--checkpoint",
            str(checkpoint_dir),
            "--config",
            str(ROOT / "configs" / "generation_smoke.yaml"),
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "4",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "generations.jsonl").is_file()
    assert (output_dir / "summary.json").is_file()
    assert "passed: True" in completed.stdout
