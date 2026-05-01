from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.generation_test_utils import write_generation_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_checkpoint.py"


def test_evaluate_checkpoint_cli_writes_artifacts(tmp_path: Path) -> None:
    checkpoint_dir = write_generation_checkpoint(tmp_path, emit_logits=True)
    output_dir = tmp_path / "eval_outputs" / "eval_smoke"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--checkpoint",
            str(checkpoint_dir),
            "--config",
            str(ROOT / "configs" / "eval_regression_smoke.yaml"),
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "3",
            "--prompt-limit",
            "2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "eval.json").is_file()
    assert (output_dir / "generations.jsonl").is_file()
    assert (output_dir / "sanity.json").is_file()
    assert "prompt_count: 2" in completed.stdout
