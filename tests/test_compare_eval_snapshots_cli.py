from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from qrwkv_xla.generation.artifacts import GenerationRecord, write_generation_jsonl

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_eval_snapshots.py"


def test_compare_eval_snapshots_cli_writes_json(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    out = tmp_path / "comparison.json"
    record = GenerationRecord(
        prompt_id="p0",
        prompt_text="Prompt",
        prompt_token_ids=(1,),
        generated_token_ids=(2,),
        full_token_ids=(1, 2),
        decoded_text="same",
    )
    write_generation_jsonl([record], baseline / "generations.jsonl")
    write_generation_jsonl([record], candidate / "generations.jsonl")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--out",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["same_count"] == 1
    assert payload["changed_count"] == 0
