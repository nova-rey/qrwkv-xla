from __future__ import annotations

import json
from pathlib import Path

from qrwkv_xla.eval import load_eval_config, replace_eval_overrides
from qrwkv_xla.eval.harness import run_checkpoint_evaluation
from tests.generation_test_utils import write_generation_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def test_run_checkpoint_evaluation_writes_artifacts(tmp_path: Path) -> None:
    checkpoint_dir = write_generation_checkpoint(tmp_path, emit_logits=True)
    config = replace_eval_overrides(
        load_eval_config(ROOT / "configs" / "eval_regression_smoke.yaml"),
        output_dir=tmp_path / "eval_outputs" / "eval_smoke",
        prompt_limit=2,
        max_new_tokens=3,
    )

    result = run_checkpoint_evaluation(
        checkpoint_dir=checkpoint_dir,
        config=config,
    )

    assert result.prompt_count == 2
    assert result.eval_json_path.is_file()
    assert result.generation_path.is_file()
    assert result.sanity_path.is_file()
    sanity = json.loads(result.sanity_path.read_text(encoding="utf-8"))
    assert sanity["prompt_count"] == 2
