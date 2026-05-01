from __future__ import annotations

import json
from pathlib import Path

from qrwkv_xla.generation import (
    GenerationRecord,
    write_generation_jsonl,
    write_generation_summary,
)


def test_write_generation_artifacts(tmp_path: Path) -> None:
    records = [
        GenerationRecord(
            prompt_id="p0",
            prompt_text="Hello",
            prompt_token_ids=(1, 2),
            generated_token_ids=(3,),
            full_token_ids=(1, 2, 3),
            decoded_text="Hello!",
            metadata={"tokenizer": "smoke"},
        )
    ]

    jsonl_path = write_generation_jsonl(records, tmp_path / "generations.jsonl")
    summary_path = write_generation_summary(
        {"passed": True, "prompt_count": 1},
        tmp_path / "summary.json",
    )

    loaded_record = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
    loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded_record["prompt_id"] == "p0"
    assert loaded_record["generated_token_ids"] == [3]
    assert loaded_summary["passed"] is True
    assert loaded_summary["prompt_count"] == 1
