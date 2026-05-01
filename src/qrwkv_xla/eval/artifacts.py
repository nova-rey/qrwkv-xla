from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from qrwkv_xla.eval.sanity import SanitySummary, sanity_summary_to_dict
from qrwkv_xla.generation.artifacts import GenerationRecord, write_generation_jsonl


def write_eval_json(summary: dict[str, Any], path: str | Path) -> Path:
    return _write_json(summary, path)


def write_sanity_json(summary: SanitySummary, path: str | Path) -> Path:
    return _write_json(sanity_summary_to_dict(summary), path)


def read_generation_jsonl(path: str | Path) -> tuple[GenerationRecord, ...]:
    records: list[GenerationRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        records.append(
            GenerationRecord(
                prompt_id=str(payload["prompt_id"]),
                prompt_text=str(payload["prompt_text"]),
                prompt_token_ids=tuple(int(v) for v in payload["prompt_token_ids"]),
                generated_token_ids=tuple(
                    int(v) for v in payload["generated_token_ids"]
                ),
                full_token_ids=tuple(int(v) for v in payload["full_token_ids"]),
                decoded_text=str(payload["decoded_text"]),
                metadata=dict(payload.get("metadata", {})),
            )
        )
    return tuple(records)


def read_eval_snapshot(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    eval_path = root / "eval.json"
    sanity_path = root / "sanity.json"
    return {
        "output_dir": str(root),
        "eval": json.loads(eval_path.read_text(encoding="utf-8"))
        if eval_path.exists()
        else {},
        "generations": read_generation_jsonl(root / "generations.jsonl"),
        "sanity": json.loads(sanity_path.read_text(encoding="utf-8"))
        if sanity_path.exists()
        else {},
    }


def write_generation_records(
    records: list[GenerationRecord] | tuple[GenerationRecord, ...],
    path: str | Path,
) -> Path:
    return write_generation_jsonl(records, path)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    return value


def _write_json(payload: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
