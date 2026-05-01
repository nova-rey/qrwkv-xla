from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerationRecord:
    prompt_id: str
    prompt_text: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    full_token_ids: tuple[int, ...]
    decoded_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def write_generation_jsonl(
    records: list[GenerationRecord] | tuple[GenerationRecord, ...],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(asdict(record), sort_keys=True, ensure_ascii=False)
        for record in records
    ]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path


def write_generation_summary(summary: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path
