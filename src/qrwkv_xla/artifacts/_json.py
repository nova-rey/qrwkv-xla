from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing JSON file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def require_fields(
    payload: dict[str, Any],
    fields: tuple[str, ...],
    *,
    source: str,
) -> list[str]:
    return [
        f"{source} missing required field: {field}"
        for field in fields
        if field not in payload
    ]


def int_value(payload: dict[str, Any], key: str) -> int | None:
    if key not in payload:
        return None
    try:
        return int(payload[key])
    except (TypeError, ValueError):
        return None
