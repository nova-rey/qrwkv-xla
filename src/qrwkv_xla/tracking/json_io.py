from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [to_jsonable(child) for child in value]
    if isinstance(value, list):
        return [to_jsonable(child) for child in value]
    if isinstance(value, set):
        return [to_jsonable(child) for child in sorted(value, key=str)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return to_jsonable(item())
        except (TypeError, ValueError):
            pass

    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "shape": [int(dim) for dim in getattr(value, "shape", ())],
            "dtype": str(value.dtype),
        }

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(child) for child in value]
    return str(value)


def write_json(path: str | Path, payload: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def append_jsonl(path: str | Path, payload: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")
        handle.flush()
    return output_path
