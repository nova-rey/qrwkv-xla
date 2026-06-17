from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CANONICAL_PROCESS_INDEX = 0
ARTIFACT_WRITE_POLICY = "process_0_canonical_with_per_process_diagnostics"


def is_canonical_process(process_index: int) -> bool:
    return process_index == CANONICAL_PROCESS_INDEX


def canonical_only_path(
    output_dir: Path,
    filename: str,
    process_index: int,
) -> Path | None:
    if not is_canonical_process(process_index):
        return None
    return output_dir / filename


def expected_canonical_path(output_dir: Path, filename: str) -> Path:
    return output_dir / filename


def per_process_path(
    output_dir: Path,
    stem: str,
    process_index: int,
    *,
    suffix: str = ".json",
) -> Path:
    return output_dir / f"{stem}_process_{process_index}{suffix}"


def expected_per_process_paths(
    output_dir: Path,
    stem: str,
    process_count: int,
    *,
    suffix: str = ".json",
) -> tuple[str, ...]:
    return tuple(
        str(per_process_path(output_dir, stem, process_index, suffix=suffix))
        for process_index in range(process_count)
    )


def write_json_canonical(
    payload: dict[str, Any],
    output_dir: Path,
    filename: str,
    *,
    process_index: int,
) -> Path | None:
    path = canonical_only_path(output_dir, filename, process_index)
    if path is None:
        return None
    _write_json(path, payload)
    return path


def write_json_per_process(
    payload: dict[str, Any],
    output_dir: Path,
    stem: str,
    *,
    process_index: int,
) -> Path:
    path = per_process_path(output_dir, stem, process_index)
    _write_json(path, payload)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
