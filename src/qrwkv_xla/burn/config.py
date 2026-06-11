from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FirstSeriousBurnConfig:
    phase: str = "P112"
    run_name: str = "p112_first_serious_burn"
    mode: str = "dry_run"
    teacher_model_id: str | None = None
    local_files_only: bool = True
    allow_downloads: bool = False
    architecture_id: str = "tiny_debug"
    runtime: str = "reference"
    target_store_path: str | None = None
    teacher_textbook_path: str | None = None
    output_dir: str = "artifacts/p112_first_serious_burn/dry_run"
    readiness_report_path: str | None = None
    max_steps: int = 1
    batch_size: int = 1
    allow_textbook_reuse: bool = False
    example_sharding: str = "auto"
    distributed_sync: str = "auto"
    sequence_length: int = 8
    checkpoint_every_steps: int = 1
    eval_every_steps: int = 1
    require_readiness_pass: bool = True
    accepted_warnings: tuple[str, ...] = ()

    def to_report(self) -> dict[str, Any]:
        return asdict(self)


def default_first_serious_burn_config(
    *,
    output_dir: str | Path = "artifacts/p112_first_serious_burn/dry_run",
    mode: str = "dry_run",
) -> FirstSeriousBurnConfig:
    return FirstSeriousBurnConfig(mode=mode, output_dir=str(output_dir))


def load_first_serious_burn_config(path: str | Path) -> FirstSeriousBurnConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "accepted_warnings" in payload:
        payload["accepted_warnings"] = tuple(payload["accepted_warnings"])
    return FirstSeriousBurnConfig(**payload)


def write_first_serious_burn_config(
    config: FirstSeriousBurnConfig,
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config.to_report(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
