from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qrwkv_xla.tracking.git import get_environment_metadata, get_git_metadata
from qrwkv_xla.tracking.json_io import to_jsonable, write_json

SCHEMA_VERSION = "0.1"
CREATED_BY = "qrwkv_xla.tracking.run"
DEFAULT_RUN_ROOT = Path("runs")


@dataclass(frozen=True)
class RunMetadata:
    schema_version: str
    run_id: str
    run_name: str | None
    created_at_utc: str
    command: list[str]
    git: dict[str, Any]
    environment: dict[str, Any]
    distillation: dict[str, Any]
    teacher_target: dict[str, Any]
    student: dict[str, Any]
    checkpoint: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    created_by: str = CREATED_BY


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    run_json: Path
    metrics_jsonl: Path
    summary_json: Path
    checkpoints_dir: Path

    @property
    def default_final_checkpoint(self) -> Path:
        return self.checkpoints_dir / "final"


@dataclass(frozen=True)
class RunContext:
    metadata: RunMetadata
    paths: RunPaths


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    status: str
    started_at_utc: str
    finished_at_utc: str
    summary: dict[str, Any]


def make_run_id(
    *,
    stage: int,
    student_architecture: str,
    run_name: str | None = None,
    now: datetime | None = None,
) -> str:
    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%d_%H%M%S")
    if run_name:
        slug = _slugify(run_name)
    else:
        slug = _slugify(f"distill_stage{stage}_{student_architecture}")
    return f"{timestamp}_{slug}"


create_run_id = make_run_id


def create_run_context(
    *,
    run_root: str | Path,
    stage: int,
    student_architecture: str,
    run_name: str | None,
    command: list[str],
    git: dict[str, Any],
    environment: dict[str, Any],
    distillation: dict[str, Any],
    teacher_target: dict[str, Any],
    student: dict[str, Any],
    checkpoint: dict[str, Any],
    tags: list[str] | None = None,
    notes: list[str] | None = None,
    overwrite: bool = False,
    now: datetime | None = None,
) -> RunContext:
    run_id = make_run_id(
        stage=stage,
        student_architecture=student_architecture,
        run_name=run_name,
        now=now,
    )
    root = Path(run_root)
    validate_run_root(root)
    run_dir = root / run_id
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=overwrite)
    paths = RunPaths(
        run_dir=run_dir,
        run_json=run_dir / "run.json",
        metrics_jsonl=run_dir / "metrics.jsonl",
        summary_json=run_dir / "summary.json",
        checkpoints_dir=run_dir / "checkpoints",
    )
    metadata = RunMetadata(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        run_name=run_name,
        created_at_utc=_utc_now(now),
        command=list(command),
        git=to_jsonable(git),
        environment=to_jsonable(environment),
        distillation=to_jsonable(distillation),
        teacher_target=to_jsonable(teacher_target),
        student=to_jsonable(student),
        checkpoint=to_jsonable(checkpoint),
        tags=list(tags or []),
        notes=list(notes or []),
    )
    return RunContext(metadata=metadata, paths=paths)


def build_run_metadata(
    *,
    run_id: str,
    run_name: str | None,
    paths: RunPaths,
    config: Any,
    tags: list[str] | None = None,
    notes: list[str] | None = None,
    repo_dir: str | Path = ".",
    created_at_utc: str | None = None,
) -> RunMetadata:
    return RunMetadata(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        run_name=run_name,
        created_at_utc=created_at_utc or _utc_now(),
        command=[],
        git=get_git_metadata(repo_dir),
        environment=get_environment_metadata(),
        distillation={"config": to_jsonable(config), "paths": to_jsonable(paths)},
        teacher_target={},
        student={},
        checkpoint={},
        tags=list(tags or []),
        notes=list(notes or []),
    )


def write_run_metadata(context: RunContext | RunMetadata) -> Path:
    if isinstance(context, RunMetadata):
        raise ValueError(
            "write_run_metadata expects a RunContext so paths are available"
        )
    return write_json(context.paths.run_json, context.metadata)


def write_run_summary(*, context: RunContext, summary: dict[str, Any]) -> Path:
    payload = RunSummary(
        run_id=context.metadata.run_id,
        status=str(summary.get("status", "completed")),
        started_at_utc=str(
            summary.get("started_at_utc", context.metadata.created_at_utc)
        ),
        finished_at_utc=str(summary.get("finished_at_utc", _utc_now())),
        summary=to_jsonable(summary),
    )
    return write_json(context.paths.summary_json, payload)


def validate_run_root(path: Path) -> None:
    if path.name != "runs" and "runs" not in path.parts:
        raise ValueError("run_root must be runs/ or a path under a runs/ directory")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "run"


def _utc_now(now: datetime | None = None) -> str:
    return (
        (now or datetime.now(UTC))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
