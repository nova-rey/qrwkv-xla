from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ArtifactRecord:
    name: str
    kind: str
    path: str
    size_bytes: int
    sha256: str | None = None
    source_path: str | None = None


@dataclass(frozen=True)
class ExperimentRunInfo:
    run_id: str
    run_name: str | None
    mode: str
    artifact_root: Path
    run_dir: Path
    metadata_path: Path
    config_path: Path
    metrics_path: Path
    summary_path: Path
    artifacts_manifest_path: Path
    files_dir: Path


@dataclass(frozen=True)
class TrackerConfig:
    mode: str = "local"
    project: str = "qrwkv-xla"
    entity: str | None = None
    run_name: str | None = None
    artifact_root: Path = Path("artifacts/p47_experiment_tracking_smoke")
    overwrite: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)


class ExperimentTracker(Protocol):
    @property
    def info(self) -> ExperimentRunInfo: ...

    def start(self, *, metadata: dict[str, Any], config: dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None: ...

    def log_artifact(
        self,
        path: str | Path,
        *,
        kind: str,
        name: str | None = None,
    ) -> ArtifactRecord: ...

    def finish(self, summary: dict[str, Any]) -> dict[str, Any]: ...
