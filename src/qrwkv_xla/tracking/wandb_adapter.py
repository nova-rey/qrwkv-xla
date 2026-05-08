from __future__ import annotations

from pathlib import Path
from typing import Any

from qrwkv_xla.tracking.base import ArtifactRecord, ExperimentRunInfo, TrackerConfig
from qrwkv_xla.tracking.local import LocalExperimentTracker


class WandbExperimentTracker:
    """Optional WandB adapter with local files as the source of truth."""

    def __init__(self, config: TrackerConfig) -> None:
        if config.mode not in {"wandb-offline", "wandb-online"}:
            raise ValueError(f"unsupported WandB tracking mode: {config.mode}")
        self.config = config
        self.local = LocalExperimentTracker(config)
        self._wandb: Any | None = None
        self._run: Any | None = None

    @property
    def info(self) -> ExperimentRunInfo:
        return self.local.info

    def start(self, *, metadata: dict[str, Any], config: dict[str, Any]) -> None:
        self.local.start(metadata=metadata, config=config)
        self._wandb = _import_wandb()
        mode = "offline" if self.config.mode == "wandb-offline" else "online"
        self._run = self._wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            name=self.config.run_name,
            mode=mode,
            config=config,
            dir=str(self.info.artifact_root),
            tags=list(self.config.tags),
            notes="\n".join(self.config.notes) if self.config.notes else None,
        )
        self._wandb.config.update({"run_metadata": metadata}, allow_val_change=True)

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        self.local.log_metrics(metrics, step=step)
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def log_artifact(
        self,
        path: str | Path,
        *,
        kind: str,
        name: str | None = None,
    ) -> ArtifactRecord:
        record = self.local.log_artifact(path, kind=kind, name=name)
        if self._wandb is not None:
            artifact = self._wandb.Artifact(name=name or Path(path).stem, type=kind)
            artifact.add_file(str(path))
            self._wandb.log_artifact(artifact)
        return record

    def finish(self, summary: dict[str, Any]) -> dict[str, Any]:
        payload = self.local.finish(summary)
        if self._wandb is not None:
            self._wandb.summary.update(payload)
        if self._run is not None:
            self._run.finish()
        return payload


def create_wandb_tracker(config: TrackerConfig) -> WandbExperimentTracker:
    return WandbExperimentTracker(config)


def is_wandb_available() -> bool:
    try:
        _import_wandb()
    except ImportError:
        return False
    return True


def _import_wandb() -> Any:
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "WandB tracking requested, but wandb is not installed. "
            "Use --tracking local or install the optional wandb package."
        ) from exc
    return wandb
