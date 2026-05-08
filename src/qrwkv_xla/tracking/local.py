from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from qrwkv_xla.tracking.base import ArtifactRecord, ExperimentRunInfo, TrackerConfig
from qrwkv_xla.tracking.json_io import append_jsonl, to_jsonable, write_json


class LocalExperimentTracker:
    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self._info = _build_run_info(config)
        self._artifact_records: list[ArtifactRecord] = []

    @property
    def info(self) -> ExperimentRunInfo:
        return self._info

    def start(self, *, metadata: dict[str, Any], config: dict[str, Any]) -> None:
        if self.info.run_dir.exists() and not self.config.overwrite:
            raise FileExistsError(f"tracking run directory exists: {self.info.run_dir}")
        if self.info.run_dir.exists():
            shutil.rmtree(self.info.run_dir)
        self.info.files_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.info.metadata_path, metadata)
        write_json(self.info.config_path, config)
        write_json(self.info.artifacts_manifest_path, {"artifacts": []})

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        payload = {"step": int(step), **to_jsonable(metrics)}
        append_jsonl(self.info.metrics_path, payload)

    def log_artifact(
        self,
        path: str | Path,
        *,
        kind: str,
        name: str | None = None,
    ) -> ArtifactRecord:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"artifact file does not exist: {source}")
        destination = self.info.files_dir / source.name
        if destination.resolve() != source.resolve():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        record = _artifact_record(
            destination,
            name=name or source.stem,
            kind=kind,
            root=self.info.run_dir,
            source_path=source,
        )
        self._artifact_records = [
            existing
            for existing in self._artifact_records
            if existing.path != record.path or existing.kind != record.kind
        ]
        self._artifact_records.append(record)
        self._write_manifest()
        return record

    def finish(self, summary: dict[str, Any]) -> dict[str, Any]:
        payload = to_jsonable(summary)
        payload["artifacts"] = [
            to_jsonable(record) for record in self._artifact_records
        ]
        write_json(self.info.summary_path, payload)
        self._write_manifest()
        return payload

    def _write_manifest(self) -> None:
        write_json(
            self.info.artifacts_manifest_path,
            {"artifacts": [to_jsonable(record) for record in self._artifact_records]},
        )


def _build_run_info(config: TrackerConfig) -> ExperimentRunInfo:
    artifact_root = Path(config.artifact_root)
    run_id = "local_run"
    run_dir = artifact_root / run_id
    return ExperimentRunInfo(
        run_id=run_id,
        run_name=config.run_name,
        mode=config.mode,
        artifact_root=artifact_root,
        run_dir=run_dir,
        metadata_path=run_dir / "run_metadata.json",
        config_path=run_dir / "config.json",
        metrics_path=run_dir / "metrics.jsonl",
        summary_path=run_dir / "summary.json",
        artifacts_manifest_path=run_dir / "artifacts_manifest.json",
        files_dir=run_dir / "files",
    )


def _artifact_record(
    path: Path,
    *,
    name: str,
    kind: str,
    root: Path,
    source_path: Path | None = None,
) -> ArtifactRecord:
    return ArtifactRecord(
        name=name,
        path=path.relative_to(root).as_posix(),
        kind=kind,
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
        source_path=str(source_path) if source_path is not None else None,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
