from __future__ import annotations

import platform
import socket
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from qrwkv_xla.datasets import TargetBatch
from qrwkv_xla.students import TinyStudent, TinyStudentConfig
from qrwkv_xla.tracking.base import ExperimentTracker, TrackerConfig
from qrwkv_xla.tracking.git import get_environment_metadata, get_git_metadata
from qrwkv_xla.tracking.local import LocalExperimentTracker
from qrwkv_xla.tracking.reports import (
    P47_DEFAULT_ARTIFACT_DIR,
    write_tracking_smoke_reports,
)
from qrwkv_xla.trainers import TrainState, batch_to_jax, make_train_step

TrackingMode = Literal["local", "wandb-offline", "wandb-online"]


@dataclass(frozen=True)
class TrackingSmokeConfig:
    phase: str = "P47"
    tracking: TrackingMode = "local"
    out: Path = P47_DEFAULT_ARTIFACT_DIR
    project: str = "qrwkv-xla"
    entity: str | None = None
    run_name: str | None = "p47-experiment-tracking-smoke"
    overwrite: bool = False
    steps: int = 2
    seed: int = 47
    learning_rate: float = 1e-2
    batch_size: int = 2
    sequence_length: int = 3
    vocab_size: int = 8
    hidden_size: int = 4
    num_layers: int = 2


def run_tracking_smoke(
    config: TrackingSmokeConfig,
    *,
    command: list[str] | None = None,
    repo_dir: str | Path = ".",
) -> dict[str, Any]:
    if config.steps <= 0:
        raise ValueError("steps must be > 0")
    tracker = create_tracker(config)
    metadata = build_experiment_metadata(
        config=config,
        command=command or sys.argv,
        repo_dir=repo_dir,
        artifact_root=tracker.info.artifact_root,
    )
    config_payload = _config_payload(config)
    tracker.start(metadata=metadata, config=config_payload)
    step_metrics = _run_tiny_steps(config=config, tracker=tracker)
    marker_path = _write_checkpoint_marker(tracker.info.run_dir)
    tracker.log_artifact(tracker.info.config_path, kind="config", name="config")
    tracker.log_artifact(tracker.info.metrics_path, kind="metrics", name="metrics")
    tracker.log_artifact(
        marker_path,
        kind="smoke-output",
        name="tiny-checkpoint-marker",
    )

    summary = _summary(config=config, step_metrics=step_metrics)
    tracker.finish(summary)
    tracker.log_artifact(tracker.info.summary_path, kind="summary", name="summary")
    finalized_summary = tracker.finish(
        {
            **summary,
            "metrics_logged_count": len(step_metrics),
            "artifacts_logged_count": _artifact_count(tracker),
            "summary_written": tracker.info.summary_path.is_file(),
        }
    )

    overall_status = "pass" if summary["final_loss_is_finite"] else "fail"
    report = {
        "phase": config.phase,
        "overall_status": overall_status,
        "status": "passed" if overall_status == "pass" else "failed",
        "tracking_mode": config.tracking,
        "artifact_path": str(config.out),
        "local_run_id": tracker.info.run_id,
        "commit": metadata["repo_commit"],
        "git_dirty": metadata["git_dirty"],
        "backend": metadata["backend"],
        "device_count": metadata["device_count"],
        "steps": config.steps,
        "final_loss": summary["final_loss"],
        "loss_is_finite": summary["final_loss_is_finite"],
        "metrics_logged_count": len(step_metrics),
        "artifacts_logged_count": _artifact_count(tracker),
        "summary_written": tracker.info.summary_path.is_file(),
        "wandb_status": _wandb_status(config.tracking, overall_status),
        "limitations": _limitations(),
        "metadata": metadata,
        "config": config_payload,
        "summary": finalized_summary,
        "paths": {
            "run_metadata": str(tracker.info.metadata_path),
            "config": str(tracker.info.config_path),
            "metrics": str(tracker.info.metrics_path),
            "summary": str(tracker.info.summary_path),
            "artifacts_manifest": str(tracker.info.artifacts_manifest_path),
            "checkpoint_marker": str(marker_path),
        },
    }
    report_paths = write_tracking_smoke_reports(
        report,
        out_dir=config.out,
        overwrite=config.overwrite,
    )
    report["paths"]["report_json"] = str(report_paths["json"])
    report["paths"]["report_markdown"] = str(report_paths["markdown"])
    write_tracking_smoke_reports(
        report,
        out_dir=config.out,
        overwrite=True,
    )
    return report


def create_tracker(config: TrackingSmokeConfig) -> ExperimentTracker:
    tracker_config = TrackerConfig(
        mode=config.tracking,
        project=config.project,
        entity=config.entity,
        run_name=config.run_name,
        artifact_root=config.out,
        overwrite=config.overwrite,
        tags=("p47", "tracking-smoke"),
        notes=("Tiny deterministic tracking smoke; not a training benchmark.",),
    )
    if config.tracking == "local":
        return LocalExperimentTracker(tracker_config)
    if config.tracking in {"wandb-offline", "wandb-online"}:
        from qrwkv_xla.tracking.wandb_adapter import create_wandb_tracker

        return create_wandb_tracker(tracker_config)
    raise ValueError(f"unsupported tracking mode: {config.tracking}")


def build_experiment_metadata(
    *,
    config: TrackingSmokeConfig,
    command: list[str],
    repo_dir: str | Path,
    artifact_root: Path,
) -> dict[str, Any]:
    git = get_git_metadata(repo_dir)
    environment = get_environment_metadata()
    devices = jax.devices()
    local_devices = jax.local_devices()
    hostname = socket.gethostname() or None
    return {
        "phase": config.phase,
        "created_at_utc": _utc_now(),
        "repo_commit": git.get("commit") or "unknown",
        "git": git,
        "git_dirty": classify_git_dirty(git),
        "python_version": platform.python_version() or "unknown",
        "jax": environment,
        "jax_version": environment.get("jax_version") or "unknown",
        "backend": environment.get("jax_backend") or jax.default_backend() or "unknown",
        "default_backend": jax.default_backend() or "unknown",
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "device_kinds": (
            sorted({str(device.device_kind) for device in devices}) or ["unknown"]
        ),
        "device_platforms": (
            sorted({str(device.platform) for device in devices}) or ["unknown"]
        ),
        "local_device_kinds": sorted(
            {str(device.device_kind) for device in local_devices}
        )
        or ["unknown"],
        "hostname": hostname or "unknown",
        "command": list(command),
        "script_name": Path(command[0]).name if command else "unknown",
        "tracking_mode": config.tracking,
        "artifact_root": str(artifact_root),
    }


def classify_git_dirty(git: dict[str, Any]) -> str:
    if not git.get("available"):
        return "unknown"
    status = str(git.get("status_short") or "")
    if not status.strip():
        return "clean"
    lines = [line for line in status.splitlines() if line.strip()]
    if all(line.startswith("??") for line in lines):
        return "untracked_artifacts_only"
    return "tracked_dirty"


def _run_tiny_steps(
    *,
    config: TrackingSmokeConfig,
    tracker: ExperimentTracker,
) -> list[dict[str, Any]]:
    student = TinyStudent(
        TinyStudentConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
        )
    )
    state = TrainState(
        params=student.init_params(jax.random.PRNGKey(config.seed)),
        step=0,
        learning_rate=config.learning_rate,
    )
    train_step = make_train_step(student.apply)
    batch = batch_to_jax(_target_batch(config))
    metrics_by_step: list[dict[str, Any]] = []
    tokens_per_step = config.batch_size * config.sequence_length
    for step in range(1, config.steps + 1):
        state, raw_metrics = train_step(state, batch)
        loss = float(raw_metrics["loss"])
        metrics = {
            "step": step,
            "train/loss": loss,
            "train/loss_is_finite": bool(jnp.isfinite(raw_metrics["loss"])),
            "train/tokens_seen": step * tokens_per_step,
            "train/examples_seen": step * config.batch_size,
            "train/learning_rate": float(state.learning_rate),
        }
        tracker.log_metrics(metrics, step=step)
        metrics_by_step.append(metrics)
    return metrics_by_step


def _target_batch(config: TrackingSmokeConfig) -> TargetBatch:
    input_ids = np.arange(
        config.batch_size * config.sequence_length,
        dtype=np.int32,
    ).reshape(config.batch_size, config.sequence_length)
    input_ids = input_ids % config.vocab_size
    return TargetBatch(
        input_ids=input_ids,
        attention_mask=np.ones_like(input_ids, dtype=np.int32),
        loss_mask=np.ones_like(input_ids, dtype=np.int32),
        hidden_states=np.zeros(
            (
                config.batch_size,
                config.num_layers,
                config.sequence_length,
                config.hidden_size,
            ),
            dtype=np.float32,
        ),
    )


def _summary(
    *,
    config: TrackingSmokeConfig,
    step_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    final = step_metrics[-1]
    return {
        "phase": config.phase,
        "status": "completed" if final["train/loss_is_finite"] else "failed",
        "steps": config.steps,
        "final_loss": final["train/loss"],
        "final_loss_is_finite": final["train/loss_is_finite"],
        "tokens_seen": final["train/tokens_seen"],
        "examples_seen": final["train/examples_seen"],
        "metrics_logged_count": len(step_metrics),
        "finished_at_utc": _utc_now(),
    }


def _config_payload(config: TrackingSmokeConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["out"] = str(config.out)
    return payload


def _write_checkpoint_marker(run_dir: Path) -> Path:
    marker_path = run_dir / "tiny_checkpoint_marker.txt"
    marker_path.write_text(
        "P47 tiny tracking smoke marker only; not a real checkpoint.\n",
        encoding="utf-8",
    )
    return marker_path


def _artifact_count(tracker: ExperimentTracker) -> int:
    manifest_path = tracker.info.artifacts_manifest_path
    if not manifest_path.is_file():
        return 0
    import json

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return len(payload.get("artifacts", []))


def _wandb_status(tracking_mode: TrackingMode, overall_status: str) -> str:
    if tracking_mode == "local":
        return "skipped"
    suffix = "pass" if overall_status == "pass" else "fail"
    if tracking_mode == "wandb-offline":
        return f"offline_{suffix}"
    return f"online_{suffix}"


def _limitations() -> list[str]:
    return [
        "P47 proves tracking smoke plumbing only.",
        "P47 does not prove real training.",
        "P47 does not prove long-run stability.",
        "P47 does not require online WandB.",
        "Local tracker is the source of truth for CI and offline runs.",
        "P47 closes the planned non-training gap ladder through P47.",
    ]


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
