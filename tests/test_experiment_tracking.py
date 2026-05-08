from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from qrwkv_xla.tracking.base import TrackerConfig
from qrwkv_xla.tracking.local import LocalExperimentTracker
from qrwkv_xla.tracking.reports import write_tracking_smoke_reports
from qrwkv_xla.tracking.smoke import TrackingSmokeConfig, run_tracking_smoke

ROOT = Path(__file__).resolve().parents[1]


def test_local_tracker_writes_config_metrics_summary_and_manifest(
    tmp_path: Path,
) -> None:
    tracker = LocalExperimentTracker(
        TrackerConfig(
            artifact_root=tmp_path / "tracking",
            run_name="unit",
            overwrite=True,
        )
    )
    tracker.start(
        metadata={"phase": "P47", "tracking_mode": "local"},
        config={"steps": 1},
    )
    tracker.log_metrics({"train/loss": 1.0, "train/loss_is_finite": True}, step=1)
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("tracked\n", encoding="utf-8")
    record = tracker.log_artifact(artifact, kind="text", name="artifact")
    tracker.finish({"status": "completed", "final_loss": 1.0})

    assert tracker.info.metadata_path.is_file()
    assert tracker.info.config_path.is_file()
    assert tracker.info.metrics_path.is_file()
    assert tracker.info.summary_path.is_file()
    assert tracker.info.artifacts_manifest_path.is_file()
    assert json.loads(tracker.info.config_path.read_text(encoding="utf-8")) == {
        "steps": 1
    }
    metric = json.loads(tracker.info.metrics_path.read_text(encoding="utf-8"))
    assert metric["step"] == 1
    assert metric["train/loss"] == 1.0
    manifest = json.loads(
        tracker.info.artifacts_manifest_path.read_text(encoding="utf-8")
    )
    assert manifest["artifacts"][0]["name"] == "artifact"
    assert manifest["artifacts"][0]["kind"] == "text"
    assert manifest["artifacts"][0]["size_bytes"] == len("tracked\n")
    assert len(manifest["artifacts"][0]["sha256"]) == 64
    assert record.path == "files/artifact.txt"


def test_tracking_smoke_finite_loss_and_report_files(tmp_path: Path) -> None:
    report = run_tracking_smoke(
        TrackingSmokeConfig(out=tmp_path, overwrite=True, steps=1),
        command=["scripts/run_tracking_smoke.py"],
        repo_dir=ROOT,
    )

    assert report["overall_status"] == "pass"
    assert report["status"] == "passed"
    assert report["metadata"]["phase"] == "P47"
    assert report["tracking_mode"] == "local"
    assert report["wandb_status"] == "skipped"
    assert math.isfinite(report["summary"]["final_loss"])
    assert report["summary"]["final_loss_is_finite"] is True
    assert report["metrics_logged_count"] == 1
    assert report["artifacts_logged_count"] >= 4
    assert report["summary_written"] is True
    for relative in (
        "P47_RESULTS.md",
        "tracking_smoke_report.json",
        "local_run/run_metadata.json",
        "local_run/config.json",
        "local_run/metrics.jsonl",
        "local_run/summary.json",
        "local_run/artifacts_manifest.json",
    ):
        assert (tmp_path / relative).is_file()

    metric_lines = (tmp_path / "local_run" / "metrics.jsonl").read_text().splitlines()
    metric = json.loads(metric_lines[0])
    assert {"train/loss", "train/loss_is_finite", "train/tokens_seen"}
    assert metric["step"] == 1
    assert metric["train/examples_seen"] == 2

    manifest = json.loads(
        (tmp_path / "local_run" / "artifacts_manifest.json").read_text(encoding="utf-8")
    )
    names = {entry["name"] for entry in manifest["artifacts"]}
    assert {"config", "metrics", "summary", "tiny-checkpoint-marker"} <= names


def test_report_writing(tmp_path: Path) -> None:
    paths = write_tracking_smoke_reports(
        {
            "overall_status": "pass",
            "status": "passed",
            "tracking_mode": "local",
            "artifact_path": "artifacts/p47_experiment_tracking_smoke",
            "local_run_id": "local_run",
            "commit": "abc123",
            "git_dirty": "clean",
            "backend": "cpu",
            "device_count": 1,
            "steps": 1,
            "final_loss": 0.5,
            "loss_is_finite": True,
            "metrics_logged_count": 1,
            "artifacts_logged_count": 4,
            "summary_written": True,
            "wandb_status": "skipped",
            "limitations": ["tiny tracking smoke only"],
            "summary": {"steps": 1, "final_loss": 0.5},
            "metadata": {"created_at_utc": "2026-05-08T00:00:00Z"},
            "paths": {"metrics": "local_run/metrics.jsonl"},
        },
        out_dir=tmp_path,
    )

    assert paths["json"].is_file()
    assert paths["markdown"].is_file()
    assert "P47 Experiment Tracking Smoke" in paths["markdown"].read_text(
        encoding="utf-8"
    )


def test_run_tracking_smoke_help() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_tracking_smoke.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--tracking" in result.stdout
    assert "--project" in result.stdout
    assert "--steps" in result.stdout


def test_wandb_adapter_is_import_safe_when_wandb_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)
    module = importlib.import_module("qrwkv_xla.tracking.wandb_adapter")

    with pytest.raises(ImportError, match="wandb is not installed"):
        module._import_wandb()
