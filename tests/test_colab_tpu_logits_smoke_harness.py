from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.distill import load_distill_stage_config
from qrwkv_xla.smoke.colab_tpu import (
    LOGITS_CONFIG_PATH,
    LOGITS_FIRST_CHECKPOINT,
    LOGITS_TARGETS_DIR,
    ColabTpuSmokeError,
    RuntimeSummary,
    export_fake_targets,
    non_tpu_backend_message,
    validate_logits_targets,
    validate_smoke_outputs,
)
from qrwkv_xla.targets import read_manifest, write_manifest
from qrwkv_xla.targets.store import manifest_path
from tests.test_colab_tpu_smoke_harness import _complete_artifacts, _summary

ROOT = Path(__file__).resolve().parents[1]


def test_colab_tpu_logits_smoke_config_enables_logits_kl() -> None:
    config = load_distill_stage_config(ROOT / LOGITS_CONFIG_PATH)

    assert config.targets_dir == LOGITS_TARGETS_DIR
    assert config.student.architecture == "rwkv7_qwen_reference"
    assert config.student.vocab_size == 512
    assert config.student.num_heads == 2
    assert config.student.num_kv_heads == 1
    assert config.student.emit_logits is True
    assert config.training.max_steps == 1
    assert config.losses.hidden_mse.enabled is True
    assert config.losses.hidden_mse.weight == 1.0
    assert config.losses.logits_kl.enabled is True
    assert config.losses.logits_kl.weight == 1.0
    assert config.losses.attention_or_mixer.enabled is False


def test_logits_target_validation_rejects_missing_logits_flag(tmp_path: Path) -> None:
    targets_dir = export_fake_targets(tmp_path / "targets", include_logits=True)
    manifest = read_manifest(manifest_path(targets_dir))
    write_manifest(
        manifest_path(targets_dir),
        replace(manifest, targets=replace(manifest.targets, logits=False)),
    )

    with pytest.raises(ColabTpuSmokeError, match="declare logits=true"):
        validate_logits_targets(targets_dir)


def test_logits_target_validation_rejects_missing_logits_array(tmp_path: Path) -> None:
    targets_dir = export_fake_targets(tmp_path / "targets", include_logits=True)
    shard_path = next((targets_dir / "shards").glob("shard_*.npz"))

    arrays = dict(np.load(shard_path))
    arrays.pop("logits")
    np.savez(shard_path, **arrays)

    with pytest.raises(ColabTpuSmokeError, match="missing logits"):
        validate_logits_targets(targets_dir)


def test_logits_smoke_output_validation_requires_logits_kl_metric(
    tmp_path: Path,
) -> None:
    first = _complete_artifacts(tmp_path / "first")
    resume = _complete_artifacts(tmp_path / "resume")
    first_summary = _summary(start=0, end=1, resume_from=None) | {
        "final_logits_kl": 0.25
    }
    resume_summary = _summary(start=1, end=2, resume_from=LOGITS_FIRST_CHECKPOINT) | {
        "final_logits_kl": 0.2
    }

    with pytest.raises(ColabTpuSmokeError, match="logits_kl metric"):
        validate_smoke_outputs(
            first=first,
            resume=resume,
            first_summary=first_summary,
            resume_summary=resume_summary,
            first_metrics={"loss": 1.0, "hidden_mse": 1.0, "optimizer_step": 1.0},
            resume_metrics={
                "loss": 1.0,
                "hidden_mse": 1.0,
                "logits_kl": 0.2,
                "optimizer_step": 2.0,
            },
            first_checkpoint_step=1,
            resume_checkpoint_step=2,
            expected_resume_from=LOGITS_FIRST_CHECKPOINT,
            metric_names=("loss", "hidden_mse", "logits_kl"),
            phase="P37",
        )


def test_logits_smoke_output_validation_accepts_logits_kl_and_step_progression(
    tmp_path: Path,
) -> None:
    first = _complete_artifacts(tmp_path / "first")
    resume = _complete_artifacts(tmp_path / "resume")
    first_summary = _summary(start=0, end=1, resume_from=None) | {
        "final_logits_kl": 0.25
    }
    resume_summary = _summary(start=1, end=2, resume_from=LOGITS_FIRST_CHECKPOINT) | {
        "final_logits_kl": 0.2
    }

    validate_smoke_outputs(
        first=first,
        resume=resume,
        first_summary=first_summary,
        resume_summary=resume_summary,
        first_metrics={
            "loss": 1.0,
            "hidden_mse": 1.0,
            "logits_kl": 0.25,
            "optimizer_step": 1.0,
        },
        resume_metrics={
            "loss": 1.0,
            "hidden_mse": 1.0,
            "logits_kl": 0.2,
            "optimizer_step": 2.0,
        },
        first_checkpoint_step=1,
        resume_checkpoint_step=2,
        expected_resume_from=LOGITS_FIRST_CHECKPOINT,
        metric_names=("loss", "hidden_mse", "logits_kl"),
        phase="P37",
    )


def test_logits_read_metrics_fixture_contains_logits_kl(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps(
            {
                "step": 1,
                "values": {
                    "loss": 1.25,
                    "hidden_mse": 1.0,
                    "logits_kl": 0.25,
                    "optimizer_step": 1.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    from qrwkv_xla.smoke.colab_tpu import read_last_metric

    metrics = read_last_metric(path)
    assert metrics["logits_kl"] == 0.25


def test_logits_non_tpu_backend_message_matches_hidden_smoke() -> None:
    summary = RuntimeSummary(
        python="3.11.0",
        platform="Linux",
        jax_version="0.4",
        default_backend="cpu",
        devices=("id=0, platform=cpu, kind=cpu",),
        git_commit="abc123",
        git_dirty=False,
    )

    assert non_tpu_backend_message(summary) == (
        "Expected JAX backend 'tpu', got 'cpu'. In Colab, select Runtime → "
        "Change runtime type → TPU, then restart the runtime."
    )
