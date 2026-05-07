from __future__ import annotations

import json
from pathlib import Path

import pytest

from qrwkv_xla.distill import load_distill_stage_config
from qrwkv_xla.smoke.colab_tpu import (
    CONFIG_PATH,
    FIRST_CHECKPOINT,
    ColabTpuSmokeError,
    RuntimeSummary,
    SmokeRunArtifacts,
    assert_tpu_backend,
    non_tpu_backend_message,
    read_last_metric,
    read_run_summary,
    validate_required_artifacts,
    validate_smoke_outputs,
)

ROOT = Path(__file__).resolve().parents[1]


def test_colab_tpu_smoke_config_is_hidden_only_qwen_reference() -> None:
    config = load_distill_stage_config(ROOT / CONFIG_PATH)

    assert config.targets_dir == Path("artifacts/teacher_targets/p36_colab_tpu_smoke")
    assert config.student.architecture == "rwkv7_qwen_reference"
    assert config.student.vocab_size == 512
    assert config.student.num_heads == 2
    assert config.student.num_kv_heads == 1
    assert config.student.emit_logits is False
    assert config.training.max_steps == 1
    assert config.training.seed == 0
    assert config.losses.hidden_mse.enabled is True
    assert config.losses.hidden_mse.weight == 1.0
    assert config.losses.logits_kl.enabled is False
    assert config.losses.attention_or_mixer.enabled is False


def test_non_tpu_backend_error_message_is_colab_friendly() -> None:
    summary = RuntimeSummary(
        python="3.11.0",
        platform="Linux",
        jax_version="0.4",
        default_backend="cpu",
        devices=("id=0, platform=cpu, kind=cpu",),
        git_commit="abc123",
        git_dirty=False,
    )

    expected = (
        "Expected JAX backend 'tpu', got 'cpu'. In Colab, select Runtime → "
        "Change runtime type → TPU, then restart the runtime."
    )

    with pytest.raises(ColabTpuSmokeError, match="Expected JAX backend"):
        assert_tpu_backend(summary)

    message = non_tpu_backend_message(summary)
    assert message == expected


def test_required_artifact_validation_reports_missing_file(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "first"
    run_dir = tmp_path / "runs" / "first"
    checkpoint_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (checkpoint_dir / "checkpoint.json").write_text("{}\n", encoding="utf-8")
    (checkpoint_dir / "params.npz").write_bytes(b"npz")
    (run_dir / "run.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text("{}\n", encoding="utf-8")

    artifacts = SmokeRunArtifacts(
        checkpoint_dir=checkpoint_dir,
        run_dir=run_dir,
        run_json=run_dir / "run.json",
        metrics_jsonl=run_dir / "metrics.jsonl",
        summary_json=run_dir / "summary.json",
    )
    with pytest.raises(ColabTpuSmokeError, match="summary.json"):
        validate_required_artifacts(artifacts)


def test_read_run_summary_requires_nested_summary_mapping(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"status": "completed"}) + "\n", encoding="utf-8")

    with pytest.raises(ColabTpuSmokeError, match="missing summary mapping"):
        read_run_summary(path)


def test_read_metrics_line_parses_optimizer_and_loss_fields(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps(
            {
                "step": 1,
                "values": {
                    "loss": 1.25,
                    "hidden_mse": 1.25,
                    "optimizer_step": 1.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = read_last_metric(path)
    assert metrics["loss"] == 1.25
    assert metrics["hidden_mse"] == 1.25
    assert metrics["optimizer_step"] == 1.0


def test_smoke_output_validation_checks_resume_semantics(tmp_path: Path) -> None:
    first = _complete_artifacts(tmp_path / "first")
    resume = _complete_artifacts(tmp_path / "resume")
    first_summary = _summary(start=0, end=1, resume_from=None)
    resume_summary = _summary(start=1, end=2, resume_from=FIRST_CHECKPOINT)

    validate_smoke_outputs(
        first=first,
        resume=resume,
        first_summary=first_summary,
        resume_summary=resume_summary,
        first_metrics={"loss": 1.0, "hidden_mse": 1.0, "optimizer_step": 1.0},
        resume_metrics={"loss": 1.0, "hidden_mse": 1.0, "optimizer_step": 2.0},
        first_checkpoint_step=1,
        resume_checkpoint_step=2,
    )


def test_smoke_output_validation_rejects_missing_resume_source(tmp_path: Path) -> None:
    first = _complete_artifacts(tmp_path / "first")
    resume = _complete_artifacts(tmp_path / "resume")

    with pytest.raises(ColabTpuSmokeError, match="resume_from"):
        validate_smoke_outputs(
            first=first,
            resume=resume,
            first_summary=_summary(start=0, end=1, resume_from=None),
            resume_summary=_summary(start=1, end=2, resume_from=None),
            first_metrics={"loss": 1.0, "hidden_mse": 1.0, "optimizer_step": 1.0},
            resume_metrics={"loss": 1.0, "hidden_mse": 1.0, "optimizer_step": 2.0},
            first_checkpoint_step=1,
            resume_checkpoint_step=2,
        )


def _complete_artifacts(root: Path) -> SmokeRunArtifacts:
    checkpoint_dir = root / "checkpoints" / "out"
    run_dir = root / "runs" / "out"
    checkpoint_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    for path in (
        checkpoint_dir / "checkpoint.json",
        checkpoint_dir / "params.npz",
        run_dir / "run.json",
        run_dir / "metrics.jsonl",
        run_dir / "summary.json",
    ):
        path.write_text("{}\n", encoding="utf-8")
    return SmokeRunArtifacts(
        checkpoint_dir=checkpoint_dir,
        run_dir=run_dir,
        run_json=run_dir / "run.json",
        metrics_jsonl=run_dir / "metrics.jsonl",
        summary_json=run_dir / "summary.json",
    )


def _summary(*, start: int, end: int, resume_from: Path | None) -> dict[str, object]:
    return {
        "start_step": start,
        "end_step": end,
        "final_loss": 1.0,
        "final_hidden_mse": 1.0,
        "resume_from": None if resume_from is None else str(resume_from),
    }
