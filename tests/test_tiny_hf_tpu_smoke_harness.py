from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.distill import load_distill_stage_config
from qrwkv_xla.smoke.colab_tpu import (
    P38_TINY_HF_SPEC,
    TINY_HF_CONFIG_PATH,
    TINY_HF_EXPORT_CONFIG_PATH,
    TINY_HF_FIRST_CHECKPOINT,
    TINY_HF_TARGETS_DIR,
    ColabTpuSmokeError,
    RuntimeSummary,
    SmokeResults,
    non_tpu_backend_message,
    validate_real_hf_targets,
    validate_smoke_outputs,
    write_results_bundle,
    write_results_markdown,
)
from qrwkv_xla.targets import (
    TargetFlags,
    TeacherTargetManifest,
    read_shard,
    write_target_bundle,
)
from qrwkv_xla.targets.store import list_shard_paths
from qrwkv_xla.teacher_export import load_teacher_export_config
from tests.test_colab_tpu_smoke_harness import _complete_artifacts, _summary

ROOT = Path(__file__).resolve().parents[1]


def test_tiny_hf_export_config_uses_real_hf_tiny_gpt2() -> None:
    config = load_teacher_export_config(ROOT / TINY_HF_EXPORT_CONFIG_PATH)

    assert config.runtime.exporter_backend == "hf"
    assert config.runtime.output_dir == (ROOT / TINY_HF_TARGETS_DIR).resolve()
    assert config.teacher.resolved_model_id == "sshleifer/tiny-gpt2"
    assert config.teacher.tokenizer_id == "sshleifer/tiny-gpt2"
    assert config.teacher.local_files_only is False
    assert config.targets.sequence_length == 8
    assert config.targets.include_logits is True
    assert config.targets.vocab_size == 50257
    assert len(config.targets.prompt_texts) == 2


def test_tiny_hf_tpu_smoke_config_enables_logits_kl_qwen_reference() -> None:
    config = load_distill_stage_config(ROOT / TINY_HF_CONFIG_PATH)

    assert config.targets_dir == TINY_HF_TARGETS_DIR
    assert config.student.architecture == "rwkv7_qwen_reference"
    assert config.student.vocab_size == 50257
    assert config.student.num_heads == 2
    assert config.student.num_kv_heads == 1
    assert config.student.emit_logits is True
    assert config.training.max_steps == 1
    assert config.losses.hidden_mse.enabled is True
    assert config.losses.hidden_mse.weight == 1.0
    assert config.losses.logits_kl.enabled is True
    assert config.losses.logits_kl.weight == 1.0
    assert config.losses.attention_or_mixer.enabled is False


def test_tiny_hf_non_tpu_backend_message_is_unchanged() -> None:
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


def test_tiny_hf_target_validation_accepts_required_arrays(tmp_path: Path) -> None:
    targets_dir = _write_hf_bundle(tmp_path / "targets")

    validate_real_hf_targets(targets_dir)


def test_tiny_hf_target_validation_rejects_missing_loss_mask(tmp_path: Path) -> None:
    targets_dir = _write_hf_bundle(tmp_path / "targets")
    shard_path = list_shard_paths(targets_dir)[0]
    arrays = read_shard(shard_path)
    arrays.pop("loss_mask")
    np.savez(shard_path, **arrays)

    with pytest.raises(ColabTpuSmokeError, match="loss_mask"):
        validate_real_hf_targets(targets_dir)


def test_tiny_hf_target_validation_rejects_bad_logits_shape(tmp_path: Path) -> None:
    targets_dir = _write_hf_bundle(tmp_path / "targets")
    shard_path = list_shard_paths(targets_dir)[0]
    arrays = read_shard(shard_path)
    arrays["logits"] = np.zeros((2, 7, 50257), dtype=np.float32)
    np.savez(shard_path, **arrays)

    with pytest.raises(ColabTpuSmokeError, match="target shapes are invalid"):
        validate_real_hf_targets(targets_dir)


def test_tiny_hf_smoke_output_validation_accepts_finite_metrics(
    tmp_path: Path,
) -> None:
    first = _complete_artifacts(tmp_path / "first")
    resume = _complete_artifacts(tmp_path / "resume")
    first_summary = _summary(start=0, end=1, resume_from=None) | {
        "final_logits_kl": 0.25
    }
    resume_summary = _summary(start=1, end=2, resume_from=TINY_HF_FIRST_CHECKPOINT) | {
        "final_logits_kl": 0.2
    }

    validate_smoke_outputs(
        first=first,
        resume=resume,
        first_summary=first_summary,
        resume_summary=resume_summary,
        first_metrics={
            "loss": 1.0,
            "hidden_mse": 0.75,
            "logits_kl": 0.25,
            "optimizer_step": 1.0,
        },
        resume_metrics={
            "loss": 0.9,
            "hidden_mse": 0.7,
            "logits_kl": 0.2,
            "optimizer_step": 2.0,
        },
        first_checkpoint_step=1,
        resume_checkpoint_step=2,
        expected_resume_from=TINY_HF_FIRST_CHECKPOINT,
        metric_names=("loss", "hidden_mse", "logits_kl"),
        phase="P38",
    )


def test_tiny_hf_results_bundle_includes_export_config_and_targets(
    tmp_path: Path,
) -> None:
    spec = P38_TINY_HF_SPEC.__class__(
        **{
            **P38_TINY_HF_SPEC.__dict__,
            "config_path": ROOT / TINY_HF_CONFIG_PATH,
            "teacher_export_config_path": ROOT / TINY_HF_EXPORT_CONFIG_PATH,
            "targets_dir": _write_hf_bundle(tmp_path / "targets"),
            "artifact_dir": tmp_path / "artifacts",
            "results_md": tmp_path / "artifacts" / "P38_RESULTS.md",
            "results_bundle": tmp_path / "artifacts" / "p38_results_bundle.tar.gz",
            "first_checkpoint": tmp_path / "checkpoints" / "first",
            "resume_checkpoint": tmp_path / "checkpoints" / "resume",
            "first_run_dir": tmp_path / "runs" / "first",
            "resume_run_dir": tmp_path / "runs" / "resume",
            "run_root": tmp_path / "runs",
        }
    )
    _write_checkpoint(spec.first_checkpoint)
    _write_checkpoint(spec.resume_checkpoint)
    spec.first_run_dir.mkdir(parents=True)
    spec.resume_run_dir.mkdir(parents=True)
    for run_dir in (spec.first_run_dir, spec.resume_run_dir):
        (run_dir / "run.json").write_text("{}\n", encoding="utf-8")
        (run_dir / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
        (run_dir / "summary.json").write_text("{}\n", encoding="utf-8")
    results = SmokeResults(
        runtime=RuntimeSummary(
            python="3.11.0",
            platform="Linux",
            jax_version="0.4",
            default_backend="tpu",
            devices=("id=0, platform=tpu, kind=TPU",),
            git_commit="abc123",
            git_dirty=False,
        ),
        matmul_sum=120.0,
        first=_summary(start=0, end=1, resume_from=None) | {"final_logits_kl": 0.25},
        resume=_summary(start=1, end=2, resume_from=spec.first_checkpoint)
        | {"final_logits_kl": 0.2},
        first_checkpoint_step=1,
        resume_checkpoint_step=2,
    )

    write_results_markdown(results, spec.results_md, spec=spec)
    bundle_path = write_results_bundle(spec.results_bundle, spec=spec)

    with tarfile.open(bundle_path, "r:gz") as archive:
        names = set(archive.getnames())
    assert any(
        name.endswith("configs/teacher_export_p38_tiny_hf_logits_smoke.yaml")
        for name in names
    )
    assert any(name.endswith("manifest.json") for name in names)
    assert any(name.endswith("shard_000000.npz") for name in names)
    assert any(name.endswith("P38_RESULTS.md") for name in names)


def _write_hf_bundle(bundle_dir: Path) -> Path:
    arrays = {
        "input_ids": np.arange(16, dtype=np.int32).reshape(2, 8),
        "attention_mask": np.ones((2, 8), dtype=np.int32),
        "loss_mask": np.ones((2, 8), dtype=np.int32),
        "hidden_states": np.ones((2, 2, 8, 2), dtype=np.float32),
        "logits": np.ones((2, 8, 50257), dtype=np.float32),
    }
    manifest = TeacherTargetManifest(
        schema_version="0.1",
        teacher_family="hf-causal-lm",
        teacher_model_id="sshleifer/tiny-gpt2",
        teacher_policy_label="tiny-hf-smoke-p38",
        fallback_policy_label=None,
        tokenizer_id="sshleifer/tiny-gpt2",
        sequence_length=8,
        hidden_size=2,
        num_layers=2,
        targets=TargetFlags(
            input_ids=True,
            attention_mask=True,
            loss_mask=True,
            hidden_states=True,
            logits=True,
            attention_targets=False,
        ),
        dtype="fp32",
        created_by="HFTeacherExporter",
        notes=["huggingface teacher exporter bundle"],
        extra={"exporter_backend": "hf", "vocab_size": 50257},
    )
    write_target_bundle(bundle_dir, manifest, [arrays])
    return bundle_dir


def _write_checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "checkpoint.json").write_text("{}\n", encoding="utf-8")
    (path / "params.npz").write_bytes(b"npz")
