from __future__ import annotations

import importlib.util
import json
import tarfile
from pathlib import Path

import pytest

from qrwkv_xla.distill import load_distill_stage_config
from qrwkv_xla.scale_planner import (
    HARDWARE_PROFILES,
    MODEL_PROFILES,
    validate_hardware_profile,
    validate_model_profile,
)
from qrwkv_xla.smoke.colab_tpu import (
    P39_GENERATED_DISTILL_CONFIG_PATH,
    P39_GENERATED_TEACHER_EXPORT_CONFIG_PATH,
    P39_HARDWARE_PROFILE,
    P39_MODEL_PROFILE,
    P39_PLANNER_TPU_SPEC,
    P39_SCALE_PLAN_JSON,
    P39_SCALE_PLAN_YAML,
    P39_TARGETS_DIR,
    ColabTpuSmokeError,
    RuntimeSummary,
    SmokeResults,
    assert_tpu_backend,
    non_tpu_backend_message,
    read_last_metric,
    validate_p39_plan_fit,
    validate_required_artifacts,
    validate_smoke_outputs,
    write_p39_planner_artifacts,
    write_results_bundle,
    write_results_markdown,
)
from qrwkv_xla.teacher_export import load_teacher_export_config
from tests.test_colab_tpu_smoke_harness import _complete_artifacts, _summary
from tests.test_tiny_hf_tpu_smoke_harness import _write_checkpoint, _write_hf_bundle

ROOT = Path(__file__).resolve().parents[1]


def test_p39_planner_profile_exists_and_is_rope_valid() -> None:
    profile = MODEL_PROFILES[P39_MODEL_PROFILE]

    validate_model_profile(profile)
    assert profile.backend == "rwkv7_qwen_reference"
    assert profile.vocab_size == 50257
    assert profile.hidden_size == 2
    assert profile.num_heads == 1
    assert profile.resolved_head_size == 2
    assert profile.resolved_head_size % 2 == 0
    assert profile.emit_logits is True


def test_p39_kaggle_tpu_v5e_profile_exists_and_selects() -> None:
    profile = HARDWARE_PROFILES[P39_HARDWARE_PROFILE]

    validate_hardware_profile(profile)
    assert profile.name == "kaggle_tpu_v5e_8"
    assert profile.device_kind == "tpu"
    assert profile.device_count == 8
    assert profile.supports_bf16 is True


def test_p39_generated_planner_artifacts_and_fit() -> None:
    plan = write_p39_planner_artifacts()

    validate_p39_plan_fit(plan)
    assert (ROOT / P39_SCALE_PLAN_YAML).is_file()
    assert (ROOT / P39_SCALE_PLAN_JSON).is_file()
    payload = json.loads((ROOT / P39_SCALE_PLAN_JSON).read_text(encoding="utf-8"))
    assert payload["model_profile"]["name"] == P39_MODEL_PROFILE
    assert payload["hardware_profile"]["name"] == P39_HARDWARE_PROFILE
    assert payload["fit"]["fit"] in {"yes", "maybe"}


def test_p39_generated_distill_config_fields() -> None:
    write_p39_planner_artifacts()
    config = load_distill_stage_config(ROOT / P39_GENERATED_DISTILL_CONFIG_PATH)

    assert config.targets_dir == P39_TARGETS_DIR
    assert config.student.architecture == "rwkv7_qwen_reference"
    assert config.student.vocab_size == 50257
    assert config.student.hidden_size == 2
    assert config.student.num_layers == 2
    assert config.student.num_heads == 1
    assert config.student.num_kv_heads == 1
    assert config.student.emit_logits is True
    assert config.training.max_steps == 1
    assert config.losses.hidden_mse.enabled is True
    assert config.losses.logits_kl.enabled is True
    assert config.losses.logits_kl.weight == 1.0


def test_p39_generated_teacher_export_target_path() -> None:
    write_p39_planner_artifacts()
    config = load_teacher_export_config(ROOT / P39_GENERATED_TEACHER_EXPORT_CONFIG_PATH)

    assert config.runtime.exporter_backend == "hf"
    assert config.runtime.output_dir == (ROOT / P39_TARGETS_DIR).resolve()
    assert config.teacher.resolved_model_id == "sshleifer/tiny-gpt2"
    assert config.targets.include_logits is True
    assert config.targets.sequence_length == 8


def test_p39_metrics_parser_accepts_and_validation_rejects_missing_logits(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps(
            {
                "step": 1,
                "values": {
                    "loss": 1.0,
                    "hidden_mse": 0.5,
                    "logits_kl": 0.25,
                    "optimizer_step": 1.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert read_last_metric(metrics_path)["logits_kl"] == 0.25

    first = _complete_artifacts(tmp_path / "first")
    resume = _complete_artifacts(tmp_path / "resume")
    with pytest.raises(ColabTpuSmokeError, match="logits_kl metric"):
        validate_smoke_outputs(
            first=first,
            resume=resume,
            first_summary=_summary(start=0, end=1, resume_from=None)
            | {"final_logits_kl": 0.25},
            resume_summary=_summary(
                start=1,
                end=2,
                resume_from=P39_PLANNER_TPU_SPEC.first_checkpoint,
            )
            | {"final_logits_kl": 0.2},
            first_metrics={"loss": 1.0, "hidden_mse": 0.5, "optimizer_step": 1.0},
            resume_metrics={
                "loss": 1.0,
                "hidden_mse": 0.5,
                "logits_kl": 0.2,
                "optimizer_step": 2.0,
            },
            first_checkpoint_step=1,
            resume_checkpoint_step=2,
            expected_resume_from=P39_PLANNER_TPU_SPEC.first_checkpoint,
            metric_names=("loss", "hidden_mse", "logits_kl"),
            phase="P39",
        )


def test_p39_artifact_validation_and_bundle_include_planner_files(
    tmp_path: Path,
) -> None:
    write_p39_planner_artifacts()
    spec = P39_PLANNER_TPU_SPEC.__class__(
        **{
            **P39_PLANNER_TPU_SPEC.__dict__,
            "targets_dir": _write_hf_bundle(tmp_path / "targets"),
            "artifact_dir": tmp_path / "artifacts",
            "results_md": tmp_path / "artifacts" / "P39_RESULTS.md",
            "results_bundle": tmp_path / "artifacts" / "p39_results_bundle.tar.gz",
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

    artifacts = _complete_artifacts(tmp_path / "complete")
    validate_required_artifacts(artifacts, phase="P39")
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

    assert "Export config source: generated" in spec.results_md.read_text(
        encoding="utf-8"
    )
    with tarfile.open(bundle_path, "r:gz") as archive:
        names = set(archive.getnames())
    assert any(name.endswith("scale_plan.yaml") for name in names)
    assert any(name.endswith("scale_plan.json") for name in names)
    assert any(name.endswith("generated_distill.yaml") for name in names)
    assert any(name.endswith("teacher_export.yaml") for name in names)
    assert any(name.endswith("P39_RESULTS.md") for name in names)


def test_p39_non_tpu_backend_message_matches_existing_text() -> None:
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
    assert non_tpu_backend_message(summary) == expected
    with pytest.raises(ColabTpuSmokeError, match="Expected JAX backend"):
        assert_tpu_backend(summary)


def test_tpu_smoke_entrypoints_import_exposure_is_preserved() -> None:
    for script in (
        "run_colab_tpu_smoke.py",
        "run_colab_tpu_logits_smoke.py",
        "run_tiny_hf_tpu_smoke.py",
        "run_planner_tpu_smoke.py",
    ):
        module_path = ROOT / "scripts" / script
        spec = importlib.util.spec_from_file_location(
            script.removesuffix(".py"),
            module_path,
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert callable(module.main)
