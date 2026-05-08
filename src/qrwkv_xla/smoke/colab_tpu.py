from __future__ import annotations

import json
import math
import platform
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from qrwkv_xla.checkpointing import load_checkpoint
from qrwkv_xla.distill import (
    DistillCheckpointConfig,
    DistillTrackingConfig,
    load_distill_stage_config,
    run_distill_stage,
)
from qrwkv_xla.scale_planner import (
    ScalePlan,
    ScalePlanRequest,
    distill_config_yaml,
    make_plan,
    plan_to_dict,
    plan_to_yaml,
    resolve_hardware_profile,
    resolve_model_profile,
    resolve_training_mode,
)
from qrwkv_xla.targets import read_manifest
from qrwkv_xla.targets.shards import read_shard, validate_shard_arrays
from qrwkv_xla.targets.store import list_shard_paths, manifest_path
from qrwkv_xla.teacher_export import (
    ExportRequest,
    FakeTeacherExporter,
    TeacherExportConfig,
    get_teacher_exporter,
    load_teacher_export_config,
)

CONFIG_PATH = Path("configs/distill_stage0_qwen_reference_colab_tpu_smoke.yaml")
TARGETS_DIR = Path("artifacts/teacher_targets/p36_colab_tpu_smoke")
ARTIFACT_DIR = Path("artifacts/p36_colab_tpu_smoke")
RESULTS_MD = ARTIFACT_DIR / "P36_RESULTS.md"
RESULTS_BUNDLE = ARTIFACT_DIR / "p36_results_bundle.tar.gz"
FIRST_CHECKPOINT = Path("checkpoints/p36_tpu_qwen_reference_first")
RESUME_CHECKPOINT = Path("checkpoints/p36_tpu_qwen_reference_resume")
FIRST_RUN_DIR = Path("runs/p36/p36_tpu_qwen_reference_first")
RESUME_RUN_DIR = Path("runs/p36/p36_tpu_qwen_reference_resume")
RUN_ROOT = Path("runs/p36")
LOGITS_CONFIG_PATH = Path(
    "configs/distill_stage0_qwen_reference_colab_tpu_logits_smoke.yaml"
)
LOGITS_TARGETS_DIR = Path("artifacts/teacher_targets/p37_colab_tpu_logits_smoke")
LOGITS_ARTIFACT_DIR = Path("artifacts/p37_colab_tpu_logits_smoke")
LOGITS_RESULTS_MD = LOGITS_ARTIFACT_DIR / "P37_RESULTS.md"
LOGITS_RESULTS_BUNDLE = LOGITS_ARTIFACT_DIR / "p37_results_bundle.tar.gz"
LOGITS_FIRST_CHECKPOINT = Path("checkpoints/p37_tpu_qwen_reference_logits_first")
LOGITS_RESUME_CHECKPOINT = Path("checkpoints/p37_tpu_qwen_reference_logits_resume")
LOGITS_FIRST_RUN_DIR = Path("runs/p37/p37_tpu_qwen_reference_logits_first")
LOGITS_RESUME_RUN_DIR = Path("runs/p37/p37_tpu_qwen_reference_logits_resume")
LOGITS_RUN_ROOT = Path("runs/p37")
TINY_HF_CONFIG_PATH = Path(
    "configs/distill_stage0_qwen_reference_p38_tiny_hf_tpu_smoke.yaml"
)
TINY_HF_EXPORT_CONFIG_PATH = Path(
    "configs/teacher_export_p38_tiny_hf_logits_smoke.yaml"
)
TINY_HF_TARGETS_DIR = Path("artifacts/teacher_targets/p38_tiny_hf_logits_smoke")
TINY_HF_ARTIFACT_DIR = Path("artifacts/p38_tiny_hf_tpu_smoke")
TINY_HF_RESULTS_MD = TINY_HF_ARTIFACT_DIR / "P38_RESULTS.md"
TINY_HF_RESULTS_BUNDLE = TINY_HF_ARTIFACT_DIR / "p38_results_bundle.tar.gz"
TINY_HF_FIRST_CHECKPOINT = Path("checkpoints/p38_tpu_qwen_reference_tiny_hf_first")
TINY_HF_RESUME_CHECKPOINT = Path("checkpoints/p38_tpu_qwen_reference_tiny_hf_resume")
TINY_HF_FIRST_RUN_DIR = Path("runs/p38/p38_tpu_qwen_reference_tiny_hf_first")
TINY_HF_RESUME_RUN_DIR = Path("runs/p38/p38_tpu_qwen_reference_tiny_hf_resume")
TINY_HF_RUN_ROOT = Path("runs/p38")
P39_ARTIFACT_DIR = Path("artifacts/p39_planner_tpu_smoke")
P39_SCALE_PLAN_YAML = P39_ARTIFACT_DIR / "scale_plan.yaml"
P39_SCALE_PLAN_JSON = P39_ARTIFACT_DIR / "scale_plan.json"
P39_GENERATED_DISTILL_CONFIG_PATH = P39_ARTIFACT_DIR / "generated_distill.yaml"
P39_GENERATED_TEACHER_EXPORT_CONFIG_PATH = P39_ARTIFACT_DIR / "teacher_export.yaml"
P39_TARGETS_DIR = Path("artifacts/teacher_targets/p39_tiny_hf_logits_smoke")
P39_RESULTS_MD = P39_ARTIFACT_DIR / "P39_RESULTS.md"
P39_RESULTS_BUNDLE = P39_ARTIFACT_DIR / "p39_results_bundle.tar.gz"
P39_FIRST_CHECKPOINT = Path("checkpoints/p39_planner_tpu_smoke_first")
P39_RESUME_CHECKPOINT = Path("checkpoints/p39_planner_tpu_smoke_resume")
P39_FIRST_RUN_DIR = Path("runs/p39/p39_planner_tpu_smoke_first")
P39_RESUME_RUN_DIR = Path("runs/p39/p39_planner_tpu_smoke_resume")
P39_RUN_ROOT = Path("runs/p39")
P39_MODEL_PROFILE = "p39_tiny_hf_qwen_rope_smoke"
P39_HARDWARE_PROFILE = "kaggle_tpu_v5e_8"
P39_TRAINING_MODE = "smoke_hidden_logits_sgd"


class ColabTpuSmokeError(RuntimeError):
    """Raised when the manual Colab TPU smoke cannot prove its contract."""


@dataclass(frozen=True)
class SmokeSpec:
    phase: str
    config_path: Path
    targets_dir: Path
    artifact_dir: Path
    results_md: Path
    results_bundle: Path
    first_checkpoint: Path
    resume_checkpoint: Path
    first_run_dir: Path
    resume_run_dir: Path
    run_root: Path
    first_run_name: str
    resume_run_name: str
    include_logits: bool
    metric_names: tuple[str, ...]
    export_kind: str
    teacher_export_config_path: Path | None
    result_title: str
    result_scope: str
    result_limits: str
    tracking_tags: tuple[str, ...]
    tracking_notes: tuple[str, ...]


P36_SPEC = SmokeSpec(
    phase="P36",
    config_path=CONFIG_PATH,
    targets_dir=TARGETS_DIR,
    artifact_dir=ARTIFACT_DIR,
    results_md=RESULTS_MD,
    results_bundle=RESULTS_BUNDLE,
    first_checkpoint=FIRST_CHECKPOINT,
    resume_checkpoint=RESUME_CHECKPOINT,
    first_run_dir=FIRST_RUN_DIR,
    resume_run_dir=RESUME_RUN_DIR,
    run_root=RUN_ROOT,
    first_run_name="p36_tpu_qwen_reference_first",
    resume_run_name="p36_tpu_qwen_reference_resume",
    include_logits=False,
    metric_names=("loss", "hidden_mse"),
    export_kind="fake",
    teacher_export_config_path=None,
    result_title="P36 Colab TPU Smoke Results",
    result_scope=(
        "This artifact proves a tiny manual TPU hidden-MSE train/resume smoke only."
    ),
    result_limits=(
        "It does not prove model quality, Qwen-scale fit, pjit sharding, "
        "Pallas kernels, or logits KL."
    ),
    tracking_tags=("p36", "colab-tpu-smoke", "manual"),
    tracking_notes=(
        "P36 manual Colab TPU smoke",
        "tiny hidden-only rwkv7_qwen_reference train/resume proof",
    ),
)
P37_LOGITS_SPEC = SmokeSpec(
    phase="P37",
    config_path=LOGITS_CONFIG_PATH,
    targets_dir=LOGITS_TARGETS_DIR,
    artifact_dir=LOGITS_ARTIFACT_DIR,
    results_md=LOGITS_RESULTS_MD,
    results_bundle=LOGITS_RESULTS_BUNDLE,
    first_checkpoint=LOGITS_FIRST_CHECKPOINT,
    resume_checkpoint=LOGITS_RESUME_CHECKPOINT,
    first_run_dir=LOGITS_FIRST_RUN_DIR,
    resume_run_dir=LOGITS_RESUME_RUN_DIR,
    run_root=LOGITS_RUN_ROOT,
    first_run_name="p37_tpu_qwen_reference_logits_first",
    resume_run_name="p37_tpu_qwen_reference_logits_resume",
    include_logits=True,
    metric_names=("loss", "hidden_mse", "logits_kl"),
    export_kind="fake",
    teacher_export_config_path=None,
    result_title="P37 Colab TPU Logits-KL Smoke Results",
    result_scope=(
        "This artifact proves a tiny manual TPU hidden-MSE plus logits-KL "
        "train/resume smoke only."
    ),
    result_limits=(
        "It does not prove scale, model quality, real Qwen target training on TPU, "
        "pjit sharding, multi-device TPU execution, Pallas kernels, or real HF "
        "teacher export on TPU."
    ),
    tracking_tags=("p37", "colab-tpu-logits-smoke", "manual"),
    tracking_notes=(
        "P37 manual Colab TPU logits-KL smoke",
        "tiny logits-bearing rwkv7_qwen_reference train/resume proof",
    ),
)
P38_TINY_HF_SPEC = SmokeSpec(
    phase="P38",
    config_path=TINY_HF_CONFIG_PATH,
    targets_dir=TINY_HF_TARGETS_DIR,
    artifact_dir=TINY_HF_ARTIFACT_DIR,
    results_md=TINY_HF_RESULTS_MD,
    results_bundle=TINY_HF_RESULTS_BUNDLE,
    first_checkpoint=TINY_HF_FIRST_CHECKPOINT,
    resume_checkpoint=TINY_HF_RESUME_CHECKPOINT,
    first_run_dir=TINY_HF_FIRST_RUN_DIR,
    resume_run_dir=TINY_HF_RESUME_RUN_DIR,
    run_root=TINY_HF_RUN_ROOT,
    first_run_name="p38_tpu_qwen_reference_tiny_hf_first",
    resume_run_name="p38_tpu_qwen_reference_tiny_hf_resume",
    include_logits=True,
    metric_names=("loss", "hidden_mse", "logits_kl"),
    export_kind="hf",
    teacher_export_config_path=TINY_HF_EXPORT_CONFIG_PATH,
    result_title="P38 Real Tiny HF TPU Distill Smoke Results",
    result_scope=(
        "This artifact proves a tiny manual TPU distill train/resume smoke using "
        "real sshleifer/tiny-gpt2 Hugging Face teacher targets."
    ),
    result_limits=(
        "It does not prove Qwen-scale export or training, model quality, "
        "multi-host TPU, pjit sharding, Pallas kernels, lm_eval, WandB, or HF "
        "student export."
    ),
    tracking_tags=("p38", "tiny-hf-tpu-smoke", "manual"),
    tracking_notes=(
        "P38 manual TPU smoke with real sshleifer/tiny-gpt2 teacher targets",
        "tiny logits-bearing rwkv7_qwen_reference train/resume proof",
    ),
)
P39_PLANNER_TPU_SPEC = SmokeSpec(
    phase="P39",
    config_path=P39_GENERATED_DISTILL_CONFIG_PATH,
    targets_dir=P39_TARGETS_DIR,
    artifact_dir=P39_ARTIFACT_DIR,
    results_md=P39_RESULTS_MD,
    results_bundle=P39_RESULTS_BUNDLE,
    first_checkpoint=P39_FIRST_CHECKPOINT,
    resume_checkpoint=P39_RESUME_CHECKPOINT,
    first_run_dir=P39_FIRST_RUN_DIR,
    resume_run_dir=P39_RESUME_RUN_DIR,
    run_root=P39_RUN_ROOT,
    first_run_name="p39_planner_tpu_smoke_first",
    resume_run_name="p39_planner_tpu_smoke_resume",
    include_logits=True,
    metric_names=("loss", "hidden_mse", "logits_kl"),
    export_kind="hf",
    teacher_export_config_path=P39_GENERATED_TEACHER_EXPORT_CONFIG_PATH,
    result_title="P39 Planner-Generated TPU Smoke Results",
    result_scope=(
        "This artifact proves the P39 scale planner can generate the tiny TPU "
        "smoke config and that the generated config runs one-step first/resume "
        "distillation on a real TPU with sshleifer/tiny-gpt2 teacher targets."
    ),
    result_limits=(
        "It does not prove Qwen-scale fit, long training, pjit sharding, "
        "multi-host TPU, Pallas kernels, lm_eval, WandB, or HF student export."
    ),
    tracking_tags=("p39", "planner-tpu-smoke", "tiny-hf", "manual"),
    tracking_notes=(
        "P39 planner-generated TPU smoke",
        "tiny RoPE-valid rwkv7_qwen_reference train/resume proof",
    ),
)


@dataclass(frozen=True)
class RuntimeSummary:
    python: str
    platform: str
    jax_version: str
    default_backend: str
    devices: tuple[str, ...]
    git_commit: str
    git_dirty: bool

    @property
    def has_tpu(self) -> bool:
        return any(
            ", platform=tpu" in device or "platform=tpu" in device
            for device in self.devices
        )


@dataclass(frozen=True)
class SmokeRunArtifacts:
    checkpoint_dir: Path
    run_dir: Path
    run_json: Path
    metrics_jsonl: Path
    summary_json: Path


@dataclass(frozen=True)
class SmokeResults:
    runtime: RuntimeSummary
    matmul_sum: float
    first: dict[str, Any]
    resume: dict[str, Any]
    first_checkpoint_step: int
    resume_checkpoint_step: int


def collect_runtime_summary(repo_root: Path) -> RuntimeSummary:
    import jax

    devices = tuple(_format_device(device) for device in jax.devices())
    return RuntimeSummary(
        python=sys.version.split()[0],
        platform=platform.platform(),
        jax_version=str(jax.__version__),
        default_backend=str(jax.default_backend()),
        devices=devices,
        git_commit=_git_output(repo_root, "rev-parse", "HEAD") or "unknown",
        git_dirty=bool(_git_output(repo_root, "status", "--porcelain")),
    )


def format_runtime_summary(summary: RuntimeSummary) -> str:
    lines = [
        f"python: {summary.python}",
        f"platform: {summary.platform}",
        f"jax_version: {summary.jax_version}",
        f"default_backend: {summary.default_backend}",
        f"git_commit: {summary.git_commit}",
        f"git_dirty: {summary.git_dirty}",
        "devices:",
    ]
    lines.extend(f"- {device}" for device in summary.devices)
    return "\n".join(lines)


def assert_tpu_backend(summary: RuntimeSummary) -> None:
    if summary.default_backend == "tpu" and summary.has_tpu:
        return
    raise ColabTpuSmokeError(non_tpu_backend_message(summary))


def non_tpu_backend_message(summary: RuntimeSummary) -> str:
    return (
        f"Expected JAX backend 'tpu', got {summary.default_backend!r}. "
        "In Colab, select Runtime → Change runtime type → TPU, then restart "
        "the runtime."
    )


def run_tiny_matmul_sanity() -> float:
    import jax
    import jax.numpy as jnp

    lhs = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
    rhs = jnp.eye(4, dtype=jnp.float32)
    return float(jax.device_get((lhs @ rhs).sum()))


def export_fake_targets(
    output_dir: Path = TARGETS_DIR,
    *,
    include_logits: bool = False,
) -> Path:
    from dataclasses import replace

    config = TeacherExportConfig()
    config = replace(
        config,
        targets=replace(
            config.targets,
            sequence_length=8,
            hidden_size=8,
            num_layers=2,
            include_logits=include_logits,
            include_attention_targets=False,
            vocab_size=512,
        ),
        runtime=replace(
            config.runtime,
            output_dir=output_dir,
            batch_size=1,
            num_shards=1,
            seed=3737 if include_logits else 3636,
        ),
    )
    FakeTeacherExporter().export(ExportRequest(config=config, output_dir=output_dir))
    return output_dir


def run_distill_smoke_pair(config_path: Path = CONFIG_PATH) -> SmokeResults:
    return _run_distill_smoke_pair(P36_SPEC, config_path=config_path)


def run_logits_smoke_pair(config_path: Path = LOGITS_CONFIG_PATH) -> SmokeResults:
    return _run_distill_smoke_pair(P37_LOGITS_SPEC, config_path=config_path)


def run_tiny_hf_smoke_pair(config_path: Path = TINY_HF_CONFIG_PATH) -> SmokeResults:
    return _run_distill_smoke_pair(P38_TINY_HF_SPEC, config_path=config_path)


def run_planner_tpu_smoke(
    config_path: Path = P39_GENERATED_DISTILL_CONFIG_PATH,
) -> SmokeResults:
    runtime = collect_runtime_summary(Path.cwd())
    print(format_runtime_summary(runtime))
    assert_tpu_backend(runtime)
    matmul_sum = run_tiny_matmul_sanity()
    print(f"tiny_matmul_sum: {matmul_sum:.8f}")

    plan = write_p39_planner_artifacts()
    validate_p39_plan_fit(plan)
    print(
        f"planner_fit: {plan.fit.fit} ({plan.fit.estimated_total_gb:.4f} GiB estimated)"
    )
    return _run_prepared_smoke_pair(
        P39_PLANNER_TPU_SPEC,
        config_path=config_path,
        runtime=runtime,
        matmul_sum=matmul_sum,
    )


def _run_distill_smoke_pair(spec: SmokeSpec, *, config_path: Path) -> SmokeResults:
    runtime = collect_runtime_summary(Path.cwd())
    print(format_runtime_summary(runtime))
    assert_tpu_backend(runtime)
    matmul_sum = run_tiny_matmul_sanity()
    print(f"tiny_matmul_sum: {matmul_sum:.8f}")

    return _run_prepared_smoke_pair(
        spec,
        config_path=config_path,
        runtime=runtime,
        matmul_sum=matmul_sum,
    )


def _run_prepared_smoke_pair(
    spec: SmokeSpec,
    *,
    config_path: Path,
    runtime: RuntimeSummary,
    matmul_sum: float,
) -> SmokeResults:
    export_smoke_targets(spec)
    validate_smoke_targets(spec.targets_dir, spec=spec)
    first_artifacts, first_summary, first_metrics = _run_one_step(
        spec=spec,
        config_path=config_path,
        checkpoint_dir=spec.first_checkpoint,
        stable_run_dir=spec.first_run_dir,
        run_name=spec.first_run_name,
        resume_from=None,
    )
    resume_artifacts, resume_summary, resume_metrics = _run_one_step(
        spec=spec,
        config_path=config_path,
        checkpoint_dir=spec.resume_checkpoint,
        stable_run_dir=spec.resume_run_dir,
        run_name=spec.resume_run_name,
        resume_from=spec.first_checkpoint,
    )
    first_step = load_checkpoint(spec.first_checkpoint).manifest.step
    resume_step = load_checkpoint(spec.resume_checkpoint).manifest.step

    validate_smoke_outputs(
        first=first_artifacts,
        resume=resume_artifacts,
        first_summary=first_summary,
        resume_summary=resume_summary,
        first_metrics=first_metrics,
        resume_metrics=resume_metrics,
        first_checkpoint_step=first_step,
        resume_checkpoint_step=resume_step,
        expected_resume_from=spec.first_checkpoint,
        metric_names=spec.metric_names,
        phase=spec.phase,
    )
    results = SmokeResults(
        runtime=runtime,
        matmul_sum=matmul_sum,
        first=first_summary,
        resume=resume_summary,
        first_checkpoint_step=first_step,
        resume_checkpoint_step=resume_step,
    )
    write_results_markdown(results, spec.results_md, spec=spec)
    write_results_bundle(spec.results_bundle, spec=spec)
    return results


def make_p39_plan() -> ScalePlan:
    request = ScalePlanRequest(
        model_profile=resolve_model_profile(P39_MODEL_PROFILE),
        hardware_profile=resolve_hardware_profile(P39_HARDWARE_PROFILE),
        training_mode=resolve_training_mode(P39_TRAINING_MODE),
        sequence_length=8,
        batch_size=2,
        microbatch_size=2,
        grad_accum_steps=1,
        dtype="fp32",
        auto=False,
        minimum_sequence_length=8,
    )
    return make_plan(request)


def write_p39_planner_artifacts() -> ScalePlan:
    plan = make_p39_plan()
    P39_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    P39_SCALE_PLAN_YAML.write_text(plan_to_yaml(plan), encoding="utf-8")
    P39_SCALE_PLAN_JSON.write_text(
        json.dumps(plan_to_dict(plan), indent=2) + "\n",
        encoding="utf-8",
    )
    P39_GENERATED_DISTILL_CONFIG_PATH.write_text(
        _p39_distill_config_yaml(plan),
        encoding="utf-8",
    )
    P39_GENERATED_TEACHER_EXPORT_CONFIG_PATH.write_text(
        _p39_teacher_export_config_yaml(),
        encoding="utf-8",
    )
    return plan


def validate_p39_plan_fit(plan: ScalePlan) -> None:
    if plan.fit.fit not in {"yes", "maybe"}:
        raise ColabTpuSmokeError(
            f"P39 planner fit must be yes/maybe for tiny profile, got {plan.fit.fit!r}"
        )


def _p39_distill_config_yaml(plan: ScalePlan) -> str:
    payload = yaml.safe_load(distill_config_yaml(plan))
    if not isinstance(payload, dict):
        raise ColabTpuSmokeError("P39 generated distill config is not a mapping")
    distillation = payload.get("distillation")
    if not isinstance(distillation, dict):
        raise ColabTpuSmokeError("P39 generated distill config missing distillation")
    distillation["targets_dir"] = str(P39_TARGETS_DIR)
    student = distillation.get("student")
    if not isinstance(student, dict):
        raise ColabTpuSmokeError("P39 generated distill config missing student")
    student["emit_logits"] = True
    distillation["optimizer"] = {"type": "sgd", "learning_rate": 0.001}
    distillation["training"] = {
        "max_steps": 1,
        "seed": 0,
        "planned_batch_size": plan.request.batch_size,
        "planned_microbatch_size": plan.request.microbatch_size,
        "planned_grad_accum_steps": plan.request.grad_accum_steps,
        "planned_sequence_length": plan.request.sequence_length,
    }
    distillation["planner"] = {
        "phase": "P39",
        "model_profile": P39_MODEL_PROFILE,
        "hardware_profile": P39_HARDWARE_PROFILE,
        "training_mode": P39_TRAINING_MODE,
        "fit": plan.fit.fit,
        "planner_recommended_logits_kl": plan.recommended["logits_kl"],
        "validated_tiny_execution_logits_kl": True,
        "source": str(P39_SCALE_PLAN_JSON),
    }
    losses = distillation.get("losses")
    if not isinstance(losses, dict):
        raise ColabTpuSmokeError("P39 generated distill config missing losses")
    losses["logits_kl"] = {"enabled": True, "weight": 1.0}
    return (
        "# Generated by P39 planner TPU smoke.\n"
        "# Tiny validated execution path only; not a Qwen-scale proof.\n"
        + yaml.safe_dump(payload, sort_keys=False)
    )


def _p39_teacher_export_config_yaml() -> str:
    payload = {
        "teacher": {
            "family": "hf-causal-lm",
            "policy_label": "tiny-hf-smoke-p39",
            "fallback_label": None,
            "resolved_model_id": "sshleifer/tiny-gpt2",
            "tokenizer_id": "sshleifer/tiny-gpt2",
            "trust_remote_code": False,
            "local_files_only": False,
            "revision": None,
            "device": "cpu",
            "dtype": "auto",
        },
        "targets": {
            "sequence_length": 8,
            "hidden_size": None,
            "num_layers": None,
            "dtype": "fp32",
            "include_logits": True,
            "include_attention_targets": False,
            "vocab_size": 50257,
            "prompt_texts": [
                "QRWKV-XLA P39 planner TPU smoke",
                "Tiny GPT-2 planner-generated teacher targets",
            ],
        },
        "runtime": {
            "exporter_backend": "hf",
            "batch_size": 2,
            "num_shards": 1,
            "seed": 3939,
            "output_dir": "../teacher_targets/p39_tiny_hf_logits_smoke",
        },
    }
    return (
        "# Generated by P39 planner TPU smoke.\n"
        "# Real tiny HF target export for sshleifer/tiny-gpt2.\n"
        + yaml.safe_dump(payload, sort_keys=False)
    )


def export_smoke_targets(spec: SmokeSpec) -> Path:
    if spec.export_kind == "fake":
        return export_fake_targets(spec.targets_dir, include_logits=spec.include_logits)
    if spec.export_kind == "hf":
        if spec.teacher_export_config_path is None:
            raise ColabTpuSmokeError(f"{spec.phase} is missing teacher export config")
        return export_real_hf_targets(spec.teacher_export_config_path, spec.targets_dir)
    raise ColabTpuSmokeError(
        f"unsupported {spec.phase} smoke export_kind: {spec.export_kind!r}"
    )


def export_real_hf_targets(config_path: Path, output_dir: Path) -> Path:
    config = load_teacher_export_config(config_path)
    if config.runtime.exporter_backend != "hf":
        raise ColabTpuSmokeError(
            f"real HF smoke requires runtime.exporter_backend='hf': {config_path}"
        )
    if config.teacher.resolved_model_id != "sshleifer/tiny-gpt2":
        raise ColabTpuSmokeError(
            "P38 real HF smoke must use sshleifer/tiny-gpt2, got "
            f"{config.teacher.resolved_model_id!r}"
        )
    exporter = get_teacher_exporter("hf")
    exporter.export(ExportRequest(config=config, output_dir=output_dir))
    return output_dir


def validate_smoke_targets(targets_dir: Path, *, spec: SmokeSpec) -> None:
    if spec.export_kind == "hf":
        validate_real_hf_targets(targets_dir, phase=spec.phase)
    elif spec.include_logits:
        validate_logits_targets(targets_dir)


def validate_logits_targets(targets_dir: Path) -> None:
    manifest = read_manifest(manifest_path(targets_dir))
    if not manifest.targets.logits:
        raise ColabTpuSmokeError(
            f"logits smoke targets must declare logits=true: {targets_dir}"
        )
    shard_paths = list_shard_paths(targets_dir)
    if not shard_paths:
        raise ColabTpuSmokeError(
            f"logits smoke targets contain no shards: {targets_dir}"
        )
    arrays = read_shard(shard_paths[0])
    if "logits" not in arrays:
        raise ColabTpuSmokeError(
            f"logits smoke target shard is missing logits: {shard_paths[0]}"
        )


def validate_real_hf_targets(targets_dir: Path, *, phase: str = "P38") -> None:
    manifest = read_manifest(manifest_path(targets_dir))
    if manifest.created_by != "HFTeacherExporter":
        raise ColabTpuSmokeError(
            f"{phase} targets must be created by HFTeacherExporter: {targets_dir}"
        )
    if manifest.teacher_model_id != "sshleifer/tiny-gpt2":
        raise ColabTpuSmokeError(
            f"{phase} targets must use sshleifer/tiny-gpt2, got "
            f"{manifest.teacher_model_id!r}"
        )
    expected_flags = {
        "input_ids": manifest.targets.input_ids,
        "attention_mask": manifest.targets.attention_mask,
        "loss_mask": manifest.targets.loss_mask,
        "hidden_states": manifest.targets.hidden_states,
        "logits": manifest.targets.logits,
    }
    missing_flags = [name for name, enabled in expected_flags.items() if not enabled]
    if missing_flags:
        raise ColabTpuSmokeError(
            f"{phase} target manifest is missing required flags: {missing_flags}"
        )
    shard_paths = list_shard_paths(targets_dir)
    if not shard_paths:
        raise ColabTpuSmokeError(f"{phase} targets contain no shards: {targets_dir}")
    arrays = read_shard(shard_paths[0])
    required_arrays = (
        "input_ids",
        "attention_mask",
        "loss_mask",
        "hidden_states",
        "logits",
    )
    missing_arrays = [name for name in required_arrays if name not in arrays]
    if missing_arrays:
        raise ColabTpuSmokeError(
            f"{phase} target shard is missing required arrays: {missing_arrays}"
        )
    try:
        validate_shard_arrays(
            arrays,
            sequence_length=manifest.sequence_length,
            hidden_size=manifest.hidden_size,
            num_layers=manifest.num_layers,
            require_hidden_states=True,
            require_logits=True,
            require_attention_targets=False,
        )
    except ValueError as exc:
        raise ColabTpuSmokeError(f"{phase} target shapes are invalid: {exc}") from exc
    input_ids = arrays["input_ids"]
    hidden_states = arrays["hidden_states"]
    logits = arrays["logits"]
    if int(input_ids.shape[0]) <= 0:
        raise ColabTpuSmokeError(f"{phase} target batch dimension must be positive")
    if int(logits.shape[2]) <= 0:
        raise ColabTpuSmokeError(f"{phase} logits vocab dimension must be positive")
    for name, array in (("hidden_states", hidden_states), ("logits", logits)):
        if not math.isfinite(float(array.mean())):
            raise ColabTpuSmokeError(f"{phase} {name} mean must be finite")


def validate_smoke_outputs(
    *,
    first: SmokeRunArtifacts,
    resume: SmokeRunArtifacts,
    first_summary: dict[str, Any],
    resume_summary: dict[str, Any],
    first_metrics: dict[str, float],
    resume_metrics: dict[str, float],
    first_checkpoint_step: int,
    resume_checkpoint_step: int,
    expected_resume_from: Path = FIRST_CHECKPOINT,
    metric_names: tuple[str, ...] = ("loss", "hidden_mse"),
    phase: str = "P36",
) -> None:
    for artifacts in (first, resume):
        validate_required_artifacts(artifacts, phase=phase)
    _assert_summary_progression(
        first_summary,
        start=0,
        end=1,
        metric_names=metric_names,
    )
    _assert_summary_progression(
        resume_summary,
        start=1,
        end=2,
        metric_names=metric_names,
    )
    _assert_metric_values(first_metrics, optimizer_step=1, metric_names=metric_names)
    _assert_metric_values(resume_metrics, optimizer_step=2, metric_names=metric_names)
    if first_checkpoint_step != 1:
        raise ColabTpuSmokeError(
            f"first checkpoint step must be 1, got {first_checkpoint_step}"
        )
    if resume_checkpoint_step != 2:
        raise ColabTpuSmokeError(
            f"resume checkpoint step must be 2, got {resume_checkpoint_step}"
        )
    if resume_summary.get("resume_from") != str(expected_resume_from):
        raise ColabTpuSmokeError(
            "resume summary does not record the first checkpoint as resume_from"
        )


def validate_required_artifacts(
    artifacts: SmokeRunArtifacts,
    *,
    phase: str = "P36",
) -> None:
    required = (
        artifacts.checkpoint_dir / "checkpoint.json",
        artifacts.checkpoint_dir / "params.npz",
        artifacts.run_json,
        artifacts.metrics_jsonl,
        artifacts.summary_json,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ColabTpuSmokeError(
            f"missing required {phase} artifacts: " + ", ".join(missing)
        )


def read_run_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ColabTpuSmokeError(f"run summary is missing summary mapping: {path}")
    return summary


def read_last_metric(path: Path) -> dict[str, float]:
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not lines:
        raise ColabTpuSmokeError(f"metrics file is empty: {path}")
    payload = json.loads(lines[-1])
    values = payload.get("values")
    if not isinstance(values, dict):
        raise ColabTpuSmokeError(f"metrics payload is missing values mapping: {path}")
    return {
        key: float(value)
        for key, value in values.items()
        if isinstance(value, int | float)
    }


def write_results_markdown(
    results: SmokeResults,
    path: Path,
    *,
    spec: SmokeSpec = P36_SPEC,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {spec.result_title}",
        "",
        spec.result_scope,
        spec.result_limits,
        "",
        "## Runtime",
        "",
        f"- Python: {results.runtime.python}",
        f"- JAX: {results.runtime.jax_version}",
        f"- Backend: {results.runtime.default_backend}",
        f"- Git commit: {results.runtime.git_commit}",
        f"- Git dirty: {results.runtime.git_dirty}",
        f"- Tiny matmul sum: {results.matmul_sum:.8f}",
        "",
    ]
    if spec.phase == "P39":
        lines.extend(_p39_results_metadata_lines())
    lines.extend(
        [
            "## Distill",
            "",
            _format_summary(
                "first",
                results.first,
                results.first_checkpoint_step,
                spec=spec,
            ),
            "",
            _format_summary(
                "resume",
                results.resume,
                results.resume_checkpoint_step,
                spec=spec,
            ),
            "",
        ]
    )
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return path


def write_results_bundle(path: Path, *, spec: SmokeSpec = P36_SPEC) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle_inputs = [
        spec.config_path,
        spec.teacher_export_config_path,
        spec.targets_dir / "manifest.json",
        spec.first_checkpoint / "checkpoint.json",
        spec.first_checkpoint / "params.npz",
        spec.resume_checkpoint / "checkpoint.json",
        spec.resume_checkpoint / "params.npz",
        spec.first_run_dir / "run.json",
        spec.first_run_dir / "metrics.jsonl",
        spec.first_run_dir / "summary.json",
        spec.resume_run_dir / "run.json",
        spec.resume_run_dir / "metrics.jsonl",
        spec.resume_run_dir / "summary.json",
        spec.results_md,
    ]
    if spec.phase == "P39":
        bundle_inputs.extend(
            [
                P39_SCALE_PLAN_YAML,
                P39_SCALE_PLAN_JSON,
                P39_GENERATED_DISTILL_CONFIG_PATH,
                P39_GENERATED_TEACHER_EXPORT_CONFIG_PATH,
            ]
        )
    shard_paths = sorted((spec.targets_dir / "shards").glob("shard_*.npz"))
    with tarfile.open(path, "w:gz") as archive:
        for item in [*bundle_inputs, *shard_paths]:
            if item is not None and item.is_file():
                archive.add(item, arcname=str(item))
    return path


def _p39_results_metadata_lines() -> list[str]:
    lines = [
        "## Planner",
        "",
        f"- Model profile: {P39_MODEL_PROFILE}",
        f"- Hardware profile: {P39_HARDWARE_PROFILE}",
        f"- Training mode: {P39_TRAINING_MODE}",
        f"- Scale plan YAML: {P39_SCALE_PLAN_YAML}",
        f"- Scale plan JSON: {P39_SCALE_PLAN_JSON}",
        f"- Generated distill config: {P39_GENERATED_DISTILL_CONFIG_PATH}",
        (
            "- Teacher export config: generated at "
            f"{P39_GENERATED_TEACHER_EXPORT_CONFIG_PATH}"
        ),
        "- Export config source: generated, not reused from P38",
    ]
    if P39_SCALE_PLAN_JSON.is_file():
        payload = json.loads(P39_SCALE_PLAN_JSON.read_text(encoding="utf-8"))
        fit = payload.get("fit", {})
        selected = payload.get("selected", {})
        if isinstance(fit, dict):
            lines.extend(
                [
                    f"- Planner fit: {fit.get('fit')}",
                    f"- Estimated total GiB: {fit.get('estimated_total_gb')}",
                    f"- Planner utilization: {fit.get('utilization')}",
                ]
            )
        if isinstance(selected, dict):
            lines.extend(
                [
                    f"- Planned batch size: {selected.get('batch_size')}",
                    f"- Planned sequence length: {selected.get('sequence_length')}",
                ]
            )
    lines.append("")
    return lines


def _run_one_step(
    *,
    spec: SmokeSpec,
    config_path: Path,
    checkpoint_dir: Path,
    stable_run_dir: Path,
    run_name: str,
    resume_from: Path | None,
) -> tuple[SmokeRunArtifacts, dict[str, Any], dict[str, float]]:
    from dataclasses import replace

    _remove_path(checkpoint_dir)
    _remove_path(stable_run_dir)
    config = load_distill_stage_config(config_path)
    config = replace(
        config,
        targets_dir=spec.targets_dir,
        checkpoint=DistillCheckpointConfig(
            checkpoint_out=checkpoint_dir,
            resume_from=resume_from,
            overwrite=True,
        ),
        tracking=DistillTrackingConfig(
            enabled=True,
            run_root=spec.run_root,
            run_name=run_name,
            tags=list(spec.tracking_tags),
            notes=list(spec.tracking_notes),
            overwrite=True,
        ),
    )
    result = run_distill_stage(config)
    if result.run_dir is None:
        raise ColabTpuSmokeError("distill tracking did not produce a run directory")
    _move_run_dir(result.run_dir, stable_run_dir)
    artifacts = SmokeRunArtifacts(
        checkpoint_dir=checkpoint_dir,
        run_dir=stable_run_dir,
        run_json=stable_run_dir / "run.json",
        metrics_jsonl=stable_run_dir / "metrics.jsonl",
        summary_json=stable_run_dir / "summary.json",
    )
    validate_required_artifacts(artifacts, phase=spec.phase)
    return (
        artifacts,
        read_run_summary(artifacts.summary_json),
        read_last_metric(artifacts.metrics_jsonl),
    )


def _assert_summary_progression(
    summary: dict[str, Any],
    *,
    start: int,
    end: int,
    metric_names: tuple[str, ...],
) -> None:
    summary_names = tuple(
        "final_loss" if name == "loss" else f"final_{name}" for name in metric_names
    )
    for name in summary_names:
        value = summary.get(name)
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ColabTpuSmokeError(f"{name} must be finite, got {value!r}")
    if int(summary.get("start_step", -1)) != start:
        raise ColabTpuSmokeError(
            f"start_step must be {start}, got {summary.get('start_step')!r}"
        )
    if int(summary.get("end_step", -1)) != end:
        raise ColabTpuSmokeError(
            f"end_step must be {end}, got {summary.get('end_step')!r}"
        )


def _assert_metric_values(
    metrics: dict[str, float],
    *,
    optimizer_step: int,
    metric_names: tuple[str, ...],
) -> None:
    for name in metric_names:
        value = metrics.get(name)
        if value is None or not math.isfinite(float(value)):
            raise ColabTpuSmokeError(f"{name} metric must be finite, got {value!r}")
    actual_optimizer_step = metrics.get("optimizer_step")
    if actual_optimizer_step is None or int(actual_optimizer_step) != optimizer_step:
        raise ColabTpuSmokeError(
            "optimizer_step metric must be "
            f"{optimizer_step}, got {actual_optimizer_step!r}"
        )


def _format_summary(
    name: str,
    summary: dict[str, Any],
    checkpoint_step: int,
    *,
    spec: SmokeSpec,
) -> str:
    lines = [
        f"### {name}",
        "",
        f"- start_step: {summary.get('start_step')}",
        f"- end_step: {summary.get('end_step')}",
        f"- checkpoint_step: {checkpoint_step}",
        f"- final_loss: {float(summary['final_loss']):.8f}",
        f"- final_hidden_mse: {float(summary['final_hidden_mse']):.8f}",
    ]
    if spec.include_logits:
        lines.append(f"- final_logits_kl: {float(summary['final_logits_kl']):.8f}")
    run_dir = spec.first_run_dir if name == "first" else spec.resume_run_dir
    lines.extend(
        [
            f"- checkpoint_out: {summary.get('checkpoint_out')}",
            f"- run_dir: {run_dir}",
        ]
    )
    return "\n".join(lines)


def _move_run_dir(source: Path, stable: Path) -> None:
    if source == stable:
        return
    _remove_path(stable)
    stable.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(stable))


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _git_output(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _format_device(device: object) -> str:
    platform_name = str(getattr(device, "platform", "unknown"))
    device_id = getattr(device, "id", "?")
    kind = str(getattr(device, "device_kind", type(device).__name__))
    process_index = getattr(device, "process_index", None)
    if callable(process_index):
        process_index = process_index()
    suffix = "" if process_index is None else f", process_index={process_index}"
    return f"id={device_id}, platform={platform_name}, kind={kind}{suffix}"
