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

from qrwkv_xla.checkpointing import load_checkpoint
from qrwkv_xla.distill import (
    DistillCheckpointConfig,
    DistillTrackingConfig,
    load_distill_stage_config,
    run_distill_stage,
)
from qrwkv_xla.targets import read_manifest
from qrwkv_xla.targets.shards import read_shard
from qrwkv_xla.targets.store import list_shard_paths, manifest_path
from qrwkv_xla.teacher_export import (
    ExportRequest,
    FakeTeacherExporter,
    TeacherExportConfig,
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


def _run_distill_smoke_pair(spec: SmokeSpec, *, config_path: Path) -> SmokeResults:
    runtime = collect_runtime_summary(Path.cwd())
    print(format_runtime_summary(runtime))
    assert_tpu_backend(runtime)
    matmul_sum = run_tiny_matmul_sanity()
    print(f"tiny_matmul_sum: {matmul_sum:.8f}")

    export_fake_targets(spec.targets_dir, include_logits=spec.include_logits)
    if spec.include_logits:
        validate_logits_targets(spec.targets_dir)
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
    text = "\n".join(
        [
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
    path.write_text(text, encoding="utf-8")
    return path


def write_results_bundle(path: Path, *, spec: SmokeSpec = P36_SPEC) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle_inputs = [
        spec.config_path,
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
    shard_paths = sorted((spec.targets_dir / "shards").glob("shard_*.npz"))
    with tarfile.open(path, "w:gz") as archive:
        for item in [*bundle_inputs, *shard_paths]:
            if item.is_file():
                archive.add(item, arcname=str(item))
    return path


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
