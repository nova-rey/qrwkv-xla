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


class ColabTpuSmokeError(RuntimeError):
    """Raised when the manual Colab TPU smoke cannot prove its contract."""


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


def export_fake_targets(output_dir: Path = TARGETS_DIR) -> Path:
    from dataclasses import replace

    config = TeacherExportConfig()
    config = replace(
        config,
        targets=replace(
            config.targets,
            sequence_length=8,
            hidden_size=8,
            num_layers=2,
            include_logits=False,
            include_attention_targets=False,
            vocab_size=512,
        ),
        runtime=replace(
            config.runtime,
            output_dir=output_dir,
            batch_size=1,
            num_shards=1,
            seed=3636,
        ),
    )
    FakeTeacherExporter().export(ExportRequest(config=config, output_dir=output_dir))
    return output_dir


def run_distill_smoke_pair(config_path: Path = CONFIG_PATH) -> SmokeResults:
    runtime = collect_runtime_summary(Path.cwd())
    print(format_runtime_summary(runtime))
    assert_tpu_backend(runtime)
    matmul_sum = run_tiny_matmul_sanity()
    print(f"tiny_matmul_sum: {matmul_sum:.8f}")

    export_fake_targets(TARGETS_DIR)
    first_artifacts, first_summary, first_metrics = _run_one_step(
        config_path=config_path,
        checkpoint_dir=FIRST_CHECKPOINT,
        stable_run_dir=FIRST_RUN_DIR,
        run_name="p36_tpu_qwen_reference_first",
        resume_from=None,
    )
    resume_artifacts, resume_summary, resume_metrics = _run_one_step(
        config_path=config_path,
        checkpoint_dir=RESUME_CHECKPOINT,
        stable_run_dir=RESUME_RUN_DIR,
        run_name="p36_tpu_qwen_reference_resume",
        resume_from=FIRST_CHECKPOINT,
    )
    first_step = load_checkpoint(FIRST_CHECKPOINT).manifest.step
    resume_step = load_checkpoint(RESUME_CHECKPOINT).manifest.step

    validate_smoke_outputs(
        first=first_artifacts,
        resume=resume_artifacts,
        first_summary=first_summary,
        resume_summary=resume_summary,
        first_metrics=first_metrics,
        resume_metrics=resume_metrics,
        first_checkpoint_step=first_step,
        resume_checkpoint_step=resume_step,
    )
    results = SmokeResults(
        runtime=runtime,
        matmul_sum=matmul_sum,
        first=first_summary,
        resume=resume_summary,
        first_checkpoint_step=first_step,
        resume_checkpoint_step=resume_step,
    )
    write_results_markdown(results, RESULTS_MD)
    write_results_bundle(RESULTS_BUNDLE)
    return results


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
) -> None:
    for artifacts in (first, resume):
        validate_required_artifacts(artifacts)
    _assert_summary_progression(first_summary, start=0, end=1)
    _assert_summary_progression(resume_summary, start=1, end=2)
    _assert_metric_values(first_metrics, optimizer_step=1)
    _assert_metric_values(resume_metrics, optimizer_step=2)
    if first_checkpoint_step != 1:
        raise ColabTpuSmokeError(
            f"first checkpoint step must be 1, got {first_checkpoint_step}"
        )
    if resume_checkpoint_step != 2:
        raise ColabTpuSmokeError(
            f"resume checkpoint step must be 2, got {resume_checkpoint_step}"
        )
    if resume_summary.get("resume_from") != str(FIRST_CHECKPOINT):
        raise ColabTpuSmokeError(
            "resume summary does not record the first checkpoint as resume_from"
        )


def validate_required_artifacts(artifacts: SmokeRunArtifacts) -> None:
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
            "missing required P36 artifacts: " + ", ".join(missing)
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


def write_results_markdown(results: SmokeResults, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "# P36 Colab TPU Smoke Results",
            "",
            (
                "This artifact proves a tiny manual TPU hidden-MSE train/resume "
                "smoke only."
            ),
            (
                "It does not prove model quality, Qwen-scale fit, pjit sharding, "
                "Pallas kernels, or logits KL."
            ),
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
            _format_summary("first", results.first, results.first_checkpoint_step),
            "",
            _format_summary("resume", results.resume, results.resume_checkpoint_step),
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    return path


def write_results_bundle(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle_inputs = [
        CONFIG_PATH,
        TARGETS_DIR / "manifest.json",
        FIRST_CHECKPOINT / "checkpoint.json",
        FIRST_CHECKPOINT / "params.npz",
        RESUME_CHECKPOINT / "checkpoint.json",
        RESUME_CHECKPOINT / "params.npz",
        FIRST_RUN_DIR / "run.json",
        FIRST_RUN_DIR / "metrics.jsonl",
        FIRST_RUN_DIR / "summary.json",
        RESUME_RUN_DIR / "run.json",
        RESUME_RUN_DIR / "metrics.jsonl",
        RESUME_RUN_DIR / "summary.json",
        RESULTS_MD,
    ]
    with tarfile.open(path, "w:gz") as archive:
        for item in bundle_inputs:
            if item.is_file():
                archive.add(item, arcname=str(item))
    return path


def _run_one_step(
    *,
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
        targets_dir=TARGETS_DIR,
        checkpoint=DistillCheckpointConfig(
            checkpoint_out=checkpoint_dir,
            resume_from=resume_from,
            overwrite=True,
        ),
        tracking=DistillTrackingConfig(
            enabled=True,
            run_root=RUN_ROOT,
            run_name=run_name,
            tags=["p36", "colab-tpu-smoke", "manual"],
            notes=[
                "P36 manual Colab TPU smoke",
                "tiny hidden-only rwkv7_qwen_reference train/resume proof",
            ],
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
    validate_required_artifacts(artifacts)
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
) -> None:
    for name in ("final_loss", "final_hidden_mse"):
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


def _assert_metric_values(metrics: dict[str, float], *, optimizer_step: int) -> None:
    for name in ("loss", "hidden_mse"):
        value = metrics.get(name)
        if value is None or not math.isfinite(float(value)):
            raise ColabTpuSmokeError(f"{name} metric must be finite, got {value!r}")
    actual_optimizer_step = metrics.get("optimizer_step")
    if actual_optimizer_step is None or int(actual_optimizer_step) != optimizer_step:
        raise ColabTpuSmokeError(
            "optimizer_step metric must be "
            f"{optimizer_step}, got {actual_optimizer_step!r}"
        )


def _format_summary(name: str, summary: dict[str, Any], checkpoint_step: int) -> str:
    return "\n".join(
        [
            f"### {name}",
            "",
            f"- start_step: {summary.get('start_step')}",
            f"- end_step: {summary.get('end_step')}",
            f"- checkpoint_step: {checkpoint_step}",
            f"- final_loss: {float(summary['final_loss']):.8f}",
            f"- final_hidden_mse: {float(summary['final_hidden_mse']):.8f}",
            f"- checkpoint_out: {summary.get('checkpoint_out')}",
            f"- run_dir: {FIRST_RUN_DIR if name == 'first' else RESUME_RUN_DIR}",
        ]
    )


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
