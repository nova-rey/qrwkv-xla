from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from qrwkv_xla.xla.inspect import get_jax_runtime_info


@dataclass(frozen=True)
class XlaSmokeResult:
    student_architecture: str
    steps: int
    initial_loss: float
    final_loss: float
    backend: str
    device_count: int
    has_tpu: bool


def run_xla_distill_smoke(
    *,
    targets_dir: str | Path,
    student_architecture: str = "rwkv7_reference",
    max_steps: int = 2,
    seed: int = 0,
    learning_rate: float = 1e-3,
    require_tpu: bool = False,
) -> XlaSmokeResult:
    runtime = get_jax_runtime_info()
    if not runtime.jax_available:
        raise RuntimeError("JAX is not available")
    if require_tpu and not runtime.has_tpu:
        raise RuntimeError("TPU is required for this smoke run but no TPU was detected")

    from qrwkv_xla.distill import (
        DistillStageConfig,
        DistillStudentConfig,
        run_distill_stage,
    )

    config = DistillStageConfig(
        targets_dir=Path(targets_dir),
        student=DistillStudentConfig(
            architecture=student_architecture,
            vocab_size=512,
            hidden_size=None,
            num_layers=None,
        ),
    )
    config = replace(
        config,
        training=replace(config.training, max_steps=max_steps, seed=seed),
        optimizer=replace(config.optimizer, learning_rate=learning_rate),
    )
    result = run_distill_stage(config)
    return XlaSmokeResult(
        student_architecture=result.student_architecture,
        steps=result.steps,
        initial_loss=result.initial_loss,
        final_loss=result.final_loss,
        backend=runtime.default_backend or "unknown",
        device_count=len(runtime.devices),
        has_tpu=runtime.has_tpu,
    )
