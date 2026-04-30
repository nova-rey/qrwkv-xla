from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax

from qrwkv_xla.datasets.target_bundle import TargetBundleDataset
from qrwkv_xla.students.base import StudentModel
from qrwkv_xla.trainers.state import TrainState
from qrwkv_xla.trainers.step import batch_to_jax, make_train_step


@dataclass(frozen=True)
class SimpleTrainResult:
    initial_loss: float
    final_loss: float
    steps: int


@dataclass(frozen=True)
class TrainHistoryResult(SimpleTrainResult):
    history: tuple[dict[str, float], ...]


def train_on_bundle_once(
    *,
    bundle_dir: str | Path,
    student: StudentModel,
    seed: int = 0,
    learning_rate: float = 1e-3,
    max_steps: int = 5,
    distillation_loss: Any | None = None,
    return_history: bool = False,
) -> SimpleTrainResult | TrainHistoryResult:
    if max_steps <= 0:
        raise ValueError(f"max_steps must be > 0, got {max_steps}")

    dataset = TargetBundleDataset.from_path(bundle_dir)
    batches = tuple(dataset.iter_shards())
    if not batches:
        raise ValueError(f"Target bundle contains no shards: {dataset.bundle_dir}")
    state = TrainState(
        params=student.init_params(jax.random.PRNGKey(seed)),
        step=0,
        learning_rate=learning_rate,
    )
    train_step = make_train_step(
        student.apply,
        distillation_loss=_typed_distillation_loss(distillation_loss),
    )
    initial_loss: float | None = None
    final_loss: float | None = None
    history: list[dict[str, float]] = []
    steps = 0

    for steps in range(max_steps):
        batch = batches[steps % len(batches)]
        state, metrics = train_step(state, batch_to_jax(batch))
        float_metrics = {name: float(value) for name, value in metrics.items()}
        loss_value = float_metrics["loss"]
        if initial_loss is None:
            initial_loss = loss_value
        final_loss = loss_value
        if return_history:
            history.append(float_metrics)
    step_count = steps + 1
    if initial_loss is None or final_loss is None:
        raise ValueError(f"Target bundle contains no shards: {dataset.bundle_dir}")
    result_kwargs = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "steps": step_count,
    }
    if return_history:
        return TrainHistoryResult(
            **result_kwargs,
            history=tuple(history),
        )
    return SimpleTrainResult(
        **result_kwargs,
    )


def _typed_distillation_loss(value: Any) -> Any:
    if value is None:
        return None
    return value
