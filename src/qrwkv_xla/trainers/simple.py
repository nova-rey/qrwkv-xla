from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax

from qrwkv_xla.datasets.target_bundle import TargetBundleDataset
from qrwkv_xla.students.tiny import TinyStudent
from qrwkv_xla.trainers.state import TrainState
from qrwkv_xla.trainers.step import batch_to_jax, make_train_step


@dataclass(frozen=True)
class SimpleTrainResult:
    initial_loss: float
    final_loss: float
    steps: int


def train_on_bundle_once(
    *,
    bundle_dir: str | Path,
    student: TinyStudent,
    seed: int = 0,
    learning_rate: float = 1e-3,
    max_steps: int = 5,
) -> SimpleTrainResult:
    if max_steps <= 0:
        raise ValueError(f"max_steps must be > 0, got {max_steps}")

    dataset = TargetBundleDataset.from_path(bundle_dir)
    state = TrainState(
        params=student.init_params(jax.random.PRNGKey(seed)),
        step=0,
        learning_rate=learning_rate,
    )
    train_step = make_train_step(student.apply)
    initial_loss: float | None = None
    final_loss: float | None = None
    steps = 0

    for batch in dataset.iter_shards():
        if steps >= max_steps:
            break
        state, metrics = train_step(state, batch_to_jax(batch))
        loss_value = float(metrics["loss"])
        if initial_loss is None:
            initial_loss = loss_value
        final_loss = loss_value
        steps += 1

    if steps == 0 or initial_loss is None or final_loss is None:
        raise ValueError(f"Target bundle contains no shards: {dataset.bundle_dir}")
    return SimpleTrainResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        steps=steps,
    )
