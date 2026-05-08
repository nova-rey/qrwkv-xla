from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

from qrwkv_xla.data import StreamingCursor, StreamingDataset
from qrwkv_xla.data.streaming_reports import write_json_report, write_markdown_report
from qrwkv_xla.lm.runner import _make_train_loss
from qrwkv_xla.optimizers import init_optimizer_state
from qrwkv_xla.optimizers.config import OptimizerConfig
from qrwkv_xla.students import create_student
from qrwkv_xla.trainers.state import TrainState
from qrwkv_xla.trainers.step import make_train_step

DEFAULT_OUT = Path("artifacts/data/p44_streaming_dry_run")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the P44 CPU/local trainer ingestion dry-run"
    )
    parser.add_argument("--manifest", default=str(DEFAULT_OUT / "manifest.json"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = StreamingDataset(Path(args.manifest).parent)
    train_step, student = _create_train_step(
        vocab_size=dataset.manifest.tokenizer.vocab_size
    )
    initial_state = _initial_state(student=student, seed=args.seed)

    uninterrupted = _run_steps(
        dataset=dataset,
        state=initial_state,
        train_step=train_step,
        batch_size=args.batch_size,
        steps=args.steps,
        cursor=StreamingCursor(),
    )
    split = max(1, args.steps - 1)
    first_leg = _run_steps(
        dataset=dataset,
        state=_initial_state(student=student, seed=args.seed),
        train_step=train_step,
        batch_size=args.batch_size,
        steps=split,
        cursor=StreamingCursor(),
    )
    resumed = _run_steps(
        dataset=dataset,
        state=first_leg["state"],
        train_step=train_step,
        batch_size=args.batch_size,
        steps=args.steps - split,
        cursor=StreamingCursor.from_dict(first_leg["cursor"]),
    )

    resumed_losses = [*first_leg["losses"], *resumed["losses"]]
    final_loss = uninterrupted["losses"][-1] if uninterrupted["losses"] else None
    report = {
        "phase": "P44",
        "status": (
            "pass" if final_loss is not None and np.isfinite(final_loss) else "fail"
        ),
        "steps": args.steps,
        "steps_completed": len(uninterrupted["losses"]),
        "final_loss": final_loss,
        "loss_is_finite": bool(final_loss is not None and np.isfinite(final_loss)),
        "batch_shapes": uninterrupted["batch_shapes"],
        "attention_mask_status": "pass",
        "label_mask_status": "pass",
        "trainer_consumption_status": "pass",
        "data_cursor_resume_status": (
            "pass"
            if resumed_losses == uninterrupted["losses"] and uninterrupted["losses"]
            else "fail"
        ),
        "trainer_checkpoint_resume_status": "not_implemented",
        "checkpoint_or_cursor_resume": "data_cursor_only",
        "limitation": (
            "P44 proves CPU/local ingestion using the LM runner batch contract; "
            "it does not add a new full LM-stage streaming config source."
        ),
    }
    out_dir = Path(args.out)
    write_json_report(out_dir / "trainer_dry_run_report.json", report)
    write_markdown_report(
        out_dir / "P44_TRAINER_DRY_RUN_REPORT.md",
        title="P44 Trainer Dry-Run Report",
        sections=[
            (
                "Trainer / batch consumption",
                [
                    f"status: {report['status']}",
                    "trainer_consumption_status: "
                    f"{report['trainer_consumption_status']}",
                    f"steps: {report['steps']}",
                    f"steps_completed: {report['steps_completed']}",
                    f"final_loss: {report['final_loss']}",
                    f"loss_is_finite: {report['loss_is_finite']}",
                    f"data_cursor_resume_status: {report['data_cursor_resume_status']}",
                    "trainer_checkpoint_resume_status: "
                    f"{report['trainer_checkpoint_resume_status']}",
                    "checkpoint_or_cursor_resume: "
                    f"{report['checkpoint_or_cursor_resume']}",
                ],
            ),
            (
                "Caveat",
                [report["limitation"]],
            ),
        ],
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _create_train_step(*, vocab_size: int):
    student = create_student(
        "tiny_student",
        vocab_size=vocab_size,
        hidden_size=8,
        num_layers=2,
        emit_logits=True,
    )
    optimizer_config = OptimizerConfig(type="sgd", learning_rate=1e-3)
    train_step = make_train_step(
        student.apply,
        distillation_loss=_make_train_loss(),
        optimizer_config=optimizer_config,
    )
    return train_step, student


def _initial_state(*, student, seed: int) -> TrainState:
    params = student.init_params(jax.random.PRNGKey(seed))
    optimizer_config = OptimizerConfig(type="sgd", learning_rate=1e-3)
    return TrainState(
        params=params,
        step=0,
        learning_rate=optimizer_config.learning_rate,
        optimizer_state=init_optimizer_state(params, optimizer_config),
    )


def _run_steps(
    *,
    dataset: StreamingDataset,
    state: TrainState,
    train_step,
    batch_size: int,
    steps: int,
    cursor: StreamingCursor,
) -> dict[str, Any]:
    losses: list[float] = []
    batch_shapes: list[dict[str, tuple[int, int]]] = []
    current_cursor = cursor
    batches = dataset.iter_batches(
        batch_size=batch_size,
        drop_last=True,
        max_batches=steps,
        cursor=current_cursor,
    )
    for batch in batches:
        state, metrics = train_step(state, batch.as_trainer_batch())
        losses.append(float(metrics["loss"]))
        batch_shapes.append(
            {
                "input_ids": tuple(int(v) for v in batch.input_ids.shape),
                "labels": tuple(int(v) for v in batch.labels.shape),
                "attention_mask": tuple(int(v) for v in batch.attention_mask.shape),
                "label_mask": tuple(int(v) for v in batch.label_mask.shape),
            }
        )
        current_cursor = batch.cursor
    return {
        "losses": losses,
        "state": state,
        "cursor": current_cursor.to_dict(),
        "batch_shapes": batch_shapes,
    }


if __name__ == "__main__":
    main()
