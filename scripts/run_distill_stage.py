from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a QRWKV-XLA distillation stage")
    parser.add_argument("--config", default="configs/distill_stage0_stub.yaml")
    parser.add_argument("--targets")
    parser.add_argument(
        "--student-architecture",
        choices=("tiny_student", "rwkv7_reference"),
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint-out")
    parser.add_argument("--resume-from")
    parser.add_argument("--checkpoint-overwrite", action="store_true")
    args = parser.parse_args()

    from qrwkv_xla.distill import load_distill_stage_config, run_distill_stage

    config = load_distill_stage_config(args.config)
    if args.targets:
        config = replace(config, targets_dir=Path(args.targets))
    if args.student_architecture:
        config = replace(
            config,
            student=replace(config.student, architecture=args.student_architecture),
        )
    if args.max_steps is not None:
        config = replace(
            config, training=replace(config.training, max_steps=args.max_steps)
        )
    if args.learning_rate is not None:
        config = replace(
            config,
            optimizer=replace(config.optimizer, learning_rate=args.learning_rate),
        )
    if args.seed is not None:
        config = replace(config, training=replace(config.training, seed=args.seed))
    if (
        args.checkpoint_out is not None
        or args.resume_from is not None
        or args.checkpoint_overwrite
    ):
        config = replace(
            config,
            checkpoint=replace(
                config.checkpoint,
                checkpoint_out=(
                    Path(args.checkpoint_out)
                    if args.checkpoint_out is not None
                    else config.checkpoint.checkpoint_out
                ),
                resume_from=(
                    Path(args.resume_from)
                    if args.resume_from is not None
                    else config.checkpoint.resume_from
                ),
                overwrite=args.checkpoint_overwrite or config.checkpoint.overwrite,
            ),
        )

    result = run_distill_stage(config)
    print(f"stage: {result.stage}")
    print(f"student_architecture: {result.student_architecture}")
    print(f"targets: {result.target_bundle}")
    print(f"steps: {result.steps}")
    print(f"start_step: {result.start_step}")
    print(f"end_step: {result.end_step}")
    if result.resume_from is not None:
        print(f"resume_from: {result.resume_from}")
    if result.checkpoint_out is not None:
        print(f"checkpoint_out: {result.checkpoint_out}")
    print(f"initial_loss: {result.initial_loss:.8f}")
    print(f"final_loss: {result.final_loss:.8f}")
    if result.final_hidden_mse is not None:
        print(f"final_hidden_mse: {result.final_hidden_mse:.8f}")
    if result.final_logits_kl is not None:
        print(f"final_logits_kl: {result.final_logits_kl:.8f}")


if __name__ == "__main__":
    try:
        main()
    except (NotImplementedError, ValueError) as exc:
        print(f"Distillation stage failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
