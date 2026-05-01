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
    parser.add_argument("--track-run", action="store_true")
    parser.add_argument("--run-root")
    parser.add_argument("--run-name")
    parser.add_argument("--run-tag", action="append", default=[])
    parser.add_argument("--run-note", action="append", default=[])
    parser.add_argument("--run-overwrite", action="store_true")
    parser.add_argument("--emit-logits", action="store_true")
    parser.add_argument("--tie-embeddings", action="store_true")
    parser.add_argument("--enable-logits-kl", action="store_true")
    parser.add_argument("--logits-kl-weight", type=float)
    parser.add_argument("--hidden-mse-weight", type=float)
    args = parser.parse_args()

    from qrwkv_xla.distill import (
        LossWeightConfig,
        load_distill_stage_config,
        run_distill_stage,
    )

    config = load_distill_stage_config(args.config)
    if args.targets:
        config = replace(config, targets_dir=Path(args.targets))
    if args.student_architecture:
        config = replace(
            config,
            student=replace(config.student, architecture=args.student_architecture),
        )
    if args.emit_logits or args.tie_embeddings:
        config = replace(
            config,
            student=replace(
                config.student,
                emit_logits=args.emit_logits or config.student.emit_logits,
                tie_embeddings=args.tie_embeddings or config.student.tie_embeddings,
            ),
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
    if (
        args.track_run
        or args.run_root is not None
        or args.run_name is not None
        or args.run_tag
        or args.run_note
        or args.run_overwrite
    ):
        config = replace(
            config,
            tracking=replace(
                config.tracking,
                enabled=args.track_run or config.tracking.enabled,
                run_root=(
                    Path(args.run_root)
                    if args.run_root is not None
                    else config.tracking.run_root
                ),
                run_name=args.run_name
                if args.run_name is not None
                else config.tracking.run_name,
                tags=[*config.tracking.tags, *args.run_tag],
                notes=[*config.tracking.notes, *args.run_note],
                overwrite=args.run_overwrite or config.tracking.overwrite,
            ),
        )
    if (
        args.enable_logits_kl
        or args.logits_kl_weight is not None
        or args.hidden_mse_weight is not None
    ):
        logits_weight = (
            args.logits_kl_weight
            if args.logits_kl_weight is not None
            else config.losses.logits_kl.weight
        )
        hidden_weight = (
            args.hidden_mse_weight
            if args.hidden_mse_weight is not None
            else config.losses.hidden_mse.weight
        )
        config = replace(
            config,
            losses=replace(
                config.losses,
                hidden_mse=LossWeightConfig(
                    enabled=config.losses.hidden_mse.enabled,
                    weight=hidden_weight,
                ),
                logits_kl=LossWeightConfig(
                    enabled=args.enable_logits_kl or config.losses.logits_kl.enabled,
                    weight=logits_weight,
                ),
            ),
        )

    result = run_distill_stage(config)
    print(f"stage: {result.stage}")
    print(f"student_architecture: {result.student_architecture}")
    print(f"emit_logits: {config.student.emit_logits}")
    print(f"tie_embeddings: {config.student.tie_embeddings}")
    print(
        "logits_kl_enabled: "
        f"{config.losses.logits_kl.enabled and config.losses.logits_kl.weight > 0}"
    )
    print(f"targets: {result.target_bundle}")
    print(f"steps: {result.steps}")
    print(f"start_step: {result.start_step}")
    print(f"end_step: {result.end_step}")
    if result.resume_from is not None:
        print(f"resume_from: {result.resume_from}")
    if result.checkpoint_out is not None:
        print(f"checkpoint_out: {result.checkpoint_out}")
    if result.run_dir is not None:
        print(f"run_dir: {result.run_dir}")
    if result.metrics_path is not None:
        print(f"metrics_path: {result.metrics_path}")
    if result.summary_path is not None:
        print(f"summary_path: {result.summary_path}")
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
