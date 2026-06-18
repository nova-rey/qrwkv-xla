from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


def main() -> None:
    from qrwkv_xla.students.factory import STUDENT_ARCHITECTURES

    parser = argparse.ArgumentParser(description="Run a QRWKV-XLA distillation stage")
    parser.add_argument("--config", default="configs/distill_stage0_stub.yaml")
    parser.add_argument(
        "--distill-mode",
        choices=("teacher_targets", "fingerprint_corridor"),
    )
    parser.add_argument("--targets")
    parser.add_argument("--fingerprint-artifact")
    parser.add_argument("--student-backend", default=None)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--fingerprint-max-records", type=int)
    parser.add_argument("--fingerprint-drop-remainder", action="store_true")
    parser.add_argument(
        "--fingerprint-input-conditioned-rehearsal", action="store_true"
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--student-architecture",
        choices=tuple(sorted(STUDENT_ARCHITECTURES)),
    )
    parser.add_argument("--steps", type=int, dest="max_steps")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--optimizer", choices=("sgd", "adam", "adamw"))
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--adam-beta1", type=float)
    parser.add_argument("--adam-beta2", type=float)
    parser.add_argument("--adam-epsilon", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "warmup_cosine"),
    )
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--disable-grad-clipping", action="store_true")
    parser.add_argument("--clip-epsilon", type=float)
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
    parser.add_argument("--emit-mixer-outputs", action="store_true")
    parser.add_argument("--tie-embeddings", action="store_true")
    parser.add_argument("--enable-logits-kl", action="store_true")
    parser.add_argument(
        "--enable-attention-mixer",
        "--enable-attention-mixer-loss",
        dest="enable_attention_mixer",
        action="store_true",
    )
    parser.add_argument("--logits-kl-weight", type=float)
    parser.add_argument("--attention-mixer-weight", type=float)
    parser.add_argument("--hidden-mse-weight", type=float)
    parser.add_argument("--bucket-shape-loss-weight", type=float)
    parser.add_argument("--bucket-shape-loss-type", choices=("kl", "log_mse"))
    args = parser.parse_args()

    from qrwkv_xla.distill import (
        DistillFingerprintConfig,
        DistillGradientConfig,
        LossWeightConfig,
        load_distill_stage_config,
        run_distill_stage,
    )

    if args.max_grad_norm is not None and args.disable_grad_clipping:
        parser.error("--max-grad-norm conflicts with --disable-grad-clipping")

    config = load_distill_stage_config(args.config)
    if args.distill_mode is not None:
        config = replace(config, mode=args.distill_mode)
    if args.targets:
        config = replace(config, targets_dir=Path(args.targets))
    if (
        args.fingerprint_artifact is not None
        or args.student_backend is not None
        or args.batch_size is not None
        or args.fingerprint_max_records is not None
        or args.fingerprint_drop_remainder
        or args.fingerprint_input_conditioned_rehearsal
        or args.output_dir is not None
    ):
        config = replace(
            config,
            fingerprint=DistillFingerprintConfig(
                artifact_dir=(
                    Path(args.fingerprint_artifact)
                    if args.fingerprint_artifact is not None
                    else config.fingerprint.artifact_dir
                ),
                batch_size=(
                    args.batch_size
                    if args.batch_size is not None
                    else config.fingerprint.batch_size
                ),
                shuffle=config.fingerprint.shuffle,
                seed=config.fingerprint.seed,
                max_records=(
                    args.fingerprint_max_records
                    if args.fingerprint_max_records is not None
                    else config.fingerprint.max_records
                ),
                drop_remainder=(
                    args.fingerprint_drop_remainder or config.fingerprint.drop_remainder
                ),
                student_backend=(
                    args.student_backend
                    if args.student_backend is not None
                    else config.fingerprint.student_backend
                ),
                student_vocab_size=config.fingerprint.student_vocab_size,
                student_max_seq_len=config.fingerprint.student_max_seq_len,
                output_dir=(
                    Path(args.output_dir)
                    if args.output_dir is not None
                    else config.fingerprint.output_dir
                ),
                input_conditioned_rehearsal=(
                    args.fingerprint_input_conditioned_rehearsal
                    or config.fingerprint.input_conditioned_rehearsal
                ),
            ),
        )
    if args.student_architecture:
        config = replace(
            config,
            student=replace(config.student, architecture=args.student_architecture),
        )
    if args.emit_logits or args.tie_embeddings or args.emit_mixer_outputs:
        config = replace(
            config,
            student=replace(
                config.student,
                emit_logits=args.emit_logits or config.student.emit_logits,
                tie_embeddings=args.tie_embeddings or config.student.tie_embeddings,
                emit_mixer_outputs=(
                    args.emit_mixer_outputs or config.student.emit_mixer_outputs
                ),
            ),
        )
    if args.max_steps is not None:
        config = replace(
            config, training=replace(config.training, max_steps=args.max_steps)
        )
    if (
        args.optimizer is not None
        or args.learning_rate is not None
        or args.adam_beta1 is not None
        or args.adam_beta2 is not None
        or args.adam_epsilon is not None
        or args.weight_decay is not None
    ):
        config = replace(
            config,
            optimizer=replace(
                config.optimizer,
                type=(
                    args.optimizer
                    if args.optimizer is not None
                    else config.optimizer.type
                ),
                learning_rate=(
                    args.learning_rate
                    if args.learning_rate is not None
                    else config.optimizer.learning_rate
                ),
                beta1=(
                    args.adam_beta1
                    if args.adam_beta1 is not None
                    else config.optimizer.beta1
                ),
                beta2=(
                    args.adam_beta2
                    if args.adam_beta2 is not None
                    else config.optimizer.beta2
                ),
                epsilon=(
                    args.adam_epsilon
                    if args.adam_epsilon is not None
                    else config.optimizer.epsilon
                ),
                weight_decay=(
                    args.weight_decay
                    if args.weight_decay is not None
                    else config.optimizer.weight_decay
                ),
            ),
        )
    if (
        args.lr_schedule is not None
        or args.warmup_steps is not None
        or args.total_steps is not None
        or args.min_learning_rate is not None
    ):
        config = replace(
            config,
            lr_schedule=replace(
                config.lr_schedule,
                type=(
                    args.lr_schedule
                    if args.lr_schedule is not None
                    else config.lr_schedule.type
                ),
                warmup_steps=(
                    args.warmup_steps
                    if args.warmup_steps is not None
                    else config.lr_schedule.warmup_steps
                ),
                total_steps=(
                    args.total_steps
                    if args.total_steps is not None
                    else config.lr_schedule.total_steps
                ),
                min_learning_rate=(
                    args.min_learning_rate
                    if args.min_learning_rate is not None
                    else config.lr_schedule.min_learning_rate
                ),
            ),
        )
    if (
        args.max_grad_norm is not None
        or args.disable_grad_clipping
        or args.clip_epsilon is not None
    ):
        config = replace(
            config,
            gradients=DistillGradientConfig(
                max_grad_norm=(
                    None
                    if args.disable_grad_clipping
                    else (
                        args.max_grad_norm
                        if args.max_grad_norm is not None
                        else config.gradients.max_grad_norm
                    )
                ),
                clip_epsilon=(
                    args.clip_epsilon
                    if args.clip_epsilon is not None
                    else config.gradients.clip_epsilon
                ),
            ),
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
                run_name=(
                    args.run_name
                    if args.run_name is not None
                    else config.tracking.run_name
                ),
                tags=[*config.tracking.tags, *args.run_tag],
                notes=[*config.tracking.notes, *args.run_note],
                overwrite=args.run_overwrite or config.tracking.overwrite,
            ),
        )
    if (
        args.enable_logits_kl
        or args.enable_attention_mixer
        or args.logits_kl_weight is not None
        or args.attention_mixer_weight is not None
        or args.hidden_mse_weight is not None
        or args.bucket_shape_loss_weight is not None
        or args.bucket_shape_loss_type is not None
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
        attention_weight = (
            args.attention_mixer_weight
            if args.attention_mixer_weight is not None
            else config.losses.attention_or_mixer.weight
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
                attention_or_mixer=LossWeightConfig(
                    enabled=(
                        args.enable_attention_mixer
                        or config.losses.attention_or_mixer.enabled
                    ),
                    weight=attention_weight,
                ),
                bucket_shape_loss_weight=(
                    args.bucket_shape_loss_weight
                    if args.bucket_shape_loss_weight is not None
                    else config.losses.bucket_shape_loss_weight
                ),
                bucket_shape_loss_type=(
                    args.bucket_shape_loss_type
                    if args.bucket_shape_loss_type is not None
                    else config.losses.bucket_shape_loss_type
                ),
            ),
        )

    result = run_distill_stage(config)
    print(f"stage: {result.stage}")
    print(f"distill_mode: {result.distill_mode}")
    print(f"student_architecture: {result.student_architecture}")
    if result.student_backend is not None:
        print(f"student_backend: {result.student_backend}")
    print(f"emit_logits: {config.student.emit_logits}")
    print(f"emit_mixer_outputs: {config.student.emit_mixer_outputs}")
    print(f"tie_embeddings: {config.student.tie_embeddings}")
    print(f"optimizer: {config.optimizer.type}")
    print(f"base_learning_rate: {config.optimizer.learning_rate}")
    print(f"lr_schedule: {config.lr_schedule.type}")
    print(f"max_grad_norm: {config.gradients.max_grad_norm}")
    print(f"clip_epsilon: {config.gradients.clip_epsilon}")
    print(f"initial_learning_rate: {result.initial_learning_rate}")
    print(f"final_learning_rate: {result.final_learning_rate}")
    print(f"learning_rate: {result.final_learning_rate}")
    print(
        "logits_kl_enabled: "
        f"{config.losses.logits_kl.enabled and config.losses.logits_kl.weight > 0}"
    )
    attention_mixer_enabled = (
        config.losses.attention_or_mixer.enabled
        and config.losses.attention_or_mixer.weight > 0
    )
    print(f"attention_or_mixer_enabled: {attention_mixer_enabled}")
    print(f"targets: {result.target_bundle}")
    if result.fingerprint_artifact is not None:
        print(f"fingerprint_artifact: {result.fingerprint_artifact}")
    print(f"steps: {result.steps}")
    print(f"start_step: {result.start_step}")
    print(f"end_step: {result.end_step}")
    print(f"optimizer_steps_completed: {result.optimizer_steps_completed}")
    print(f"batches_consumed: {result.batches_consumed}")
    if result.resume_from is not None:
        print(f"resume_from: {result.resume_from}")
    if result.checkpoint_out is not None:
        print(f"checkpoint_out: {result.checkpoint_out}")
    if result.run_dir is not None:
        print(f"run_dir: {result.run_dir}")
    if result.metrics_path is not None:
        print(f"metrics_path: {result.metrics_path}")
    if result.report_path is not None:
        print(f"report_path: {result.report_path}")
    if result.summary_path is not None:
        print(f"summary_path: {result.summary_path}")
    print(f"initial_loss: {result.initial_loss:.8f}")
    print(f"final_loss: {result.final_loss:.8f}")
    if result.final_hidden_mse is not None:
        print(f"final_hidden_mse: {result.final_hidden_mse:.8f}")
    if result.final_logits_kl is not None:
        print(f"final_logits_kl: {result.final_logits_kl:.8f}")
    if result.final_attention_or_mixer is not None:
        print(f"final_attention_or_mixer: {result.final_attention_or_mixer:.8f}")
        print(f"final_attention_or_mixer_mse: {result.final_attention_or_mixer:.8f}")
    print(f"bucket_shape_loss_weight: {config.losses.bucket_shape_loss_weight}")
    print(f"bucket_shape_loss_type: {config.losses.bucket_shape_loss_type}")
    if result.final_grad_global_norm is not None:
        print(f"final_grad_global_norm: {result.final_grad_global_norm:.8f}")
    if result.final_grad_clipped_global_norm is not None:
        print(
            "final_grad_clipped_global_norm: "
            f"{result.final_grad_clipped_global_norm:.8f}"
        )
    if result.final_grad_clip_scale is not None:
        print(f"final_grad_clip_scale: {result.final_grad_clip_scale:.8f}")


if __name__ == "__main__":
    try:
        main()
    except (NotImplementedError, ValueError) as exc:
        print(f"Distillation stage failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
