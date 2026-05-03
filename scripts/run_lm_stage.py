from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run QRWKV-XLA Stage 3 next-token CE fine-tuning"
    )
    parser.add_argument("--config", default="configs/lm_stage3_smoke.yaml")
    parser.add_argument("--prompt-corpus")
    parser.add_argument("--tokenized-corpus")
    parser.add_argument("--prompt-split")
    parser.add_argument("--prompt-tag", action="append", default=[])
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--student-architecture",
        choices=("tiny_student", "rwkv7_reference"),
    )
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--tie-embeddings", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--optimizer", choices=("sgd", "adam", "adamw"))
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--adam-beta1", type=float)
    parser.add_argument("--adam-beta2", type=float)
    parser.add_argument("--adam-epsilon", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--lr-schedule", choices=("constant", "warmup_cosine"))
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--disable-grad-clipping", action="store_true")
    parser.add_argument("--clip-epsilon", type=float)
    parser.add_argument("--checkpoint-out")
    parser.add_argument("--resume-from")
    parser.add_argument("--checkpoint-overwrite", action="store_true")
    parser.add_argument("--track-run", action="store_true")
    parser.add_argument("--run-root")
    parser.add_argument("--run-name")
    parser.add_argument("--run-tag", action="append", default=[])
    parser.add_argument("--run-note", action="append", default=[])
    parser.add_argument("--run-overwrite", action="store_true")
    args = parser.parse_args()

    from qrwkv_xla.distill import DistillGradientConfig
    from qrwkv_xla.generation import normalize_tokenizer_config
    from qrwkv_xla.lm import load_lm_stage_config, run_lm_stage

    if args.max_grad_norm is not None and args.disable_grad_clipping:
        parser.error("--max-grad-norm conflicts with --disable-grad-clipping")

    config = load_lm_stage_config(args.config, validate=False)
    if args.prompt_corpus is not None and args.tokenized_corpus is not None:
        parser.error("--prompt-corpus conflicts with --tokenized-corpus")

    if (
        args.prompt_corpus is not None
        or args.tokenized_corpus is not None
        or args.prompt_split is not None
        or args.prompt_tag
        or args.prompt_limit is not None
        or args.sequence_length is not None
        or args.batch_size is not None
        or args.shuffle
        or args.seed is not None
    ):
        config = replace(
            config,
            data=replace(
                config.data,
                prompt_corpus=Path(args.prompt_corpus)
                if args.prompt_corpus is not None
                else (
                    None
                    if args.tokenized_corpus is not None
                    else config.data.prompt_corpus
                ),
                tokenized_corpus=Path(args.tokenized_corpus)
                if args.tokenized_corpus is not None
                else (
                    None
                    if args.prompt_corpus is not None
                    else config.data.tokenized_corpus
                ),
                prompt_split=args.prompt_split
                if args.prompt_split is not None
                else config.data.prompt_split,
                prompt_tags=(*config.data.prompt_tags, *args.prompt_tag),
                prompt_limit=args.prompt_limit
                if args.prompt_limit is not None
                else config.data.prompt_limit,
                sequence_length=args.sequence_length
                if args.sequence_length is not None
                else config.data.sequence_length,
                batch_size=args.batch_size
                if args.batch_size is not None
                else config.data.batch_size,
                shuffle=args.shuffle or config.data.shuffle,
                seed=args.seed if args.seed is not None else config.data.seed,
            ),
        )
    if (
        args.student_architecture is not None
        or args.vocab_size is not None
        or args.hidden_size is not None
        or args.num_layers is not None
        or args.tie_embeddings
    ):
        config = replace(
            config,
            student=replace(
                config.student,
                architecture=args.student_architecture
                if args.student_architecture is not None
                else config.student.architecture,
                vocab_size=args.vocab_size
                if args.vocab_size is not None
                else config.student.vocab_size,
                hidden_size=args.hidden_size
                if args.hidden_size is not None
                else config.student.hidden_size,
                num_layers=args.num_layers
                if args.num_layers is not None
                else config.student.num_layers,
                emit_logits=True,
                tie_embeddings=args.tie_embeddings or config.student.tie_embeddings,
            ),
        )
    if args.max_steps is not None or args.seed is not None:
        config = replace(
            config,
            training=replace(
                config.training,
                max_steps=args.max_steps
                if args.max_steps is not None
                else config.training.max_steps,
                seed=args.seed if args.seed is not None else config.training.seed,
            ),
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
                type=args.optimizer
                if args.optimizer is not None
                else config.optimizer.type,
                learning_rate=args.learning_rate
                if args.learning_rate is not None
                else config.optimizer.learning_rate,
                beta1=args.adam_beta1
                if args.adam_beta1 is not None
                else config.optimizer.beta1,
                beta2=args.adam_beta2
                if args.adam_beta2 is not None
                else config.optimizer.beta2,
                epsilon=args.adam_epsilon
                if args.adam_epsilon is not None
                else config.optimizer.epsilon,
                weight_decay=args.weight_decay
                if args.weight_decay is not None
                else config.optimizer.weight_decay,
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
                type=args.lr_schedule
                if args.lr_schedule is not None
                else config.lr_schedule.type,
                warmup_steps=args.warmup_steps
                if args.warmup_steps is not None
                else config.lr_schedule.warmup_steps,
                total_steps=args.total_steps
                if args.total_steps is not None
                else config.lr_schedule.total_steps,
                min_learning_rate=args.min_learning_rate
                if args.min_learning_rate is not None
                else config.lr_schedule.min_learning_rate,
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
                clip_epsilon=args.clip_epsilon
                if args.clip_epsilon is not None
                else config.gradients.clip_epsilon,
            ),
        )
    if (
        args.checkpoint_out is not None
        or args.resume_from is not None
        or args.checkpoint_overwrite
    ):
        config = replace(
            config,
            checkpoint=replace(
                config.checkpoint,
                checkpoint_out=Path(args.checkpoint_out)
                if args.checkpoint_out is not None
                else config.checkpoint.checkpoint_out,
                resume_from=Path(args.resume_from)
                if args.resume_from is not None
                else config.checkpoint.resume_from,
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
                run_root=Path(args.run_root)
                if args.run_root is not None
                else config.tracking.run_root,
                run_name=args.run_name
                if args.run_name is not None
                else config.tracking.run_name,
                tags=[*config.tracking.tags, *args.run_tag],
                notes=[*config.tracking.notes, *args.run_note],
                overwrite=args.run_overwrite or config.tracking.overwrite,
            ),
        )

    result = run_lm_stage(config)
    print(f"stage: {result.stage}")
    print("mode: lm_stage3_ce")
    print(f"student_architecture: {result.student_architecture}")
    print(f"emit_logits: {config.student.emit_logits}")
    print(f"tie_embeddings: {config.student.tie_embeddings}")
    print(f"optimizer: {config.optimizer.type}")
    print(f"base_learning_rate: {config.optimizer.learning_rate}")
    print(f"lr_schedule: {config.lr_schedule.type}")
    print(f"max_grad_norm: {config.gradients.max_grad_norm}")
    print(f"clip_epsilon: {config.gradients.clip_epsilon}")
    if result.prompt_corpus is not None:
        print(f"prompt_corpus: {result.prompt_corpus}")
    if result.tokenized_corpus is not None:
        print(f"tokenized_corpus: {result.tokenized_corpus}")
    tokenizer_config = normalize_tokenizer_config(config.data.tokenizer)
    print(f"tokenizer_backend: {tokenizer_config.backend}")
    if tokenizer_config.tokenizer_id is not None:
        print(f"tokenizer_id: {tokenizer_config.tokenizer_id}")
    print(f"sequence_length: {config.data.sequence_length}")
    print(f"batch_size: {config.data.batch_size}")
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
    print(f"final_ce_loss: {result.final_ce_loss:.8f}")
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
    except (NotImplementedError, RuntimeError, ValueError) as exc:
        print(f"LM stage failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
