from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a JAX student smoke train on an existing target bundle"
    )
    parser.add_argument("--targets", required=True, help="Existing target bundle path")
    parser.add_argument("--max-steps", type=int, default=2, help="Maximum train steps")
    parser.add_argument("--seed", type=int, default=0, help="JAX parameter seed")
    parser.add_argument(
        "--student-architecture",
        choices=(
            "tiny_student",
            "rwkv7_reference",
            "rwkv7_radlads_reference",
        ),
        default="tiny_student",
        help="Student architecture to instantiate for the smoke train",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=512,
        help="Student vocabulary size",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Smoke train learning rate"
    )
    args = parser.parse_args()

    from qrwkv_xla.students import create_student
    from qrwkv_xla.targets import read_manifest
    from qrwkv_xla.targets.store import manifest_path
    from qrwkv_xla.trainers import train_on_bundle_once

    if args.vocab_size <= 0:
        raise ValueError(f"vocab_size must be > 0, got {args.vocab_size}")

    targets = Path(args.targets)
    manifest = read_manifest(manifest_path(targets))
    student = create_student(
        args.student_architecture,
        vocab_size=args.vocab_size,
        hidden_size=manifest.hidden_size,
        num_layers=manifest.num_layers,
    )
    result = train_on_bundle_once(
        bundle_dir=targets,
        student=student,
        seed=args.seed,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
    )

    print(f"targets: {targets}")
    print(f"student_architecture: {args.student_architecture}")
    print(f"sequence_length: {manifest.sequence_length}")
    print(f"hidden_size: {manifest.hidden_size}")
    print(f"num_layers: {manifest.num_layers}")
    print(f"vocab_size: {args.vocab_size}")
    print(f"steps: {result.steps}")
    print(f"initial_loss: {result.initial_loss:.8f}")
    print(f"final_loss: {result.final_loss:.8f}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"Student smoke training failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
