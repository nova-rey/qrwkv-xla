from __future__ import annotations

import argparse
import sys

from qrwkv_xla.xla import (
    format_jax_runtime_info,
    get_jax_runtime_info,
    run_xla_distill_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a TPU-ready distillation smoke on the available JAX backend"
    )
    parser.add_argument("--targets", required=True)
    parser.add_argument(
        "--student-architecture",
        choices=("tiny_student", "rwkv7_reference"),
        default="rwkv7_reference",
    )
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--require-tpu", action="store_true")
    args = parser.parse_args()

    runtime = get_jax_runtime_info()
    print(format_jax_runtime_info(runtime))

    try:
        result = run_xla_distill_smoke(
            targets_dir=args.targets,
            student_architecture=args.student_architecture,
            max_steps=args.max_steps,
            seed=args.seed,
            learning_rate=args.learning_rate,
            require_tpu=args.require_tpu,
        )
    except RuntimeError as exc:
        print(f"TPU distill smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"backend: {result.backend}")
    print(f"device_count: {result.device_count}")
    print(f"has_tpu: {result.has_tpu}")
    print(f"student_architecture: {result.student_architecture}")
    print(f"steps: {result.steps}")
    print(f"initial_loss: {result.initial_loss:.8f}")
    print(f"final_loss: {result.final_loss:.8f}")


if __name__ == "__main__":
    main()
