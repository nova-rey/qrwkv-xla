from __future__ import annotations

import argparse

from qrwkv_xla.validation.pipeline import (
    build_validation_commands,
    format_command,
    format_pipeline_validation_result,
    run_pipeline_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the canonical QRWKV-XLA end-to-end validation pipeline"
    )
    parser.add_argument(
        "--include-hf",
        action="store_true",
        help="Include the optional tiny Hugging Face teacher export smoke.",
    )
    parser.add_argument(
        "--require-tpu",
        action="store_true",
        help="Require TPU availability for the TPU distillation smoke.",
    )
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Run all validation steps even after a failure.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-step output while keeping summaries and failure details.",
    )
    args = parser.parse_args()

    if args.include_hf:
        print(
            "Warning: --include-hf enables optional teacher-hf validation and "
            "requires torch/transformers plus local model availability.",
            flush=True,
        )
    if args.require_tpu:
        print(
            "Warning: --require-tpu makes TPU distillation smoke a hard TPU check.",
            flush=True,
        )

    total_steps = len(
        build_validation_commands(
            include_hf=args.include_hf,
            require_tpu=args.require_tpu,
            max_steps=args.max_steps,
        )
    )

    result = run_pipeline_validation(
        include_hf=args.include_hf,
        require_tpu=args.require_tpu,
        max_steps=args.max_steps,
        stop_on_failure=not args.continue_on_failure,
    )

    if not args.quiet:
        for index, step in enumerate(result.steps, start=1):
            status = "passed" if step.passed else "failed"
            print(f"[{index}/{total_steps}] {step.name}: {status}")
            print(f"  command: {format_command(step.command)}")

    print(format_pipeline_validation_result(result))
    if not result.passed:
        for step in result.failed_steps:
            print(f"\nFailed step: {step.name}")
            print(f"Command: {format_command(step.command)}")
            if step.stdout:
                print("stdout:")
                print(step.stdout.rstrip())
            if step.stderr:
                print("stderr:")
                print(step.stderr.rstrip())
        raise SystemExit(1)

    raise SystemExit(0)


if __name__ == "__main__":
    main()
