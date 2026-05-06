from __future__ import annotations

import argparse
import json
from pathlib import Path

from qrwkv_xla.scale_planner import (
    HARDWARE_PROFILES,
    MODEL_PROFILES,
    TRAINING_MODES,
    ScalePlanRequest,
    distill_config_yaml,
    make_plan,
    plan_to_dict,
    plan_to_yaml,
    resolve_hardware_profile,
    resolve_model_profile,
    resolve_training_mode,
    summarize_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate QRWKV-XLA model scale memory fit"
    )
    parser.add_argument(
        "--model-profile", required=True, choices=tuple(sorted(MODEL_PROFILES))
    )
    parser.add_argument(
        "--hardware-profile", required=True, choices=tuple(sorted(HARDWARE_PROFILES))
    )
    parser.add_argument(
        "--training-mode", required=True, choices=tuple(sorted(TRAINING_MODES))
    )
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--target-sequence-length", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--minimum-sequence-length", type=int, default=128)
    parser.add_argument("--output")
    parser.add_argument("--format", choices=("yaml", "json"), default=None)
    parser.add_argument("--emit-distill-config")
    args = parser.parse_args()

    sequence_length = _sequence_length(
        args.sequence_length, args.target_sequence_length
    )
    request = ScalePlanRequest(
        model_profile=resolve_model_profile(args.model_profile),
        hardware_profile=resolve_hardware_profile(args.hardware_profile),
        training_mode=resolve_training_mode(args.training_mode),
        sequence_length=sequence_length,
        batch_size=args.batch_size,
        microbatch_size=args.microbatch_size,
        grad_accum_steps=args.grad_accum_steps,
        dtype=args.dtype,
        auto=args.auto,
        minimum_sequence_length=args.minimum_sequence_length,
    )
    plan = make_plan(request)
    output_format = _output_format(args.output, args.format)
    payload = (
        json.dumps(plan_to_dict(plan), indent=2) + "\n"
        if output_format == "json"
        else plan_to_yaml(plan)
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    if args.emit_distill_config:
        distill_path = Path(args.emit_distill_config)
        distill_path.parent.mkdir(parents=True, exist_ok=True)
        distill_path.write_text(distill_config_yaml(plan), encoding="utf-8")
    print(summarize_plan(plan))


def _sequence_length(
    sequence_length: int | None, target_sequence_length: int | None
) -> int:
    if sequence_length is not None and target_sequence_length is not None:
        raise SystemExit("--sequence-length conflicts with --target-sequence-length")
    value = sequence_length if sequence_length is not None else target_sequence_length
    if value is None:
        raise SystemExit(
            "one of --sequence-length or --target-sequence-length is required"
        )
    if value <= 0:
        raise SystemExit("sequence length must be > 0")
    return value


def _output_format(output: str | None, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    if output and output.endswith(".json"):
        return "json"
    if output and output.endswith((".yaml", ".yml")):
        return "yaml"
    return "yaml"


if __name__ == "__main__":
    main()
