from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export teacher targets")
    parser.add_argument("--config", required=True, help="Path to teacher export YAML")
    parser.add_argument("--out", help="Output bundle directory override")
    parser.add_argument("--backend", help="Exporter backend override")
    parser.add_argument("--model-id", help="HF model id override")
    parser.add_argument("--tokenizer-id", help="HF tokenizer id override")
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt text for HF export; may be repeated",
    )
    parser.add_argument("--prompt-file", help="File with one HF export prompt per line")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow Hugging Face remote model code for HF export",
    )
    parser.add_argument("--revision", help="HF model/tokenizer revision")
    parser.add_argument("--device", help="Teacher device override: cpu or auto")
    parser.add_argument(
        "--dtype",
        help="Teacher dtype override: auto, fp32, fp16, or bf16",
    )
    parser.add_argument("--num-shards", type=int, help="Number of output shards")
    parser.add_argument("--batch-size", type=int, help="Examples per shard")
    parser.add_argument("--seed", type=int, help="Deterministic seed override")
    parser.add_argument(
        "--include-logits",
        action="store_true",
        help="Include logits in the exported shards",
    )
    args = parser.parse_args()

    from qrwkv_xla.targets import inspect_target_bundle, validate_target_bundle
    from qrwkv_xla.teacher_export import (
        ExportRequest,
        get_teacher_exporter,
        load_teacher_export_config,
        validate_teacher_export_config,
    )

    config = load_teacher_export_config(args.config)
    teacher = config.teacher
    targets = config.targets
    runtime = config.runtime

    if args.model_id is not None:
        teacher = replace(teacher, resolved_model_id=args.model_id)
    if args.tokenizer_id is not None:
        teacher = replace(teacher, tokenizer_id=args.tokenizer_id)
    if args.trust_remote_code:
        teacher = replace(teacher, trust_remote_code=True)
    if args.revision is not None:
        teacher = replace(teacher, revision=args.revision)
    if args.device is not None:
        teacher = replace(teacher, device=args.device)
    if args.dtype is not None:
        teacher = replace(teacher, dtype=args.dtype)
    if args.backend is not None:
        runtime = replace(runtime, exporter_backend=args.backend)
    if args.num_shards is not None:
        runtime = replace(runtime, num_shards=args.num_shards)
    if args.batch_size is not None:
        runtime = replace(runtime, batch_size=args.batch_size)
    if args.seed is not None:
        runtime = replace(runtime, seed=args.seed)
    if args.out is not None:
        runtime = replace(runtime, output_dir=Path(args.out))
    if args.include_logits:
        targets = replace(targets, include_logits=True)
    if args.prompt is not None:
        targets = replace(targets, prompt_texts=tuple(args.prompt))
    if args.prompt_file is not None:
        targets = replace(targets, prompt_file=Path(args.prompt_file))

    config = replace(config, teacher=teacher, targets=targets, runtime=runtime)
    validate_teacher_export_config(config)

    exporter = get_teacher_exporter(config.runtime.exporter_backend)
    result = exporter.export(
        ExportRequest(config=config, output_dir=config.runtime.output_dir)
    )
    validate_target_bundle(result.output_dir)
    summary = inspect_target_bundle(result.output_dir)

    print(f"output_dir: {result.output_dir}")
    print(f"teacher_policy_label: {result.manifest.teacher_policy_label}")
    print(f"backend: {exporter.name}")
    print(f"shard_count: {result.shard_count}")
    print(f"total_examples: {result.total_examples}")
    print(f"target_keys: {', '.join(summary['target_keys'])}")


if __name__ == "__main__":
    try:
        main()
    except (NotImplementedError, RuntimeError, ValueError) as exc:
        print(f"Teacher export failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
