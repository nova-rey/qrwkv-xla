from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QWEN_POLICY = ROOT / "configs" / "qwen_policy.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export teacher targets")
    parser.add_argument("--config", required=True, help="Path to teacher export YAML")
    parser.add_argument("--out", help="Output bundle directory override")
    parser.add_argument("--backend", help="Exporter backend override")
    parser.add_argument("--model-id", help="HF model id override")
    parser.add_argument("--tokenizer-id", help="HF tokenizer id override")
    parser.add_argument("--qwen-policy", help="Path to local Qwen policy YAML")
    parser.add_argument(
        "--resolve-qwen-policy",
        action="store_true",
        help="Resolve teacher.policy_label through the local Qwen policy file",
    )
    parser.add_argument(
        "--allow-unresolved-policy",
        action="store_true",
        help="Allow unresolved Qwen policy labels during dry-run inspection",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the intended export configuration without loading a model",
    )
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
        load_qwen_policy,
        load_teacher_export_config,
        resolve_qwen_policy_map,
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
    if args.qwen_policy is not None:
        runtime = replace(runtime, qwen_policy_path=Path(args.qwen_policy))
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

    resolution_status = "not_requested"
    if _should_resolve_qwen_policy(config, args.resolve_qwen_policy):
        policy_path = config.runtime.qwen_policy_path or DEFAULT_QWEN_POLICY
        policy = load_qwen_policy(policy_path)
        resolution = resolve_qwen_policy_map(
            policy,
            config.teacher.policy_label,
            allow_unresolved=(
                args.allow_unresolved_policy or bool(config.teacher.resolved_model_id)
            ),
        )
        resolution_status = "resolved" if resolution.is_resolved else "unresolved"
        teacher = replace(
            config.teacher,
            resolved_model_id=(
                config.teacher.resolved_model_id or resolution.resolved_model_id
            ),
            tokenizer_id=config.teacher.tokenizer_id or resolution.tokenizer_id,
            trust_remote_code=(
                config.teacher.trust_remote_code or resolution.trust_remote_code
            ),
            device=config.teacher.device or resolution.device,
            dtype=config.teacher.dtype or resolution.dtype,
        )
        config = replace(config, teacher=teacher)
        validate_teacher_export_config(config)
        if config.teacher.resolved_model_id:
            resolution_status = "resolved"

    if (
        config.runtime.exporter_backend == "hf"
        and _is_qwen_family(config.teacher.family)
        and not config.teacher.resolved_model_id
    ):
        if not (args.dry_run and args.allow_unresolved_policy):
            raise ValueError(
                f"Qwen policy {config.teacher.policy_label!r} is unresolved. Set "
                "resolved_model_id in config, update configs/qwen_policy.yaml, "
                "or pass --model-id. Use --dry-run "
                "--allow-unresolved-policy to inspect unresolved config."
            )
        resolution_status = "unresolved"

    if args.dry_run:
        tokenizer_id = config.teacher.tokenizer_id or config.teacher.resolved_model_id
        print("dry_run: true")
        print(f"backend: {config.runtime.exporter_backend}")
        print(f"family: {config.teacher.family}")
        print(f"policy_label: {config.teacher.policy_label}")
        print(f"model_id: {config.teacher.resolved_model_id or '<unresolved>'}")
        print(f"tokenizer_id: {tokenizer_id or '<unresolved>'}")
        print(f"trust_remote_code: {config.teacher.trust_remote_code}")
        print(f"dtype: {config.teacher.dtype}")
        print(f"device: {config.teacher.device}")
        print(f"output_dir: {config.runtime.output_dir}")
        print(f"resolution_status: {resolution_status}")
        return

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


def _should_resolve_qwen_policy(config, requested: bool) -> bool:
    return _is_qwen_family(config.teacher.family) and (
        requested or config.runtime.qwen_policy_path is not None
    )


def _is_qwen_family(family: str) -> bool:
    return family.strip().lower().startswith("qwen")


if __name__ == "__main__":
    try:
        main()
    except (NotImplementedError, RuntimeError, ValueError) as exc:
        print(f"Teacher export failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
