from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


def _add_src_to_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Export teacher targets")
    parser.add_argument("--config", required=True, help="Path to teacher export YAML")
    parser.add_argument("--out", help="Output bundle directory override")
    parser.add_argument("--backend", help="Exporter backend override")
    parser.add_argument("--num-shards", type=int, help="Number of output shards")
    parser.add_argument("--batch-size", type=int, help="Examples per shard")
    parser.add_argument("--seed", type=int, help="Deterministic seed override")
    parser.add_argument(
        "--include-logits",
        action="store_true",
        help="Include logits in the exported shards",
    )
    args = parser.parse_args()

    _add_src_to_path()

    from qrwkv_xla.targets import inspect_target_bundle, validate_target_bundle
    from qrwkv_xla.teacher_export import (
        ExportRequest,
        get_teacher_exporter,
        load_teacher_export_config,
        validate_teacher_export_config,
    )

    config = load_teacher_export_config(args.config)
    targets = config.targets
    runtime = config.runtime

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

    config = replace(config, targets=targets, runtime=runtime)
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
    except (NotImplementedError, ValueError) as exc:
        print(f"Teacher export failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
