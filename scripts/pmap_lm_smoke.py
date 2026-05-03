from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from qrwkv_xla.distributed.devices import format_device_topology
from qrwkv_xla.distributed.lm_pmap import (
    PmapLMSkip,
    prepare_pmap_lm_smoke,
    run_pmap_lm_smoke,
)
from qrwkv_xla.lm import load_lm_stage_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a skip-safe multi-device pmap LM smoke"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device-count", type=int)
    parser.add_argument("--require-multiple-devices", action="store_true")
    parser.add_argument("--min-device-count", type=int, default=2)
    parser.add_argument("--checkpoint-out")
    parser.add_argument("--checkpoint-overwrite", action="store_true")
    args = parser.parse_args()

    config = load_lm_stage_config(args.config)
    if args.max_steps is not None:
        config = replace(
            config,
            training=replace(config.training, max_steps=args.max_steps),
        )
    if (
        args.min_device_count != config.distributed.min_device_count
        or args.require_multiple_devices
    ):
        config = replace(
            config,
            distributed=replace(
                config.distributed,
                min_device_count=args.min_device_count,
                require_multiple_devices=(
                    args.require_multiple_devices
                    or config.distributed.require_multiple_devices
                ),
            ),
        )
    if args.checkpoint_out is not None or args.checkpoint_overwrite:
        config = replace(
            config,
            checkpoint=replace(
                config.checkpoint,
                checkpoint_out=args.checkpoint_out or config.checkpoint.checkpoint_out,
                overwrite=args.checkpoint_overwrite or config.checkpoint.overwrite,
            ),
        )

    prepared = prepare_pmap_lm_smoke(config, device_count_cap=args.device_count)
    topology = prepared.topology
    print(format_device_topology(topology))
    if isinstance(prepared, PmapLMSkip):
        message = f"SKIPPED: {prepared.reason}"
        if args.require_multiple_devices or config.distributed.require_multiple_devices:
            print(message, file=sys.stderr)
            raise SystemExit(1)
        print(message)
        raise SystemExit(0)

    result = run_pmap_lm_smoke(config, device_count_cap=args.device_count)
    print(f"active_device_count: {result.device_count}")
    print(f"device_count: {result.device_count}")
    print(f"per_device_batch_size: {result.per_device_batch_size}")
    print(f"steps: {result.steps}")
    print(f"initial_loss: {result.initial_loss:.8f}")
    print(f"final_loss: {result.final_loss:.8f}")
    print(f"checkpoint_out: {result.checkpoint_out}")


if __name__ == "__main__":
    main()
