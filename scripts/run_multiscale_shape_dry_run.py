from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.scale_dry_run import (
    P45_DEFAULT_ARTIFACT_DIR,
    P45_HARDWARE_PROFILES,
    P45_TARGET_MODEL_PROFILES,
    run_shape_dry_run,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run safe P45 multi-scale shape dry-runs. "
            "Defaults to metadata-only and does not allocate full model arrays."
        )
    )
    parser.add_argument(
        "--out",
        "--output-dir",
        type=Path,
        default=P45_DEFAULT_ARTIFACT_DIR,
        help="Directory for dry-run reports and checkpoint skeletons.",
    )
    parser.add_argument(
        "--profiles",
        "--model-profile",
        action="append",
        choices=P45_TARGET_MODEL_PROFILES + ("tiny_debug",),
        help="Model profile to dry-run; repeatable. Defaults to all P45 targets.",
    )
    parser.add_argument(
        "--hardware",
        action="append",
        choices=P45_HARDWARE_PROFILES,
        help="Hardware profile to include when no scale-plan is supplied; repeatable.",
    )
    parser.add_argument(
        "--scale-plan",
        type=Path,
        help=(
            "Read model/hardware selections and fit metadata from "
            "scale_plan_report.json."
        ),
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        default=True,
        help="Run metadata-only validation. This is the default safe behavior.",
    )
    parser.add_argument(
        "--materialize-init",
        action="store_true",
        help=(
            "Initialize real params when safe. Larger profiles are blocked "
            "unless explicitly allowed."
        ),
    )
    parser.add_argument(
        "--allow-large-materialization",
        action="store_true",
        help=(
            "Permit materialization for non-debug profiles. This is "
            "intentionally off by default."
        ),
    )
    parser.add_argument(
        "--no-checkpoint-skeleton",
        action="store_true",
        help="Skip metadata checkpoint skeleton emission.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory before writing P45 dry-run artifacts.",
    )
    args = parser.parse_args()

    payload = run_shape_dry_run(
        args.out,
        model_profiles=tuple(args.profiles or P45_TARGET_MODEL_PROFILES),
        hardware_profiles=tuple(args.hardware or P45_HARDWARE_PROFILES),
        metadata_only=args.metadata_only,
        materialize_init=args.materialize_init,
        allow_large_materialization=args.allow_large_materialization,
        checkpoint_skeleton=not args.no_checkpoint_skeleton,
        scale_plan_path=args.scale_plan,
        overwrite=args.overwrite,
    )
    print("P45 multi-scale shape dry-run complete")
    print("status: dry-run-only; metadata-only by default")
    print(f"profiles: {len(payload['dry_runs'])}")
    print(args.out / "multiscale_shape_dry_run_report.json")
    print(args.out / "P45_RESULTS.md")


if __name__ == "__main__":
    main()
