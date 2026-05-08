from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.scale_dry_run import (
    P45_DEFAULT_ARTIFACT_DIR,
    P45_HARDWARE_PROFILES,
    P45_TARGET_MODEL_PROFILES,
    P45_TRAINING_MODE,
    generate_multiscale_configs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate P45 multi-scale planning configs and fit reports. "
            "Artifacts are planning-only / dry-run-only."
        )
    )
    parser.add_argument(
        "--out",
        "--output-dir",
        type=Path,
        default=P45_DEFAULT_ARTIFACT_DIR,
        help="Directory for generated configs and reports.",
    )
    parser.add_argument(
        "--profiles",
        "--model-profile",
        action="append",
        choices=P45_TARGET_MODEL_PROFILES,
        help="Limit generation to one P45 model profile; repeatable.",
    )
    parser.add_argument(
        "--hardware",
        "--hardware-profile",
        action="append",
        choices=P45_HARDWARE_PROFILES,
        help="Limit fit matrix to one P45 hardware profile; repeatable.",
    )
    parser.add_argument(
        "--training-mode",
        default=P45_TRAINING_MODE,
        help="Planner training mode to use for generated reports.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Planning batch size for generated reports.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory before writing P45 artifacts.",
    )
    args = parser.parse_args()

    paths = generate_multiscale_configs(
        args.out,
        model_profiles=tuple(args.profiles or P45_TARGET_MODEL_PROFILES),
        hardware_profiles=tuple(args.hardware or P45_HARDWARE_PROFILES),
        training_mode_name=args.training_mode,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print("P45 multi-scale configs generated")
    print("status: planning-only / dry-run-only")
    for name, path in paths.items():
        print(f"{name}: {path}")
    print(f"results: {args.out / 'P45_RESULTS.md'}")


if __name__ == "__main__":
    main()
