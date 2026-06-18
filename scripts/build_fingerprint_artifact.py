#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from qrwkv_xla.fingerprint import (
    FingerprintCaptureConfig,
    FingerprintCorridorBoundsConfig,
    FingerprintExemplarReservoirCaptureConfig,
    FingerprintModeDiscoveryConfig,
    build_synthetic_capture_examples,
    capture_fingerprint_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a P143 synthetic behavioral_fingerprint artifact."
    )
    parser.add_argument("--synthetic-fixture", choices=("tiny",), default="tiny")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--max-exemplars", type=int, default=6)
    parser.add_argument("--max-modes", type=int, default=256)
    parser.add_argument(
        "--bounds-method", choices=("minmax", "quantile"), default="minmax"
    )
    parser.add_argument("--lower-quantile", type=float, default=0.05)
    parser.add_argument("--upper-quantile", type=float, default=0.95)
    parser.add_argument(
        "--exemplar-selection-policy",
        choices=("top_interestingness_v0", "stratified_interestingness_v0"),
        default="top_interestingness_v0",
    )
    parser.add_argument("--per-mode-min", type=int, default=0)
    parser.add_argument("--disable-exemplars", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    examples = build_synthetic_capture_examples(
        num_examples=args.num_examples,
        max_seq_len=args.max_seq_len,
        vocab_size=args.vocab_size,
    )
    config = FingerprintCaptureConfig(
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        mode_discovery=replace(
            FingerprintModeDiscoveryConfig(),
            max_modes=args.max_modes,
        ),
        corridor_bounds=FingerprintCorridorBoundsConfig(
            method=args.bounds_method,
            lower_quantile=args.lower_quantile,
            upper_quantile=args.upper_quantile,
        ),
        exemplar_reservoir=FingerprintExemplarReservoirCaptureConfig(
            enabled=not args.disable_exemplars,
            max_exemplars=args.max_exemplars,
            selection_policy=args.exemplar_selection_policy,
            per_mode_min=args.per_mode_min,
        ),
    )
    result = capture_fingerprint_artifact(config, examples)
    print(f"status={'pass' if result.validation_ok else 'fail'}")
    print(f"artifact_dir={result.output_dir}")
    print(f"modes_discovered={result.summary['modes_discovered']}")
    print(f"target_positions_processed={result.summary['target_positions_processed']}")
    print(f"exemplars_retained={result.summary['exemplars_retained']}")


if __name__ == "__main__":
    main()
