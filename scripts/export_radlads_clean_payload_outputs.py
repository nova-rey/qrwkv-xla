from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.parity.radlads_clean_loader import (
    DEFAULT_SEED,
    export_qrwkv_clean_payload_outputs,
    export_radlads_clean_payload_outputs,
)
from qrwkv_xla.parity.radlads_numerical_fixtures import DEFAULT_RADLADS_SOURCE

DEFAULT_PARAMETERS = Path(
    "artifacts/p53_radlads_qrwkv_head_to_head/radlads_parameters.npz"
)
DEFAULT_RADLADS_OUT = Path("artifacts/p54_radlads_loader_export_repair/radlads_outputs")
DEFAULT_QRWKV_OUT = Path("artifacts/p54_radlads_loader_export_repair/qrwkv_outputs")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export clean RADLADS and QRWKV payload outputs for the P53/P54 tiny cases."
        )
    )
    parser.add_argument("--parameters", type=Path, default=DEFAULT_PARAMETERS)
    parser.add_argument("--radlads-repo", type=Path, default=DEFAULT_RADLADS_SOURCE)
    parser.add_argument("--radlads-out", type=Path, default=DEFAULT_RADLADS_OUT)
    parser.add_argument("--qrwkv-out", type=Path, default=DEFAULT_QRWKV_OUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    radlads_result = export_radlads_clean_payload_outputs(
        args.parameters,
        args.radlads_out,
        radlads_source_path=args.radlads_repo,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(
        f"wrote RADLADS clean payload outputs to {args.radlads_out} "
        f"with overall_status={radlads_result.output_manifest['overall_status']} "
        f"load_status={radlads_result.load_result.overall_status}"
    )

    qrwkv_manifest = export_qrwkv_clean_payload_outputs(
        args.parameters,
        args.qrwkv_out,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(
        f"wrote QRWKV clean payload outputs to {args.qrwkv_out} "
        f"with overall_status={qrwkv_manifest['overall_status']}"
    )


if __name__ == "__main__":
    main()
