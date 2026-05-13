from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from qrwkv_xla.parity import write_audit_report
from qrwkv_xla.parity.radlads_fixture_validation import (
    audit_parameter_payload,
    to_audit_report,
)
from qrwkv_xla.parity.radlads_parameter_import import load_radlads_parameter_npz

DEFAULT_PARAMETERS = Path(
    "artifacts/p49_radlads_numerical_parity/radlads_fixtures/radlads_parameters.npz"
)
DEFAULT_OUT = Path("artifacts/p52_radlads_fixture_parameter_cleanup/validation")
DEFAULT_EXTREME_THRESHOLD = 1e6


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate RADLADS parameter payload for finite/extreme values."
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=DEFAULT_PARAMETERS,
        help="Path to radlads_parameters.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output directory"
    )
    parser.add_argument(
        "--extreme-threshold",
        type=float,
        default=DEFAULT_EXTREME_THRESHOLD,
        help="Threshold for flagging extreme values (default: 1e6)",
    )
    args = parser.parse_args()

    if args.out.exists() and args.overwrite:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    # Load and validate parameters
    parameters = load_radlads_parameter_npz(args.parameters)
    results = audit_parameter_payload(
        parameters,
        stage="saved_npz",
        extreme_threshold=args.extreme_threshold,
    )
    report = to_audit_report(results)

    # Write reports
    write_audit_report(report, args.out)

    # Print summary
    summary = report["summary"]

    status_total = (
        summary["finite_ok"] + summary["extreme_value"] + summary["non_finite"]
    )
    print(
        f"wrote parameter validation to {args.out}\n"
        f"total parameters: {report['parameter_count']}\n"
        f"finite_ok: {summary['finite_ok']}\n"
        f"non_finite: {summary['non_finite']}\n"
        f"extreme: {summary['extreme_value']}\n"
        f"status: finite={status_total}\n"
    )

    if summary["non_finite"] > 0 or summary["extreme_value"] > 0:
        print("Offending parameters:")
        for param in report.get("parameters", {}).get("non_finite", []):
            print(
                "  - "
                f"{param['name']}: status=non_finite, "
                f"abs_max={param['abs_max']:.6e}"
            )
        for param in report.get("parameters", {}).get("extreme_value", []):
            print(
                "  - "
                f"{param['name']}: status=extreme_value, "
                f"abs_max={param['abs_max']:.6e}"
            )


if __name__ == "__main__":
    main()
