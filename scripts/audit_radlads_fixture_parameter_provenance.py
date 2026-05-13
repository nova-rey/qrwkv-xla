from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from qrwkv_xla.parity import (
    audit_radlads_parameter_provenance,
    load_numerical_manifest,
    write_provenance_audit_report,
)

DEFAULT_MANIFEST = Path(
    "artifacts/p49_radlads_numerical_parity/radlads_fixtures/manifest.json"
)
DEFAULT_PARAMETERS = Path(
    "artifacts/p49_radlads_numerical_parity/radlads_fixtures/radlads_parameters.npz"
)
DEFAULT_OUT = Path("artifacts/p52_radlads_fixture_parameter_cleanup")
DEFAULT_RADLADS_SOURCE = Path("/home/nyx/.openclaw/workspace/_refs/RADLADS")
DEFAULT_SEED = 5050


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit RADLADS fixture parameter provenance across stages (P52)."
    )
    parser.add_argument(
        "--radlads-repo",
        type=Path,
        default=DEFAULT_RADLADS_SOURCE,
        help="Path to RADLADS repo for live introspection",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to manifest.json",
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
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed for deterministic hashing",
    )
    parser.add_argument(
        "--extreme-threshold",
        type=float,
        default=1e6,
        help="Threshold for flagging extreme values (default: 1e6)",
    )
    args = parser.parse_args()

    if args.out.exists() and args.overwrite:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    # Load manifest to get parameter payload path if not provided explicitly
    manifest = load_numerical_manifest(args.manifest)
    parameter_path = (
        args.parameters
        if args.parameters != DEFAULT_PARAMETERS
        else args.manifest.parent
        / str(manifest.get("parameter_payload", "radlads_parameters.npz"))
    )

    # Run audit
    result = audit_radlads_parameter_provenance(
        radlads_source_path=args.radlads_repo,
        manifest_path=args.manifest,
        parameters_path=parameter_path,
        seed=args.seed,
        extreme_threshold=args.extreme_threshold,
    )

    # Write reports
    write_provenance_audit_report(result, args.out)

    # Print summary
    summary = result.get("summary", {})
    blocking_issues = result.get("blocking_issues", [])
    recommendations = result.get("recommendations", [])

    print(
        f"wrote provenance audit to {args.out}\n"
        f"stages audited: {list(result.get('stages', {}).keys())}\n"
        f"total parameters: {summary.get('total_parameters', 'N/A')}\n"
        f"non-finite parameters: {summary.get('nonfinite_parameter_count', 0)}\n"
        f"extreme parameters: {summary.get('extreme_parameter_count', 0)}\n"
        f"blocking issues: {len(blocking_issues)}\n"
    )

    if blocking_issues:
        print("Blocking issues:")
        for issue in blocking_issues[:10]:  # Show first 10
            print(f"  - {issue}")

    if recommendations:
        print("\nRecommendations:")
        for rec in recommendations[:10]:  # Show first 10
            print(f"  - {rec}")


if __name__ == "__main__":
    main()
