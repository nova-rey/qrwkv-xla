from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from qrwkv_xla.parity import load_numerical_manifest
from qrwkv_xla.parity.radlads_parameter_import import (
    import_radlads_parameters_for_replay,
    load_radlads_parameter_npz,
)
from qrwkv_xla.parity.radlads_parameter_mapping import (
    normalize_radlads_parameter_arrays,
)
from qrwkv_xla.parity.radlads_replay import (
    diagnose_replay_case,
    replay_profile_for_case,
)
from qrwkv_xla.parity.radlads_replay_diagnostics import (
    build_diagnostic_report,
    summarize_parameter_payload,
    write_diagnostic_reports,
    write_parameter_sanity_reports,
)

DEFAULT_MANIFEST = Path(
    "artifacts/p49_radlads_numerical_parity/radlads_fixtures/manifest.json"
)
DEFAULT_OUT = Path("artifacts/p51_radlads_replay_diagnostics")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose P51 RADLADS replay non-finite failures."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--parameters", type=Path, default=None)
    parser.add_argument("--case", default="tiny_no_mask")
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--max-tensors", type=int, default=None)
    parser.add_argument("--include-finite-tensors", action="store_true")
    parser.add_argument("--stop-at-first-nonfinite", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and args.overwrite:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = load_numerical_manifest(args.manifest)
    parameter_path = args.parameters or args.manifest.parent / str(
        manifest.get("parameter_payload", "radlads_parameters.npz")
    )
    import_result = import_radlads_parameters_for_replay(
        parameter_path,
        manifest_path=args.manifest,
        allow_defaults=True,
    )
    normalized = normalize_radlads_parameter_arrays(
        load_radlads_parameter_npz(parameter_path)
    )

    cases = (
        manifest["cases"]
        if args.all_cases
        else [next(case for case in manifest["cases"] if case["name"] == args.case)]
    )

    tensor_summaries = []
    case_reports = []
    for case in cases:
        diagnostic = diagnose_replay_case(
            args.manifest,
            case,
            base_config=import_result.qrwkv_config,
            params=import_result.params,
        )
        summaries = diagnostic.pop("tensor_summaries")
        if not args.include_finite_tensors:
            summaries = [row for row in summaries if int(row["nonfinite_count"]) > 0]
        if args.stop_at_first_nonfinite and diagnostic["first_nonfinite"] is not None:
            summaries = [diagnostic["first_nonfinite"]]
        if args.max_tensors is not None:
            summaries = summaries[: args.max_tensors]
        tensor_summaries.extend(summaries)
        case_reports.append(diagnostic)

    active_defaults = set()
    for case in cases:
        active_defaults.update(replay_profile_for_case(case).active_defaulted_surfaces)
    parameter_sanity = summarize_parameter_payload(
        normalized,
        mapping_entries=import_result.report["mapping_entries"],
        active_defaulted_surfaces=active_defaults,
    )
    report = build_diagnostic_report(
        case_reports=case_reports,
        parameter_sanity=parameter_sanity,
    )
    write_parameter_sanity_reports(parameter_sanity, args.out)
    write_diagnostic_reports(
        report,
        tensor_summaries=tensor_summaries,
        out_dir=args.out,
    )

    if args.json:
        print(report)
    else:
        print(
            f"wrote P51 replay diagnostics to {args.out} "
            "cases="
            f"{len(case_reports)} "
            "nonfinite_params="
            f"{parameter_sanity['nonfinite_parameter_count']}"
        )


if __name__ == "__main__":
    main()
