from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qrwkv_xla.parity.radlads_clean_loader import load_radlads_clean_payload

DEFAULT_RADLADS_REPO = Path("/home/nyx/.openclaw/workspace/_refs/RADLADS")
DEFAULT_PARAMETERS = Path(
    "artifacts/p53_radlads_qrwkv_head_to_head/radlads_parameters.npz"
)
DEFAULT_OUT = Path("artifacts/p54_radlads_clean_payload_audit")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit RADLADS clean payload loading against the live tiny model."
    )
    parser.add_argument("--radlads-repo", type=Path, default=DEFAULT_RADLADS_REPO)
    parser.add_argument("--parameters", type=Path, default=DEFAULT_PARAMETERS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=5353)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = audit_clean_payload_loader(
        args.radlads_repo,
        args.parameters,
        seed=args.seed,
    )
    write_audit_reports(report, args.out, overwrite=args.overwrite)
    print(
        f"wrote P54 RADLADS loader audit to {args.out} "
        f"status={report['status']} "
        f"unsupported={report['summary']['unsupported']} "
        f"shape_mismatches={report['summary']['shape_mismatches']}"
    )


def audit_clean_payload_loader(
    radlads_repo: Path,
    parameter_payload: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    before = {
        "status": "blocked",
        "unsupported": 0,
        "shape_mismatches": 0,
        "missing_required": 0,
    }
    result = load_radlads_clean_payload(
        parameter_payload,
        radlads_source_path=radlads_repo,
        seed=seed,
        run_smoke=False,
    )
    after = {
        "status": result.overall_status,
        "unsupported": len(result.unsupported),
        "shape_mismatches": len(result.shape_mismatches),
        "missing_required": len(result.missing_required),
    }
    return {
        "schema": "radlads_clean_payload_loader_audit.v1",
        "parameter_payload": str(parameter_payload),
        "radlads_repo": str(radlads_repo),
        "seed": seed,
        "status": result.overall_status,
        "reason": result.reason,
        "summary": {
            "unsupported": len(result.unsupported),
            "shape_mismatches": len(result.shape_mismatches),
            "missing_required": len(result.missing_required),
            "defaulted": len(result.defaulted),
        },
        "blockers_before": before,
        "blockers_after": after,
        "mapping_entries": result.mapping_entries,
        "caveats": result.caveats,
    }


def write_audit_reports(
    report: dict[str, Any],
    out_dir: Path,
    *,
    overwrite: bool = False,
) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise SystemExit(f"{out_dir} is not empty; pass --overwrite to replace reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("*"):
        if path.is_file():
            path.unlink()
    (out_dir / "clean_payload_loader_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "P54_RADLADS_CLEAN_LOADER_AUDIT.md").write_text(
        _audit_markdown(report),
        encoding="utf-8",
    )


def _audit_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P54 RADLADS Clean Payload Loader Audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Unsupported: `{report['summary']['unsupported']}`",
        f"- Shape mismatches: `{report['summary']['shape_mismatches']}`",
        f"- Missing required: `{report['summary']['missing_required']}`",
        "",
        "## Blockers Before",
        "",
    ]
    for key, value in report["blockers_before"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Blockers After", ""])
    for key, value in report["blockers_after"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
