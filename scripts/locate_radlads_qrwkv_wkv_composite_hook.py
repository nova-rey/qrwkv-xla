#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_OUT = Path("artifacts/p64_composite_balance_hook/source_locator")
SEARCH_HINTS = (
    "state",
    "wkv",
    "balance",
    "matmul",
    "einsum",
    "outer",
    "decay",
    "update",
    "value",
    "key",
    "kv",
    "state + update",
    "state_after",
)
SOURCE_FILES = {
    "radlads": Path("src/qrwkv_xla/students/rwkv7_radlads_reference.py"),
    "qrwkv": Path("src/qrwkv_xla/students/rwkv7_qwen_reference.py"),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P64 locate RADLADS/QRWKV WKV composite balance-state hook sites."
    )
    parser.add_argument("--radlads-repo", type=Path)
    parser.add_argument("--qrwkv-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out} is not empty; pass --overwrite")
    args.out.mkdir(parents=True, exist_ok=True)

    qrwkv_root = args.qrwkv_root.resolve()
    radlads_root = args.radlads_repo.resolve() if args.radlads_repo else qrwkv_root
    radlads_file = _existing_file(radlads_root, SOURCE_FILES["radlads"])
    qrwkv_file = _existing_file(qrwkv_root, SOURCE_FILES["qrwkv"])
    report = {
        "schema": "radlads_qrwkv_wkv_composite_hook_locator.v1",
        "phase": "P64",
        "audit_root": str(qrwkv_root),
        "used_external_radlads_repo": args.radlads_repo is not None,
        "search_hints": list(SEARCH_HINTS),
        "sources": [
            _scan_side("radlads", radlads_file),
            _scan_side("qrwkv", qrwkv_file),
        ],
        "diagnostic_only": True,
        "recurrence_math_changed": False,
    }
    (args.out / "composite_hook_source_locator.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out / "COMPOSITE_HOOK_SOURCE_LOCATOR.md").write_text(
        _markdown(report),
        encoding="utf-8",
    )
    print(f"wrote P64 composite balance hook locator report to {args.out}")
    return 0


def _existing_file(root: Path, relative: Path) -> Path:
    candidate = root / relative
    if candidate.is_file():
        return candidate
    raise SystemExit(f"source file not found: {candidate}")


def _scan_side(side: str, path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8").splitlines()
    matches = []
    patterns = _side_patterns(side)
    for line_number, line in enumerate(lines, start=1):
        matched = [pattern for pattern in patterns if pattern in line]
        if matched:
            matches.append(
                {
                    "line": line_number,
                    "patterns": matched,
                    "text": line.strip(),
                }
            )
    source_expression = _pick_expression(matches, side)
    return {
        "side": side,
        "source_file": str(path),
        "function_class": _side_function(side),
        "source_expression": source_expression,
        "source_variable_name": "ab",
        "comparison_label": "composite_balance_update_term",
        "capture_method": "unavailable",
        "reason_if_unavailable": (
            "locator-only source audit; live capture handled by extraction"
        ),
        "matches": matches,
    }


def _side_patterns(side: str) -> tuple[str, ...]:
    if side == "radlads":
        return (
            "prev_state * decay",
            "prev_state @ ab",
            "next_state =",
            "ab =",
        )
    return (
        "prev_wkv * decay",
        "prev_wkv @ ab",
        "next_wkv =",
        "ab =",
    )


def _pick_expression(matches: list[dict[str, object]], side: str) -> str:
    for match in matches:
        text = str(match["text"])
        if "@ ab" in text or "next_state =" in text or "next_wkv =" in text:
            return text
    return "prev_state @ ab" if side == "radlads" else "prev_wkv @ ab"


def _side_function(side: str) -> str:
    return (
        "rwkv7_radlads_reference_layer/step"
        if side == "radlads"
        else "RWKV7QwenReference.step/apply_with_state"
    )


def _markdown(report: dict[str, object]) -> str:
    lines = [
        "# P64 Composite Balance Hook Locator",
        "",
        f"- audit_root: `{report['audit_root']}`",
        f"- used_external_radlads_repo: `{report['used_external_radlads_repo']}`",
        f"- recurrence_math_changed: `{report['recurrence_math_changed']}`",
        "",
        "## Sources",
        "",
    ]
    for source in report["sources"]:  # type: ignore[index]
        lines.extend(
            [
                f"### {source['side']}",
                f"- source file: `{source['source_file']}`",
                f"- function/class: `{source['function_class']}`",
                f"- source expression: `{source['source_expression']}`",
                f"- source variable name: `{source['source_variable_name']}`",
                f"- comparison label: `{source['comparison_label']}`",
                f"- capture method: `{source['capture_method']}`",
                f"- reason if unavailable: `{source['reason_if_unavailable']}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
