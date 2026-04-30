from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "qwen_policy.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve a local Qwen policy label")
    parser.add_argument("label", help="Qwen policy label, for example Qwen3.latest")
    parser.add_argument(
        "--policy",
        default=str(DEFAULT_POLICY),
        help="Path to local Qwen policy YAML",
    )
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Allow unresolved manual-only policy labels",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print resolution as JSON",
    )
    args = parser.parse_args()

    from qrwkv_xla.teacher_export import resolve_qwen_policy

    resolution = resolve_qwen_policy(
        args.label,
        policy_path=args.policy,
        allow_unresolved=args.allow_unresolved,
    )

    payload = {
        "label": resolution.label,
        "resolved_model_id": resolution.resolved_model_id,
        "tokenizer_id": resolution.tokenizer_id,
        "trust_remote_code": resolution.trust_remote_code,
        "dtype": resolution.dtype,
        "device": resolution.device,
        "is_resolved": resolution.is_resolved,
        "requires_manual_resolution": resolution.requires_manual_resolution,
        "notes": list(resolution.notes),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"label: {resolution.label}")
    print(f"resolved_model_id: {resolution.resolved_model_id or '<unresolved>'}")
    print(f"tokenizer_id: {resolution.tokenizer_id or '<unresolved>'}")
    print(f"trust_remote_code: {resolution.trust_remote_code}")
    print(f"dtype: {resolution.dtype}")
    print(f"device: {resolution.device}")
    print(f"is_resolved: {resolution.is_resolved}")
    print(f"requires_manual_resolution: {resolution.requires_manual_resolution}")
    print(f"notes: {list(resolution.notes)}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"Qwen policy resolution failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
