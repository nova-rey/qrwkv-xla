from __future__ import annotations

import argparse
import json
from pathlib import Path

from qrwkv_xla.sharding.reports import P46_DEFAULT_ARTIFACT_DIR, write_p46_reports
from qrwkv_xla.sharding.smoke import run_pjit_sharding_smoke


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the P46 tiny pjit / sharding compile smoke."
    )
    parser.add_argument("--require-multi-device", action="store_true")
    parser.add_argument("--mesh-axis", default="data")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument(
        "--compile-api",
        choices=("auto", "pjit", "jit"),
        default="auto",
    )
    parser.add_argument(
        "--policy",
        choices=(
            "data_parallel_single_axis",
            "model_parallel_placeholder",
            "fsdp_placeholder",
        ),
        default="data_parallel_single_axis",
    )
    parser.add_argument("--skip-update", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", type=Path, default=P46_DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = run_pjit_sharding_smoke(
        require_multi_device=args.require_multi_device,
        mesh_axis=args.mesh_axis,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        compile_api=args.compile_api,
        policy_name=args.policy,
        skip_update=args.skip_update,
    )
    paths = write_p46_reports(result, out_dir=args.out, overwrite=args.overwrite)
    payload = result.to_dict()
    payload["artifacts"] = {key: str(path) for key, path in paths.items()}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print("P46 pjit / sharding compile smoke complete")
    print(f"status: {result.status}")
    print(f"compile_api: {result.compile_api}")
    print(f"multi_device_execution: {result.multi_device_execution}")
    if result.mesh.fallback_reason is not None:
        print(f"fallback_reason: {result.mesh.fallback_reason}")
    print(paths["markdown"])
    print(paths["json"])


if __name__ == "__main__":
    main()
