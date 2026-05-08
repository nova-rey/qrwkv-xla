from __future__ import annotations

import argparse
import json
import sys

from qrwkv_xla.export import (
    export_checkpoint_to_hf_safetensors,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a QRWKV-XLA student checkpoint to HF-style safetensors"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = export_checkpoint_to_hf_safetensors(
        args.checkpoint,
        args.output_dir,
        overwrite=args.overwrite,
    )
    summary = {
        "export_dir": str(result.export_dir),
        "files": {
            "config": str(result.config_path),
            "model": str(result.model_path),
            "metadata": str(result.metadata_path),
            "weight_map": str(result.weight_map_path),
        },
        "tensor_count": result.metadata["tensor_count"],
        "student_architecture": result.metadata["student_architecture"],
        "checkpoint_step": result.metadata["checkpoint_step"],
    }
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(f"export_dir: {result.export_dir}")
        print(f"tensor_count: {result.metadata['tensor_count']}")


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        print(f"HF/safetensors export failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except (FileExistsError, FileNotFoundError, ValueError, TypeError) as exc:
        print(f"HF/safetensors export failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
