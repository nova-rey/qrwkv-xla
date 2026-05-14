#!/usr/bin/env python3
# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from qrwkv_xla.parity.radlads_wkv_state_provenance import (
    WKV_STATE_PROVENANCE_SCHEMA,
    trace_qrwkv_state_provenance,
    write_provenance_jsonl,
    write_provenance_reports,
)

DEFAULT_OUT = Path("artifacts/p59_wkv_state_provenance")


def _synthetic_inputs(
    *,
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    input_ids = (
        np.arange(batch_size * sequence_length, dtype=np.int32).reshape(
            batch_size, sequence_length
        )
        % vocab_size
    )
    attention_mask = np.ones((batch_size, sequence_length), dtype=np.int32)
    if sequence_length >= 3:
        attention_mask[:, 1] = 0
    return input_ids, attention_mask


def _write_manifest(
    out_dir: Path,
    *,
    records_path: Path,
    report_path: Path,
    args: argparse.Namespace,
) -> None:
    payload = {
        "schema": WKV_STATE_PROVENANCE_SCHEMA,
        "records": str(records_path),
        "report": str(report_path),
        "diagnostic_only": True,
        "synthetic_case": args.case,
        "config": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "vocab_size": args.vocab_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "seed": args.seed,
        },
        "tolerances": {"atol": args.atol, "rtol": args.rtol},
    }
    (out_dir / "wkv_state_provenance_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Trace P59 RADLADS/QRWKV WKV state handoff provenance using the "
            "existing QRWKV student state APIs."
        )
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--case", default="synthetic_masked_handoff")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=59)
    parser.add_argument("--max-inline-values", type=int, default=4096)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out} is not empty; pass --overwrite")
    args.out.mkdir(parents=True, exist_ok=True)

    import jax
    from qrwkv_xla.students.rwkv7_qwen_reference import (
        RWKV7QwenReferenceConfig,
        RWKV7QwenReferenceStudent,
    )

    config = RWKV7QwenReferenceConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=1,
        use_rope=False,
        emit_logits=True,
        attention_qkv_bias=True,
    )
    student = RWKV7QwenReferenceStudent(config)
    params = student.init_params(jax.random.PRNGKey(args.seed))
    input_ids, attention_mask = _synthetic_inputs(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        vocab_size=args.vocab_size,
    )
    records = trace_qrwkv_state_provenance(
        student,
        params,
        input_ids,
        attention_mask=attention_mask,
        case=args.case,
        max_inline_values=args.max_inline_values,
        atol=args.atol,
        rtol=args.rtol,
    )
    records_path = args.out / "wkv_state_provenance.jsonl"
    write_provenance_jsonl(records, records_path)
    write_provenance_reports(records, args.out)
    _write_manifest(
        args.out,
        records_path=records_path,
        report_path=args.out / "wkv_state_provenance_report.json",
        args=args,
    )
    print(f"wrote P59 WKV state provenance to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
