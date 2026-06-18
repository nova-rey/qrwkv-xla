#!/usr/bin/env python3
from __future__ import annotations

import argparse

from qrwkv_xla.training import (
    RealStudentFingerprintForwardConfig,
    run_real_student_fingerprint_forward_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the P140 real student fingerprint forward smoke."
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--drop-remainder", action="store_true")
    parser.add_argument("--architecture-id", default="current_qrwkv")
    parser.add_argument("--student-max-seq-len", type=int, default=None)
    args = parser.parse_args()

    result = run_real_student_fingerprint_forward_smoke(
        RealStudentFingerprintForwardConfig(
            artifact_dir=args.artifact,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            seed=args.seed,
            shuffle=args.shuffle,
            max_records=args.max_records,
            drop_remainder=args.drop_remainder,
            architecture_id=args.architecture_id,
            student_max_seq_len=args.student_max_seq_len,
        )
    )
    print(
        f"status={result.status} "
        f"backend={result.student_backend_name} "
        f"logits_shape={result.logits_shape} "
        f"corridor_loss={result.metrics['fingerprint/corridor/loss_total']:.6f} "
        f"output_dir={result.output_dir}"
    )


if __name__ == "__main__":
    main()
