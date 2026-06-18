#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.artifacts import summarize_fingerprint_artifact


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize a behavioral_fingerprint artifact."
    )
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()

    summary = summarize_fingerprint_artifact(args.artifact)
    print(f"artifact_type={summary.artifact_type}")
    print(f"artifact_version={summary.artifact_version}")
    print(f"teacher_model_name={summary.teacher_model_name}")
    print(f"tokenizer_name={summary.tokenizer_name}")
    print(f"vocab_size={summary.vocab_size}")
    print(f"max_seq_len={summary.max_seq_len}")
    print(f"tracked_stats={','.join(summary.tracked_stats)}")
    print(f"num_modes={summary.num_modes}")
    print(f"num_corridor_records={summary.num_corridor_records}")
    print(f"has_exemplars={str(summary.has_exemplars).lower()}")
    print(f"exemplar_payload_type={summary.exemplar_payload_type}")
    print(f"num_exemplar_records={summary.num_exemplar_records}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
