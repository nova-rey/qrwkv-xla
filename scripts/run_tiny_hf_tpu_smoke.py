from __future__ import annotations

import sys

from qrwkv_xla.smoke.colab_tpu import ColabTpuSmokeError, run_tiny_hf_smoke_pair


def main() -> None:
    run_tiny_hf_smoke_pair()


if __name__ == "__main__":
    try:
        main()
    except ColabTpuSmokeError as exc:
        print(f"P38 tiny HF TPU smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
