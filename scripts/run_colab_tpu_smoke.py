from __future__ import annotations

import sys

from qrwkv_xla.smoke.colab_tpu import ColabTpuSmokeError, run_distill_smoke_pair


def main() -> None:
    run_distill_smoke_pair()


if __name__ == "__main__":
    try:
        main()
    except ColabTpuSmokeError as exc:
        print(f"P36 Colab TPU smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
