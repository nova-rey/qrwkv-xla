from __future__ import annotations

import sys

from qrwkv_xla.smoke.colab_tpu import ColabTpuSmokeError, run_planner_tpu_smoke


def main() -> int:
    try:
        run_planner_tpu_smoke()
    except ColabTpuSmokeError as exc:
        print(f"P39 planner TPU smoke failed: {exc}", file=sys.stderr)
        return 1
    print("P39_PLANNER_TPU_SMOKE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
