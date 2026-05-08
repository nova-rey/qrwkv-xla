from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.parity import write_parameter_surface_map_reports

DEFAULT_OUT = Path("artifacts/parity/radlads_source_bridge")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the P40 RADLADS-to-QRWKV parameter surface map."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    report = write_parameter_surface_map_reports(args.out_dir)
    print(
        f"wrote parameter surface map to {args.out_dir} "
        f"with {len(report['mappings'])} rows"
    )


if __name__ == "__main__":
    main()
