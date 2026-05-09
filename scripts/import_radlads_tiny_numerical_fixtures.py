from __future__ import annotations

import argparse
from pathlib import Path

from qrwkv_xla.parity import (
    import_numerical_fixture_directory,
    write_current_behavior_numerical_fixtures,
)

DEFAULT_OUT = Path("artifacts/p49_radlads_numerical_parity/radlads_fixtures")
DEFAULT_SEED = 4949


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Import canonical P49 tiny RADLADS numerical fixtures, or write "
            "offline-safe synthetic placeholders marked missing_source."
        )
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--source-fixtures",
        "--source-dir",
        dest="source_fixtures",
        type=Path,
        default=None,
        help="Existing P49 manifest directory/file to validate and copy.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.source_fixtures is not None:
        manifest = import_numerical_fixture_directory(
            args.source_fixtures,
            args.out,
            overwrite=args.overwrite,
        )
        print(
            "imported "
            f"{len(manifest['cases'])} P49 tiny numerical fixtures to {args.out}"
        )
        return

    manifest = write_current_behavior_numerical_fixtures(
        args.out,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(
        "wrote "
        f"{len(manifest['cases'])} missing_source synthetic P49 fixtures to "
        f"{args.out}"
    )


if __name__ == "__main__":
    main()
