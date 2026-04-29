from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import qrwkv_xla

    print("CPU smoke placeholder ran")
    print(f"Imported qrwkv_xla {qrwkv_xla.__version__}")


if __name__ == "__main__":
    main()
