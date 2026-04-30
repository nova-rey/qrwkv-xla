from __future__ import annotations

import platform
import sys

from qrwkv_xla.xla import format_jax_runtime_info, get_jax_runtime_info


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(format_jax_runtime_info(get_jax_runtime_info()))


if __name__ == "__main__":
    main()
