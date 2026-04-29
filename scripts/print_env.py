from __future__ import annotations

import platform
import sys


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    try:
        import jax  # type: ignore
    except ImportError:
        print("JAX: not installed")
        return

    print(f"JAX: {jax.__version__}")
    for device in jax.devices():
        print(f"- {device}")


if __name__ == "__main__":
    main()
