from __future__ import annotations

import platform
import sys

print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")

try:
    import jax  # type: ignore
except ImportError:
    print("JAX: not installed")
else:
    print(f"JAX: {jax.__version__}")
    try:
        devices = [str(device) for device in jax.devices()]
    except Exception as exc:  # pragma: no cover - defensive placeholder
        print(f"JAX devices: unavailable ({exc})")
    else:
        print("JAX devices:")
        for device in devices:
            print(f"- {device}")
