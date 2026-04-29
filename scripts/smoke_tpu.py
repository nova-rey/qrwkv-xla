from __future__ import annotations

import sys

try:
    import jax  # type: ignore
except ImportError:
    print("JAX is not installed; TPU smoke placeholder exiting gracefully.")
    sys.exit(0)

devices = list(jax.devices())
print("JAX devices:")
for device in devices:
    print(f"- {device}")

if not any(getattr(device, "platform", "") == "tpu" for device in devices):
    print("No TPU detected; Phase 0 smoke exits gracefully.")
    sys.exit(0)

print("TPU detected; Phase 0 smoke placeholder completed.")
