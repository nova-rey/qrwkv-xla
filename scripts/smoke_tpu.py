from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from qrwkv_xla.config import load_config

    config = load_config(root / "configs" / "tiny_tpu_smoke.yaml")
    print(f"Loaded config backend: {config.runtime.backend}")
    print(f"Configured sequence length: {config.model.sequence_length}")

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


if __name__ == "__main__":
    main()
