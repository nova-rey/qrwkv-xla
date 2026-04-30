from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import qrwkv_xla
    from qrwkv_xla.config import load_config
    from qrwkv_xla.xla import format_jax_runtime_info, get_jax_runtime_info

    config = load_config(root / "configs" / "tiny_cpu.yaml")
    print("CPU smoke placeholder ran")
    print(f"Imported qrwkv_xla {qrwkv_xla.__version__}")
    print(f"Backend: {config.runtime.backend}")
    print(f"Student architecture: {config.model.student_architecture}")
    print(f"Sequence length: {config.model.sequence_length}")
    print(format_jax_runtime_info(get_jax_runtime_info()))


if __name__ == "__main__":
    main()
