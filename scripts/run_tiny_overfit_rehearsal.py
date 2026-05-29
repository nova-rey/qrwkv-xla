from __future__ import annotations

import argparse
import json
from pathlib import Path

from qrwkv_xla.training import run_tiny_overfit_rehearsal


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the P96 tiny overfit rehearsal.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/p96_tiny_overfit/tiny_overfit_report.json"),
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    args = parser.parse_args()

    result = run_tiny_overfit_rehearsal(
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    report = result.to_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
