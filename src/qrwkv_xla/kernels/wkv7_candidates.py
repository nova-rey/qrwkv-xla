from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qrwkv_xla.kernels.wkv7_fixtures import wkv7_reference_full_scan

SUPPORTED_CANDIDATES = ("reference", "pallas")


@dataclass(frozen=True)
class UnsupportedCandidate(Exception):
    candidate: str
    reason: str


def run_wkv7_candidate(
    candidate: str, inputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    if candidate == "reference":
        return wkv7_reference_full_scan(inputs)
    if candidate == "pallas":
        raise UnsupportedCandidate(
            candidate="pallas",
            reason=(
                "P43 intentionally provides only a correctness fixture harness; "
                "the optimized Pallas WKV7 kernel is not implemented yet."
            ),
        )
    raise UnsupportedCandidate(candidate=candidate, reason="unknown candidate")
