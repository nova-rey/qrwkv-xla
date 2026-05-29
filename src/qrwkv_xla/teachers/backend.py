from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np

from qrwkv_xla.targets import TargetStoreMetadata


class TeacherBackend(Protocol):
    """Teacher-side target emission boundary.

    P94 only proves deterministic synthetic emission. Real HF/Qwen backends are
    future phases.
    """

    name: str

    def build_metadata(
        self,
        *,
        num_examples: int,
        sequence_length: int,
    ) -> TargetStoreMetadata:
        raise NotImplementedError

    def emit_targets(
        self,
        *,
        num_examples: int,
        sequence_length: int,
    ) -> Mapping[str, np.ndarray]:
        raise NotImplementedError
