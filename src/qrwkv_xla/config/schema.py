from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str = "cpu"
    require_accelerator: bool = False


@dataclass(frozen=True)
class ModelConfig:
    student_architecture: str = "rwkv7_style"
    vocab_size: int = 512
    hidden_size: int = 128
    num_layers: int = 2
    sequence_length: int = 64


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 2
    max_steps: int = 10


@dataclass(frozen=True)
class QRWKVConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    raw: dict[str, Any] = field(default_factory=dict)
