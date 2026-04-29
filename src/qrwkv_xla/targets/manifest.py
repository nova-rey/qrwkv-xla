from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TargetFlags:
    input_ids: bool = True
    attention_mask: bool = True
    hidden_states: bool = True
    logits: bool = False
    attention_targets: bool = False


@dataclass(frozen=True)
class TeacherTargetManifest:
    schema_version: str
    teacher_family: str
    teacher_model_id: str | None
    teacher_policy_label: str
    fallback_policy_label: str | None
    tokenizer_id: str | None
    sequence_length: int
    hidden_size: int
    num_layers: int
    targets: TargetFlags
    dtype: str
    created_by: str = "teacher_exporter"
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
