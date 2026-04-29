from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ALLOWED_DTYPES = {"fp32", "fp16", "bf16"}
_ALLOWED_EXPORTER_BACKENDS = {"fake"}


@dataclass(frozen=True)
class TeacherModelConfig:
    family: str = "qwen"
    policy_label: str = "Qwen3.latest"
    fallback_label: str | None = "Qwen3.0"
    resolved_model_id: str | None = None
    tokenizer_id: str | None = None


@dataclass(frozen=True)
class ExportTargetConfig:
    sequence_length: int = 64
    hidden_size: int = 128
    num_layers: int = 2
    dtype: str = "fp32"
    include_logits: bool = False
    include_attention_targets: bool = False
    vocab_size: int = 512


@dataclass(frozen=True)
class ExportRuntimeConfig:
    exporter_backend: str = "fake"
    batch_size: int = 2
    num_shards: int = 2
    seed: int = 1234
    output_dir: Path = Path("artifacts/teacher_targets/fake_export")


@dataclass(frozen=True)
class TeacherExportConfig:
    teacher: TeacherModelConfig = field(default_factory=TeacherModelConfig)
    targets: ExportTargetConfig = field(default_factory=ExportTargetConfig)
    runtime: ExportRuntimeConfig = field(default_factory=ExportRuntimeConfig)


def validate_teacher_export_config(config: TeacherExportConfig) -> None:
    if not config.teacher.family.strip():
        raise ValueError("teacher.family must be non-empty")
    if not config.teacher.policy_label.strip():
        raise ValueError("teacher.policy_label must be non-empty")

    _require_positive("targets.sequence_length", config.targets.sequence_length)
    _require_positive("targets.hidden_size", config.targets.hidden_size)
    _require_positive("targets.num_layers", config.targets.num_layers)
    _require_positive("targets.vocab_size", config.targets.vocab_size)

    if config.targets.dtype not in _ALLOWED_DTYPES:
        allowed = ", ".join(sorted(_ALLOWED_DTYPES))
        raise ValueError(
            f"targets.dtype must be one of {{{allowed}}}, got {config.targets.dtype!r}"
        )

    if config.runtime.exporter_backend not in _ALLOWED_EXPORTER_BACKENDS:
        allowed = ", ".join(sorted(_ALLOWED_EXPORTER_BACKENDS))
        raise ValueError(
            "runtime.exporter_backend must be one of "
            f"{{{allowed}}}, got {config.runtime.exporter_backend!r}"
        )

    _require_positive("runtime.batch_size", config.runtime.batch_size)
    _require_positive("runtime.num_shards", config.runtime.num_shards)


def load_teacher_export_config(path: str | Path) -> TeacherExportConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Teacher export config root must be a mapping")

    teacher_data = _mapping_section(data, "teacher")
    targets_data = _mapping_section(data, "targets")
    runtime_data = _mapping_section(data, "runtime")

    config = TeacherExportConfig(
        teacher=TeacherModelConfig(
            family=str(teacher_data.get("family", "qwen")),
            policy_label=str(teacher_data.get("policy_label", "Qwen3.latest")),
            fallback_label=_optional_str(teacher_data.get("fallback_label", "Qwen3.0")),
            resolved_model_id=_optional_str(teacher_data.get("resolved_model_id")),
            tokenizer_id=_optional_str(teacher_data.get("tokenizer_id")),
        ),
        targets=ExportTargetConfig(
            sequence_length=int(targets_data.get("sequence_length", 64)),
            hidden_size=int(targets_data.get("hidden_size", 128)),
            num_layers=int(targets_data.get("num_layers", 2)),
            dtype=str(targets_data.get("dtype", "fp32")),
            include_logits=bool(targets_data.get("include_logits", False)),
            include_attention_targets=bool(
                targets_data.get("include_attention_targets", False)
            ),
            vocab_size=int(targets_data.get("vocab_size", 512)),
        ),
        runtime=ExportRuntimeConfig(
            exporter_backend=str(runtime_data.get("exporter_backend", "fake")),
            batch_size=int(runtime_data.get("batch_size", 2)),
            num_shards=int(runtime_data.get("num_shards", 2)),
            seed=int(runtime_data.get("seed", 1234)),
            output_dir=Path(
                str(
                    runtime_data.get(
                        "output_dir", "artifacts/teacher_targets/fake_export"
                    )
                )
            ),
        ),
    )
    validate_teacher_export_config(config)
    return config


def _mapping_section(root: dict[str, Any], key: str) -> dict[str, Any]:
    value = root.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} section must be a mapping")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
