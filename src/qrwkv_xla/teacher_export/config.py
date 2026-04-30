from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ALLOWED_EXPORTER_BACKENDS = {"fake", "hf"}
_ALLOWED_EXPORT_DTYPES = {"fp32", "fp16", "bf16"}
_ALLOWED_TEACHER_DTYPES = {"auto", "fp32", "fp16", "bf16"}
_ALLOWED_TEACHER_DEVICES = {"cpu", "auto"}
_MISSING = object()


@dataclass(frozen=True)
class TeacherModelConfig:
    family: str = "qwen"
    policy_label: str = "Qwen3.latest"
    fallback_label: str | None = "Qwen3.0"
    resolved_model_id: str | None = None
    tokenizer_id: str | None = None
    trust_remote_code: bool = False
    revision: str | None = None
    device: str = "cpu"
    dtype: str = "auto"


@dataclass(frozen=True)
class ExportTargetConfig:
    sequence_length: int = 64
    hidden_size: int | None = 128
    num_layers: int | None = 2
    dtype: str = "fp32"
    include_logits: bool = False
    include_attention_targets: bool = False
    vocab_size: int = 512
    prompt_texts: tuple[str, ...] = field(default_factory=tuple)
    prompt_file: Path | None = None
    max_new_tokens: int = 0


@dataclass(frozen=True)
class ExportRuntimeConfig:
    exporter_backend: str = "fake"
    batch_size: int = 2
    num_shards: int = 2
    seed: int = 1234
    output_dir: Path = Path("artifacts/teacher_targets/fake_export")
    require_resolved_model: bool = False
    qwen_policy_path: Path | None = None


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
    if config.teacher.device not in _ALLOWED_TEACHER_DEVICES:
        allowed = ", ".join(sorted(_ALLOWED_TEACHER_DEVICES))
        raise ValueError(
            "teacher.device must be one of "
            f"{{{allowed}}}, got {config.teacher.device!r}"
        )
    if config.teacher.dtype not in _ALLOWED_TEACHER_DTYPES:
        allowed = ", ".join(sorted(_ALLOWED_TEACHER_DTYPES))
        raise ValueError(
            f"teacher.dtype must be one of {{{allowed}}}, got {config.teacher.dtype!r}"
        )

    _require_positive("targets.sequence_length", config.targets.sequence_length)
    _require_non_negative("targets.vocab_size", config.targets.vocab_size)
    _require_non_negative("targets.max_new_tokens", config.targets.max_new_tokens)

    if config.targets.hidden_size is not None:
        _require_positive("targets.hidden_size", config.targets.hidden_size)
    if config.targets.num_layers is not None:
        _require_positive("targets.num_layers", config.targets.num_layers)

    if config.targets.dtype not in _ALLOWED_EXPORT_DTYPES:
        allowed = ", ".join(sorted(_ALLOWED_EXPORT_DTYPES))
        raise ValueError(
            f"targets.dtype must be one of {{{allowed}}}, got {config.targets.dtype!r}"
        )

    if config.runtime.exporter_backend not in _ALLOWED_EXPORTER_BACKENDS:
        allowed = ", ".join(sorted(_ALLOWED_EXPORTER_BACKENDS))
        raise ValueError(
            "runtime.exporter_backend must be one of "
            f"{{{allowed}}}, got {config.runtime.exporter_backend!r}"
        )
    if (
        config.runtime.require_resolved_model
        and not config.teacher.resolved_model_id
        and config.runtime.qwen_policy_path is None
    ):
        raise ValueError(
            "runtime.require_resolved_model requires teacher.resolved_model_id "
            "or runtime.qwen_policy_path"
        )

    _require_positive("runtime.batch_size", config.runtime.batch_size)
    _require_positive("runtime.num_shards", config.runtime.num_shards)

    for index, prompt in enumerate(config.targets.prompt_texts):
        if not prompt.strip():
            raise ValueError(f"targets.prompt_texts[{index}] must be non-empty")

    if config.runtime.exporter_backend == "fake":
        _require_positive("targets.vocab_size", config.targets.vocab_size)
        if config.targets.hidden_size is None:
            raise ValueError("targets.hidden_size must be set for fake export")
        if config.targets.num_layers is None:
            raise ValueError("targets.num_layers must be set for fake export")


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

    prompt_file = _optional_path(targets_data.get("prompt_file"))
    if prompt_file is not None and not prompt_file.is_absolute():
        prompt_file = config_path.parent / prompt_file
    qwen_policy_path = _optional_path(runtime_data.get("qwen_policy_path"))
    if qwen_policy_path is not None and not qwen_policy_path.is_absolute():
        config_relative = config_path.parent / qwen_policy_path
        repo_relative = config_path.parent.parent / qwen_policy_path
        qwen_policy_path = (
            config_relative if config_relative.exists() else repo_relative
        )

    config = TeacherExportConfig(
        teacher=TeacherModelConfig(
            family=str(teacher_data.get("family", "qwen")),
            policy_label=str(teacher_data.get("policy_label", "Qwen3.latest")),
            fallback_label=_optional_str(teacher_data.get("fallback_label", "Qwen3.0")),
            resolved_model_id=_optional_str(teacher_data.get("resolved_model_id")),
            tokenizer_id=_optional_str(teacher_data.get("tokenizer_id")),
            trust_remote_code=bool(teacher_data.get("trust_remote_code", False)),
            revision=_optional_str(teacher_data.get("revision")),
            device=str(teacher_data.get("device", "cpu")),
            dtype=str(teacher_data.get("dtype", "auto")),
        ),
        targets=ExportTargetConfig(
            sequence_length=int(targets_data.get("sequence_length", 64)),
            hidden_size=_optional_int(
                targets_data.get("hidden_size", _MISSING), default=128
            ),
            num_layers=_optional_int(
                targets_data.get("num_layers", _MISSING), default=2
            ),
            dtype=str(targets_data.get("dtype", "fp32")),
            include_logits=bool(targets_data.get("include_logits", False)),
            include_attention_targets=bool(
                targets_data.get("include_attention_targets", False)
            ),
            vocab_size=int(targets_data.get("vocab_size", 512)),
            prompt_texts=_string_tuple(targets_data.get("prompt_texts", ())),
            prompt_file=prompt_file,
            max_new_tokens=int(targets_data.get("max_new_tokens", 0)),
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
            require_resolved_model=bool(
                runtime_data.get("require_resolved_model", False)
            ),
            qwen_policy_path=qwen_policy_path,
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


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


def _optional_int(value: Any, *, default: int) -> int | None:
    if value is _MISSING:
        return default
    if value is None:
        return None
    return int(value)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("targets.prompt_texts must be a sequence")
    return tuple(str(item).strip() for item in value)


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _require_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
