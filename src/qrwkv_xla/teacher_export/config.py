from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ALLOWED_EXPORTER_BACKENDS = {"fake", "hf"}
_ALLOWED_EXPORT_DTYPES = {"fp32", "fp16", "bf16"}
_ALLOWED_TEACHER_DTYPES = {"auto", "fp32", "fp16", "bf16"}
_ALLOWED_TEACHER_DEVICES = {"cpu", "auto"}
_ALLOWED_ATTENTION_CAPTURE_STRATEGIES = {
    "disabled",
    "explicit_module_names",
    "auto_qwen",
}
_MISSING = object()


@dataclass(frozen=True)
class TeacherModelConfig:
    family: str = "qwen"
    policy_label: str = "Qwen3.latest"
    fallback_label: str | None = "Qwen3.0"
    resolved_model_id: str | None = None
    tokenizer_id: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
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
    prompt_corpus: Path | None = None
    tokenized_corpus: Path | None = None
    prompt_split: str | None = None
    prompt_tags: tuple[str, ...] = field(default_factory=tuple)
    prompt_limit: int | None = None
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
class AttentionCaptureConfig:
    enabled: bool = False
    strategy: str = "disabled"
    module_names: tuple[str, ...] = field(default_factory=tuple)
    output_index: int = 0
    require_all_layers: bool = True


@dataclass(frozen=True)
class TeacherExportConfig:
    teacher: TeacherModelConfig = field(default_factory=TeacherModelConfig)
    targets: ExportTargetConfig = field(default_factory=ExportTargetConfig)
    runtime: ExportRuntimeConfig = field(default_factory=ExportRuntimeConfig)
    attention_capture: AttentionCaptureConfig = field(
        default_factory=AttentionCaptureConfig
    )


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
    if config.targets.prompt_limit is not None:
        _require_positive("targets.prompt_limit", config.targets.prompt_limit)

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
    for index, tag in enumerate(config.targets.prompt_tags):
        if not tag.strip():
            raise ValueError(f"targets.prompt_tags[{index}] must be non-empty")

    if config.targets.prompt_split is not None:
        from qrwkv_xla.prompting import canonical_split

        canonical_split(config.targets.prompt_split)

    if (
        config.targets.prompt_corpus is not None
        and not config.targets.prompt_corpus.exists()
    ):
        raise ValueError(
            f"targets.prompt_corpus path does not exist: {config.targets.prompt_corpus}"
        )
    if (
        config.targets.tokenized_corpus is not None
        and not config.targets.tokenized_corpus.exists()
    ):
        raise ValueError(
            "targets.tokenized_corpus path does not exist: "
            f"{config.targets.tokenized_corpus}"
        )

    if config.targets.prompt_corpus is not None and (
        config.targets.prompt_texts or config.targets.prompt_file is not None
    ):
        raise ValueError(
            "Use either prompt_texts/prompt_file or prompt_corpus, not both."
        )
    if config.targets.tokenized_corpus is not None and (
        config.targets.prompt_texts
        or config.targets.prompt_file is not None
        or config.targets.prompt_corpus is not None
    ):
        raise ValueError(
            "Use tokenized_corpus or prompt_texts/prompt_file/prompt_corpus, not both."
        )

    if config.runtime.exporter_backend == "fake":
        _require_positive("targets.vocab_size", config.targets.vocab_size)
        if config.targets.hidden_size is None:
            raise ValueError("targets.hidden_size must be set for fake export")
        if config.targets.num_layers is None:
            raise ValueError("targets.num_layers must be set for fake export")

    if config.attention_capture.strategy not in _ALLOWED_ATTENTION_CAPTURE_STRATEGIES:
        allowed = ", ".join(sorted(_ALLOWED_ATTENTION_CAPTURE_STRATEGIES))
        raise ValueError(
            "attention_capture.strategy must be one of "
            f"{{{allowed}}}, got {config.attention_capture.strategy!r}"
        )
    if config.attention_capture.output_index < 0:
        raise ValueError("attention_capture.output_index must be >= 0")
    for index, module_name in enumerate(config.attention_capture.module_names):
        if not module_name.strip():
            raise ValueError(
                f"attention_capture.module_names[{index}] must be non-empty"
            )
    if (
        config.attention_capture.enabled
        and config.attention_capture.strategy == "explicit_module_names"
        and not config.attention_capture.module_names
    ):
        raise ValueError(
            "attention_capture.module_names must be set for explicit_module_names"
        )


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
    attention_capture_data = _mapping_section(data, "attention_capture")

    config_dir = config_path.resolve().parent
    prompt_file = _config_relative_path(targets_data.get("prompt_file"), config_dir)
    prompt_corpus = _config_relative_path(targets_data.get("prompt_corpus"), config_dir)
    tokenized_corpus = _config_relative_path(
        targets_data.get("tokenized_corpus"), config_dir
    )
    output_dir = _config_relative_path(
        runtime_data.get("output_dir"),
        config_dir,
        default=Path("artifacts/teacher_targets/fake_export"),
    )
    qwen_policy_path = _config_relative_path(
        runtime_data.get("qwen_policy_path"), config_dir
    )

    config = TeacherExportConfig(
        teacher=TeacherModelConfig(
            family=str(teacher_data.get("family", "qwen")),
            policy_label=str(teacher_data.get("policy_label", "Qwen3.latest")),
            fallback_label=_optional_str(teacher_data.get("fallback_label", "Qwen3.0")),
            resolved_model_id=_optional_str(teacher_data.get("resolved_model_id")),
            tokenizer_id=_optional_str(teacher_data.get("tokenizer_id")),
            trust_remote_code=bool(teacher_data.get("trust_remote_code", False)),
            local_files_only=bool(teacher_data.get("local_files_only", False)),
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
            prompt_corpus=prompt_corpus,
            tokenized_corpus=tokenized_corpus,
            prompt_split=_optional_str(targets_data.get("prompt_split")),
            prompt_tags=_string_tuple(targets_data.get("prompt_tags", ())),
            prompt_limit=_optional_int(targets_data.get("prompt_limit"), default=None),
            max_new_tokens=int(targets_data.get("max_new_tokens", 0)),
        ),
        runtime=ExportRuntimeConfig(
            exporter_backend=str(runtime_data.get("exporter_backend", "fake")),
            batch_size=int(runtime_data.get("batch_size", 2)),
            num_shards=int(runtime_data.get("num_shards", 2)),
            seed=int(runtime_data.get("seed", 1234)),
            output_dir=output_dir,
            require_resolved_model=bool(
                runtime_data.get("require_resolved_model", False)
            ),
            qwen_policy_path=qwen_policy_path,
        ),
        attention_capture=AttentionCaptureConfig(
            enabled=bool(attention_capture_data.get("enabled", False)),
            strategy=str(attention_capture_data.get("strategy", "disabled")),
            module_names=_string_tuple(attention_capture_data.get("module_names", ())),
            output_index=int(attention_capture_data.get("output_index", 0)),
            require_all_layers=bool(
                attention_capture_data.get("require_all_layers", True)
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


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


def _config_relative_path(
    value: Any,
    config_dir: Path,
    *,
    default: Path | None = None,
) -> Path | None:
    path = _optional_path(value)
    if path is None:
        return default
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


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
