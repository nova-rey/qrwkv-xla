from __future__ import annotations

from pathlib import Path

import yaml

from qrwkv_xla.config.schema import (
    ModelConfig,
    QRWKVConfig,
    RuntimeConfig,
    TrainingConfig,
)

_ALLOWED_BACKENDS = {"cpu", "tpu", "gpu"}


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def load_config(path: str | Path) -> QRWKVConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")

    runtime_data = data.get("runtime") or {}
    model_data = data.get("model") or {}
    training_data = data.get("training") or {}

    if not isinstance(runtime_data, dict):
        raise ValueError("runtime section must be a mapping")
    if not isinstance(model_data, dict):
        raise ValueError("model section must be a mapping")
    if not isinstance(training_data, dict):
        raise ValueError("training section must be a mapping")

    runtime = RuntimeConfig(
        backend=str(runtime_data.get("backend", "cpu")),
        require_accelerator=bool(runtime_data.get("require_accelerator", False)),
    )
    model = ModelConfig(
        student_architecture=str(model_data.get("student_architecture", "rwkv7_style")),
        vocab_size=int(model_data.get("vocab_size", 512)),
        hidden_size=int(model_data.get("hidden_size", 128)),
        num_layers=int(model_data.get("num_layers", 2)),
        sequence_length=int(model_data.get("sequence_length", 64)),
    )
    training = TrainingConfig(
        batch_size=int(training_data.get("batch_size", 2)),
        max_steps=int(training_data.get("max_steps", 10)),
    )

    _validate(runtime, model, training)
    return QRWKVConfig(runtime=runtime, model=model, training=training, raw=data)


def _validate(
    runtime: RuntimeConfig, model: ModelConfig, training: TrainingConfig
) -> None:
    if runtime.backend not in _ALLOWED_BACKENDS:
        allowed = ", ".join(sorted(_ALLOWED_BACKENDS))
        raise ValueError(
            f"runtime.backend must be one of {{{allowed}}}, got {runtime.backend!r}"
        )

    if model.student_architecture != "rwkv7_style":
        raise ValueError(
            "model.student_architecture must currently be 'rwkv7_style', "
            f"got {model.student_architecture!r}"
        )

    _require_positive("model.hidden_size", model.hidden_size)
    _require_positive("model.vocab_size", model.vocab_size)
    _require_positive("model.num_layers", model.num_layers)
    _require_positive("model.sequence_length", model.sequence_length)
    _require_positive("training.batch_size", training.batch_size)
    _require_positive("training.max_steps", training.max_steps)
