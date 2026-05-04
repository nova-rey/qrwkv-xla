from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import qrwkv_xla.teacher_export.hf as hf_module
from qrwkv_xla.targets import read_shard, validate_target_bundle
from qrwkv_xla.targets.store import shard_path
from qrwkv_xla.teacher_export import (
    ExportRequest,
    HFTeacherExporter,
    HFTeacherExportError,
    TeacherExportConfig,
    get_teacher_exporter,
)


class FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = np.asarray(value)

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def to(self, _device: str) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class FakeTorch:
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"

    @staticmethod
    def stack(values: tuple[FakeTensor, ...], dim: int) -> FakeTensor:
        return FakeTensor(np.stack([value.value for value in values], axis=dim))

    @staticmethod
    def no_grad() -> FakeNoGrad:
        return FakeNoGrad()


class FakeNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


class FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, prompts: list[str], **kwargs: object) -> dict[str, FakeTensor]:
        self.calls.append({"prompts": prompts, **kwargs})
        batch = len(prompts)
        sequence_length = int(kwargs["max_length"])
        input_ids = np.arange(batch * sequence_length).reshape(batch, sequence_length)
        attention_mask = np.ones((batch, sequence_length), dtype=np.int64)
        return {
            "input_ids": FakeTensor(input_ids),
            "attention_mask": FakeTensor(attention_mask),
        }


class FakeModel:
    def __init__(self) -> None:
        self.eval_called = False
        self.device: str | None = None

    def to(self, device: str) -> FakeModel:
        self.device = device
        return self

    def eval(self) -> None:
        self.eval_called = True

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        input_ids = kwargs["input_ids"].value
        batch, sequence_length = input_ids.shape
        hidden_states = (
            FakeTensor(np.zeros((batch, sequence_length, 3), dtype=np.float32)),
            FakeTensor(np.ones((batch, sequence_length, 3), dtype=np.float32)),
            FakeTensor(np.full((batch, sequence_length, 3), 2.0, dtype=np.float32)),
        )
        logits = FakeTensor(np.zeros((batch, sequence_length, 7), dtype=np.float32))
        attention_outputs = (
            FakeTensor(np.full((batch, sequence_length, 3), 3.0, dtype=np.float32)),
            FakeTensor(np.full((batch, sequence_length, 3), 4.0, dtype=np.float32)),
        )
        return SimpleNamespace(
            hidden_states=hidden_states,
            logits=logits,
            attention_outputs=attention_outputs,
        )

    def named_modules(self):
        yield "", self


def test_registry_returns_hf_exporter_without_runtime_import_error() -> None:
    exporter = get_teacher_exporter("hf")
    assert exporter.name == "hf"


def test_hf_exporter_writes_manifest_compatible_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    tokenizer_calls = []
    model_calls = []
    auto_tokenizer = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: (
            tokenizer_calls.append((args, kwargs)) or tokenizer
        )
    )
    auto_model = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: (
            model_calls.append((args, kwargs)) or model
        )
    )
    monkeypatch.setattr(
        hf_module,
        "_import_hf_dependencies",
        lambda: (FakeTorch, auto_tokenizer, auto_model),
    )
    config = TeacherExportConfig()
    config = replace(
        config,
        teacher=replace(
            config.teacher,
            family="hf-causal-lm",
            resolved_model_id="tiny-model",
            tokenizer_id="tiny-tokenizer",
            device="cpu",
            dtype="auto",
            local_files_only=True,
        ),
        targets=replace(
            config.targets,
            sequence_length=5,
            hidden_size=None,
            num_layers=None,
            include_logits=True,
            prompt_texts=("a", "b", "c"),
        ),
        runtime=replace(
            config.runtime,
            exporter_backend="hf",
            batch_size=2,
            output_dir=tmp_path / "bundle",
        ),
    )

    result = HFTeacherExporter().export(
        ExportRequest(config=config, output_dir=config.runtime.output_dir)
    )

    validate_target_bundle(result.output_dir)
    assert result.shard_count == 2
    assert result.total_examples == 3
    assert result.manifest.hidden_size == 3
    assert result.manifest.num_layers == 2
    assert result.manifest.targets.logits is True
    assert result.manifest.prompt_source == {"type": "inline", "prompt_count": 3}
    assert result.manifest.teacher_family == "hf-causal-lm"
    assert result.manifest.teacher_model_id == "tiny-model"
    assert result.manifest.tokenizer_id == "tiny-tokenizer"
    assert result.manifest.dtype == "fp32"
    assert result.manifest.extra["local_files_only"] is True
    assert tokenizer_calls[0][1]["local_files_only"] is True
    assert model_calls[0][1]["local_files_only"] is True
    assert tokenizer.pad_token == "<eos>"
    assert tokenizer.calls[0]["padding"] == "max_length"
    assert tokenizer.calls[0]["truncation"] is True
    shard = read_shard(shard_path(result.output_dir, 0))
    assert shard["input_ids"].dtype == np.int32
    assert shard["attention_mask"].dtype == np.int32
    assert shard["loss_mask"].dtype == np.int32
    assert shard["hidden_states"].shape == (2, 2, 5, 3)
    assert shard["hidden_states"].dtype == np.float32
    assert "logits" in shard
    assert model.eval_called is True
    assert model.device == "cpu"


def test_hf_exporter_requires_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def raise_missing() -> object:
        raise HFTeacherExportError(
            "HF teacher export requires optional torch/transformers dependencies. "
            'Install them with: python -m pip install -e ".[teacher-hf]"'
        )

    monkeypatch.setattr(hf_module, "_import_hf_dependencies", raise_missing)
    config = TeacherExportConfig()
    config = replace(
        config,
        teacher=replace(
            config.teacher,
            family="hf-causal-lm",
            resolved_model_id="tiny-model",
        ),
        targets=replace(config.targets, hidden_size=None, num_layers=None),
        runtime=replace(config.runtime, exporter_backend="hf", output_dir=tmp_path),
    )

    with pytest.raises(HFTeacherExportError, match=r"\.\[teacher-hf\]"):
        HFTeacherExporter().export(ExportRequest(config=config, output_dir=tmp_path))


def test_hf_tokenizer_without_pad_or_eos_fails() -> None:
    tokenizer = SimpleNamespace(pad_token=None, eos_token=None)

    with pytest.raises(HFTeacherExportError, match="no pad_token and no eos_token"):
        hf_module._ensure_pad_token(tokenizer)


def test_hf_exporter_omits_logits_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    auto_tokenizer = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: tokenizer
    )
    auto_model = SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: model)
    monkeypatch.setattr(
        hf_module,
        "_import_hf_dependencies",
        lambda: (FakeTorch, auto_tokenizer, auto_model),
    )
    config = TeacherExportConfig()
    config = replace(
        config,
        teacher=replace(
            config.teacher,
            family="hf-causal-lm",
            resolved_model_id="tiny-model",
        ),
        targets=replace(
            config.targets,
            hidden_size=None,
            num_layers=None,
            include_logits=False,
            prompt_texts=("a",),
        ),
        runtime=replace(config.runtime, exporter_backend="hf", output_dir=tmp_path),
    )

    result = HFTeacherExporter().export(
        ExportRequest(config=config, output_dir=tmp_path)
    )
    shard = read_shard(shard_path(result.output_dir, 0))
    assert "logits" not in shard
    assert result.manifest.targets.logits is False


def test_hf_exporter_can_include_attention_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    auto_tokenizer = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: tokenizer
    )
    auto_model = SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: model)
    monkeypatch.setattr(
        hf_module,
        "_import_hf_dependencies",
        lambda: (FakeTorch, auto_tokenizer, auto_model),
    )
    config = TeacherExportConfig()
    config = replace(
        config,
        teacher=replace(
            config.teacher,
            family="qwen",
            resolved_model_id="tiny-model",
        ),
        targets=replace(
            config.targets,
            hidden_size=None,
            num_layers=None,
            include_attention_targets=True,
            prompt_texts=("a",),
        ),
        runtime=replace(config.runtime, exporter_backend="hf", output_dir=tmp_path),
        attention_capture=replace(
            config.attention_capture,
            enabled=True,
            strategy="auto_qwen",
        ),
    )

    result = HFTeacherExporter().export(
        ExportRequest(config=config, output_dir=tmp_path)
    )
    shard = read_shard(shard_path(result.output_dir, 0))
    assert shard["attention_targets"].shape == (1, 2, 64, 3)
    assert result.manifest.targets.attention_targets is True
