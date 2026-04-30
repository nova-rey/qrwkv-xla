from __future__ import annotations

import shutil
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from qrwkv_xla.targets import (
    TargetFlags,
    TeacherTargetManifest,
    inspect_target_bundle,
    write_target_bundle,
)
from qrwkv_xla.teacher_export.base import ExportRequest, ExportResult
from qrwkv_xla.teacher_export.config import validate_teacher_export_config
from qrwkv_xla.teacher_export.prompts import resolve_prompt_texts

INSTALL_HINT = 'python -m pip install -e ".[teacher-hf]"'


class HFTeacherExportError(RuntimeError):
    pass


class HFTeacherExporter:
    name = "hf"

    def export(self, request: ExportRequest) -> ExportResult:
        config = request.config
        validate_teacher_export_config(config)
        if config.targets.include_attention_targets:
            raise NotImplementedError(
                "attention target export is not implemented yet for HFTeacherExporter"
            )
        if not config.teacher.resolved_model_id:
            raise HFTeacherExportError(
                "HF teacher export requires teacher.resolved_model_id or --model-id"
            )

        torch, auto_tokenizer, auto_model = _import_hf_dependencies()
        tokenizer = _load_tokenizer_and_model(auto_tokenizer, config)
        _ensure_pad_token(tokenizer)
        model = _load_model(auto_model, torch, config)
        model.eval()

        prompts = resolve_prompt_texts(config)
        shards = list(self._export_shards(torch, tokenizer, model, config, prompts))
        if not shards:
            raise HFTeacherExportError("HF teacher export produced no shards")

        first_hidden = np.asarray(shards[0]["hidden_states"])
        actual_num_layers = int(first_hidden.shape[1])
        actual_hidden_size = int(first_hidden.shape[3])
        tokenizer_id = config.teacher.tokenizer_id or config.teacher.resolved_model_id
        manifest = TeacherTargetManifest(
            schema_version="0.1",
            teacher_family=config.teacher.family,
            teacher_model_id=config.teacher.resolved_model_id,
            teacher_policy_label=config.teacher.policy_label,
            fallback_policy_label=config.teacher.fallback_label,
            tokenizer_id=tokenizer_id,
            sequence_length=config.targets.sequence_length,
            hidden_size=actual_hidden_size,
            num_layers=actual_num_layers,
            targets=TargetFlags(
                input_ids=True,
                attention_mask=True,
                hidden_states=True,
                logits=config.targets.include_logits,
                attention_targets=False,
            ),
            dtype="fp32",
            created_by="HFTeacherExporter",
            notes=["huggingface teacher exporter bundle"],
            extra={
                "exporter_backend": self.name,
                "revision": config.teacher.revision,
                "trust_remote_code": config.teacher.trust_remote_code,
                "prompt_count": len(prompts),
                "teacher_dtype": config.teacher.dtype,
                "vocab_size": config.targets.vocab_size,
            },
        )

        shutil.rmtree(request.output_dir, ignore_errors=True)
        write_target_bundle(request.output_dir, manifest, shards)
        summary = inspect_target_bundle(request.output_dir)
        return ExportResult(
            output_dir=request.output_dir,
            manifest=manifest,
            shard_count=int(summary["shard_count"]),
            total_examples=int(summary["total_examples"]),
        )

    def _export_shards(
        self,
        torch: Any,
        tokenizer: Any,
        model: Any,
        config: Any,
        prompts: Sequence[str],
    ) -> Iterator[dict[str, np.ndarray]]:
        for batch_prompts in _batched(prompts, config.runtime.batch_size):
            encoded = tokenizer(
                list(batch_prompts),
                padding="max_length",
                truncation=True,
                max_length=config.targets.sequence_length,
                return_tensors="pt",
            )
            encoded = _move_batch_to_device(encoded, config.teacher.device)
            with torch.no_grad():
                outputs = _run_model(model, encoded)
            yield _outputs_to_shard(
                torch=torch,
                encoded=encoded,
                outputs=outputs,
                include_logits=config.targets.include_logits,
            )


def _import_hf_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise HFTeacherExportError(
            "HF teacher export requires optional torch/transformers dependencies. "
            f"Install them with: {INSTALL_HINT}"
        ) from exc
    return torch, AutoTokenizer, AutoModelForCausalLM


def _load_tokenizer_and_model(auto_tokenizer: Any, config: Any) -> Any:
    tokenizer_id = config.teacher.tokenizer_id or config.teacher.resolved_model_id
    return auto_tokenizer.from_pretrained(
        tokenizer_id,
        revision=config.teacher.revision,
        trust_remote_code=config.teacher.trust_remote_code,
    )


def _load_model(auto_model: Any, torch: Any, config: Any) -> Any:
    model = auto_model.from_pretrained(
        config.teacher.resolved_model_id,
        revision=config.teacher.revision,
        trust_remote_code=config.teacher.trust_remote_code,
        torch_dtype=_torch_dtype(torch, config.teacher.dtype),
    )
    if hasattr(model, "to"):
        model = model.to(config.teacher.device)
    return model


def _run_model(model: Any, encoded: dict[str, Any]) -> Any:
    return model(**encoded, output_hidden_states=True, return_dict=True)


def _ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token", None) is not None:
        return
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token is None:
        raise HFTeacherExportError(
            "HF tokenizer has no pad_token and no eos_token to reuse as padding"
        )
    tokenizer.pad_token = eos_token


def _torch_dtype(torch: Any, dtype: str) -> Any:
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype in {"auto", "fp32"}:
        return torch.float32
    raise HFTeacherExportError(f"Unsupported teacher dtype: {dtype!r}")


def _outputs_to_shard(
    *,
    torch: Any,
    encoded: dict[str, Any],
    outputs: Any,
    include_logits: bool,
) -> dict[str, np.ndarray]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise HFTeacherExportError("HF model did not return hidden_states")
    layer_states = tuple(hidden_states[1:] if len(hidden_states) > 1 else hidden_states)
    if not layer_states:
        raise HFTeacherExportError("HF model returned an empty hidden_states sequence")

    stacked_hidden = torch.stack(layer_states, dim=1)
    shard = {
        "input_ids": _tensor_to_numpy(encoded["input_ids"]).astype(np.int32),
        "attention_mask": _tensor_to_numpy(encoded["attention_mask"]).astype(np.int32),
        "hidden_states": _tensor_to_numpy(stacked_hidden).astype(np.float32),
    }
    if include_logits:
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise HFTeacherExportError("HF model did not return logits")
        shard["logits"] = _tensor_to_numpy(logits).astype(np.float32)
    return shard


def _tensor_to_numpy(tensor: Any) -> np.ndarray:
    value = tensor
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _move_batch_to_device(encoded: Any, device: str) -> dict[str, Any]:
    if hasattr(encoded, "to"):
        encoded = encoded.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in dict(encoded).items()
    }


def _batched(items: Sequence[str], batch_size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
