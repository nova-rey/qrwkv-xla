from __future__ import annotations

import shutil
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np

from qrwkv_xla.lm.tokenized_corpus import LoadedTokenizedCorpus, load_tokenized_corpus
from qrwkv_xla.targets import (
    TargetFlags,
    TeacherTargetManifest,
    inspect_target_bundle,
    write_target_bundle,
)
from qrwkv_xla.teacher_export.attention_capture import (
    AttentionCaptureError,
    attention_outputs_from_model_output,
    capture_module_outputs,
    discover_auto_qwen_module_names,
)
from qrwkv_xla.teacher_export.base import ExportRequest, ExportResult
from qrwkv_xla.teacher_export.config import validate_teacher_export_config
from qrwkv_xla.teacher_export.prompts import resolve_prompts

INSTALL_HINT = 'python -m pip install -e ".[teacher-hf]"'


class HFTeacherExportError(RuntimeError):
    pass


class HFTeacherExporter:
    name = "hf"

    def export(self, request: ExportRequest) -> ExportResult:
        config = request.config
        validate_teacher_export_config(config)
        if (
            config.targets.include_attention_targets
            and not config.attention_capture.enabled
        ):
            raise HFTeacherExportError(
                "HF attention target export requires attention_capture.enabled=true "
                "or CLI attention capture flags"
            )
        if not config.teacher.resolved_model_id:
            raise HFTeacherExportError(
                "HF teacher export requires teacher.resolved_model_id or --model-id"
            )

        torch, auto_tokenizer, auto_model = _import_hf_dependencies()
        tokenized = None
        tokenizer = None
        if config.targets.tokenized_corpus is not None:
            tokenized = load_tokenized_corpus(
                config.targets.tokenized_corpus,
                expected_sequence_length=config.targets.sequence_length,
            )
        else:
            tokenizer = _load_tokenizer_and_model(auto_tokenizer, config)
            _ensure_pad_token(tokenizer)
        model = _load_model(auto_model, torch, config)
        model.eval()

        if tokenized is not None:
            prompt_source = _tokenized_prompt_source(tokenized)
            shards = list(
                self._export_tokenized_shards(torch, model, config, tokenized)
            )
        else:
            prompts = resolve_prompts(config)
            prompt_source = prompts.metadata
            shards = list(
                self._export_shards(torch, tokenizer, model, config, prompts.texts)
            )
        if not shards:
            raise HFTeacherExportError("HF teacher export produced no shards")

        first_hidden = np.asarray(shards[0]["hidden_states"])
        actual_num_layers = int(first_hidden.shape[1])
        actual_hidden_size = int(first_hidden.shape[3])
        tokenizer_id = config.teacher.tokenizer_id or (
            tokenized.manifest.tokenizer.tokenizer_id
            if tokenized is not None
            else config.teacher.resolved_model_id
        )
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
                loss_mask=True,
                hidden_states=True,
                logits=config.targets.include_logits,
                attention_targets=config.targets.include_attention_targets,
            ),
            dtype="fp32",
            created_by="HFTeacherExporter",
            notes=["huggingface teacher exporter bundle"],
            prompt_source=prompt_source,
            extra={
                "exporter_backend": self.name,
                "revision": config.teacher.revision,
                "trust_remote_code": config.teacher.trust_remote_code,
                "local_files_only": config.teacher.local_files_only,
                "prompt_count": tokenized.manifest.totals.num_sequences
                if tokenized is not None
                else len(prompts.texts),
                "teacher_dtype": config.teacher.dtype,
                "vocab_size": (
                    tokenized.manifest.tokenizer.vocab_size
                    if tokenized is not None
                    else config.targets.vocab_size
                ),
                "attention_targets": {
                    "kind": "attention_output_vectors",
                    "shape": "[batch,num_layers,sequence_length,hidden_size]",
                    "semantic": "teacher_attention_output_vectors",
                    "capture": "manual_model_output_attribute",
                }
                if config.targets.include_attention_targets
                else None,
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
                model=model,
                attention_capture_config=config.attention_capture,
                include_logits=config.targets.include_logits,
                include_attention_targets=config.targets.include_attention_targets,
            )

    def _export_tokenized_shards(
        self,
        torch: Any,
        model: Any,
        config: Any,
        tokenized: LoadedTokenizedCorpus,
    ) -> Iterator[dict[str, np.ndarray]]:
        batch_size = config.runtime.batch_size
        for start in range(0, tokenized.input_ids.shape[0], batch_size):
            stop = min(start + batch_size, tokenized.input_ids.shape[0])
            encoded = {
                "input_ids": _numpy_to_tensor(
                    torch,
                    np.ascontiguousarray(tokenized.input_ids[start:stop]),
                    config.teacher.device,
                ),
                "attention_mask": _numpy_to_tensor(
                    torch,
                    np.ascontiguousarray(tokenized.attention_mask[start:stop]),
                    config.teacher.device,
                ),
            }
            with torch.no_grad():
                outputs = _run_model(model, encoded)
            shard = _outputs_to_shard(
                torch=torch,
                encoded=encoded,
                outputs=outputs,
                model=model,
                attention_capture_config=config.attention_capture,
                include_logits=config.targets.include_logits,
                include_attention_targets=config.targets.include_attention_targets,
            )
            shard["loss_mask"] = np.ascontiguousarray(
                tokenized.loss_mask[start:stop],
                dtype=np.int32,
            )
            yield shard


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
        local_files_only=config.teacher.local_files_only,
    )


def _load_model(auto_model: Any, torch: Any, config: Any) -> Any:
    model = auto_model.from_pretrained(
        config.teacher.resolved_model_id,
        revision=config.teacher.revision,
        trust_remote_code=config.teacher.trust_remote_code,
        local_files_only=config.teacher.local_files_only,
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
    model: Any,
    attention_capture_config: Any,
    include_logits: bool,
    include_attention_targets: bool = False,
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
        "loss_mask": _tensor_to_numpy(encoded["attention_mask"]).astype(np.int32),
        "hidden_states": _tensor_to_numpy(stacked_hidden).astype(np.float32),
    }
    if include_logits:
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise HFTeacherExportError("HF model did not return logits")
        shard["logits"] = _tensor_to_numpy(logits).astype(np.float32)
    if include_attention_targets:
        try:
            shard["attention_targets"] = _resolve_attention_targets(
                torch=torch,
                model=model,
                outputs=outputs,
                encoded=encoded,
                attention_capture_config=attention_capture_config,
            )
        except AttentionCaptureError as exc:
            raise HFTeacherExportError(str(exc)) from exc
    return shard


def _resolve_attention_targets(
    *,
    torch: Any,
    model: Any,
    outputs: Any,
    encoded: dict[str, Any],
    attention_capture_config: Any,
) -> np.ndarray:
    try:
        return attention_outputs_from_model_output(outputs, torch=torch)
    except AttentionCaptureError:
        pass

    strategy = attention_capture_config.strategy
    if strategy == "explicit_module_names":
        module_names = attention_capture_config.module_names
    elif strategy == "auto_qwen":
        module_names = discover_auto_qwen_module_names(model)
    else:
        raise AttentionCaptureError(
            "attention capture is enabled but no supported strategy was configured"
        )

    return capture_module_outputs(
        model,
        module_names=module_names,
        forward_fn=lambda: _run_model(model, encoded),
        torch=torch,
        output_index=attention_capture_config.output_index,
    )


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


def _numpy_to_tensor(torch: Any, array: np.ndarray, device: str) -> Any:
    if hasattr(torch, "as_tensor"):
        value = torch.as_tensor(array)
    elif hasattr(torch, "tensor"):
        value = torch.tensor(array)
    else:
        raise HFTeacherExportError("torch module does not expose as_tensor or tensor")
    return value.to(device) if hasattr(value, "to") else value


def _batched(items: Sequence[str], batch_size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _tokenized_prompt_source(tokenized: LoadedTokenizedCorpus) -> dict[str, object]:
    return {
        "type": "tokenized_corpus",
        "prompt_count": tokenized.manifest.source.selected_count,
        "path": str(tokenized.root),
        "source": asdict(tokenized.manifest.source),
        "tokenizer": asdict(tokenized.manifest.tokenizer),
        "packing": asdict(tokenized.manifest.packing),
        "totals": asdict(tokenized.manifest.totals),
        "shards": [asdict(shard) for shard in tokenized.manifest.shards],
    }
