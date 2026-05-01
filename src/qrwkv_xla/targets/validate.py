from __future__ import annotations

from dataclasses import asdict
from typing import Any

from qrwkv_xla.targets.manifest import TargetFlags, TeacherTargetManifest

_ALLOWED_DTYPES = {"fp32", "bf16", "fp16"}
_MANIFEST_FIELDS = {
    "schema_version",
    "teacher_family",
    "teacher_model_id",
    "teacher_policy_label",
    "fallback_policy_label",
    "tokenizer_id",
    "sequence_length",
    "hidden_size",
    "num_layers",
    "targets",
    "dtype",
    "created_by",
    "notes",
    "prompt_source",
    "prompt_provenance",
}


def manifest_from_dict(data: dict[str, Any]) -> TeacherTargetManifest:
    if not isinstance(data, dict):
        raise ValueError("Manifest root must be a mapping")

    target_data = data.get("targets") or {}
    if not isinstance(target_data, dict):
        raise ValueError("targets must be a mapping")

    prompt_source = data.get("prompt_source")
    if prompt_source is None and "prompt_provenance" in data:
        prompt_source = data.get("prompt_provenance")

    manifest = TeacherTargetManifest(
        schema_version=str(data.get("schema_version", "")),
        teacher_family=str(data.get("teacher_family", "")),
        teacher_model_id=_optional_str(data.get("teacher_model_id")),
        teacher_policy_label=str(data.get("teacher_policy_label", "")),
        fallback_policy_label=_optional_str(data.get("fallback_policy_label")),
        tokenizer_id=_optional_str(data.get("tokenizer_id")),
        sequence_length=int(data.get("sequence_length", 0)),
        hidden_size=int(data.get("hidden_size", 0)),
        num_layers=int(data.get("num_layers", 0)),
        targets=TargetFlags(
            input_ids=bool(target_data.get("input_ids", True)),
            attention_mask=bool(target_data.get("attention_mask", True)),
            hidden_states=bool(target_data.get("hidden_states", True)),
            logits=bool(target_data.get("logits", False)),
            attention_targets=bool(target_data.get("attention_targets", False)),
        ),
        dtype=str(data.get("dtype", "")),
        created_by=str(data.get("created_by", "teacher_exporter")),
        notes=[str(note) for note in data.get("notes", [])],
        prompt_source=prompt_source,
        extra={
            key: value for key, value in data.items() if key not in _MANIFEST_FIELDS
        },
    )
    validate_manifest(manifest)
    return manifest


def manifest_to_dict(manifest: TeacherTargetManifest) -> dict[str, Any]:
    payload = asdict(manifest)
    extra = payload.pop("extra")
    if payload.get("prompt_source") is None:
        payload.pop("prompt_source")
    payload.update(extra)
    return payload


def validate_manifest(manifest: TeacherTargetManifest) -> None:
    if not manifest.schema_version.strip():
        raise ValueError("schema_version must be non-empty")
    if not manifest.teacher_family.strip():
        raise ValueError("teacher_family must be non-empty")
    if not manifest.teacher_policy_label.strip():
        raise ValueError("teacher_policy_label must be non-empty")
    if manifest.sequence_length <= 0:
        raise ValueError(f"sequence_length must be > 0, got {manifest.sequence_length}")
    if manifest.hidden_size <= 0:
        raise ValueError(f"hidden_size must be > 0, got {manifest.hidden_size}")
    if manifest.num_layers <= 0:
        raise ValueError(f"num_layers must be > 0, got {manifest.num_layers}")
    if manifest.dtype not in _ALLOWED_DTYPES:
        allowed = ", ".join(sorted(_ALLOWED_DTYPES))
        raise ValueError(f"dtype must be one of {{{allowed}}}, got {manifest.dtype!r}")
    if not manifest.created_by.strip():
        raise ValueError("created_by must be non-empty")
    if not manifest.targets.input_ids:
        raise ValueError("targets.input_ids must currently be true")
    if manifest.prompt_source is not None:
        _validate_prompt_source(manifest.prompt_source)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _validate_prompt_source(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("prompt_source must be a mapping")
    source_type = str(value.get("type", "")).strip()
    if source_type not in {"default", "inline", "file", "corpus"}:
        raise ValueError(
            "prompt_source.type must be one of {default, inline, file, corpus}"
        )
    prompt_count = int(value.get("prompt_count", 0))
    if prompt_count <= 0:
        raise ValueError("prompt_source.prompt_count must be > 0")
    if source_type == "corpus":
        required = {"corpus_id", "corpus_sha256", "corpus_path", "prompt_ids"}
        missing = sorted(key for key in required if key not in value)
        if missing:
            raise ValueError(
                f"prompt_source missing required corpus keys: {', '.join(missing)}"
            )
        if not isinstance(value["prompt_ids"], list) or not value["prompt_ids"]:
            raise ValueError("prompt_source.prompt_ids must be a non-empty list")
