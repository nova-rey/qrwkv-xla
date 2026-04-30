from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ALLOWED_DTYPES = {"auto", "fp32", "fp16", "bf16"}
_ALLOWED_DEVICES = {"cpu", "auto"}


@dataclass(frozen=True)
class QwenPolicyEntry:
    label: str
    description: str
    resolved_model_id: str | None
    tokenizer_id: str | None
    trust_remote_code: bool
    dtype: str
    device: str
    requires_manual_resolution: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class QwenPolicyMap:
    schema_version: str
    policies: dict[str, QwenPolicyEntry]


@dataclass(frozen=True)
class QwenResolution:
    label: str
    resolved_model_id: str | None
    tokenizer_id: str | None
    trust_remote_code: bool
    dtype: str
    device: str
    is_resolved: bool
    requires_manual_resolution: bool
    notes: tuple[str, ...] = field(default_factory=tuple)


def load_qwen_policy(path: str | Path) -> QwenPolicyMap:
    policy_path = Path(path)
    data = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Qwen policy root must be a mapping")

    raw_policies = data.get("policies")
    if not isinstance(raw_policies, dict):
        raise ValueError("Qwen policy 'policies' must be a mapping")

    policies: dict[str, QwenPolicyEntry] = {}
    for key, value in raw_policies.items():
        if not isinstance(value, dict):
            raise ValueError(f"Qwen policy entry {key!r} must be a mapping")
        label = str(key).strip()
        policies[label] = QwenPolicyEntry(
            label=label,
            description=str(value.get("description", "")).strip(),
            resolved_model_id=_optional_str(value.get("resolved_model_id")),
            tokenizer_id=_optional_str(value.get("tokenizer_id")),
            trust_remote_code=_required_bool(
                value.get("trust_remote_code", False), "trust_remote_code"
            ),
            dtype=str(value.get("dtype", "auto")).strip(),
            device=str(value.get("device", "cpu")).strip(),
            requires_manual_resolution=_required_bool(
                value.get("requires_manual_resolution", True),
                "requires_manual_resolution",
            ),
            notes=_string_tuple(value.get("notes", ())),
        )

    policy_map = QwenPolicyMap(
        schema_version=str(data.get("schema_version", "")).strip(),
        policies=policies,
    )
    validate_qwen_policy(policy_map)
    return policy_map


def validate_qwen_policy(policy: QwenPolicyMap) -> None:
    if not policy.schema_version:
        raise ValueError("Qwen policy schema_version must be non-empty")
    if not policy.policies:
        raise ValueError("Qwen policy must contain at least one policy")

    for key, entry in policy.policies.items():
        if not key.strip():
            raise ValueError("Qwen policy labels must be non-empty")
        if not entry.label.strip():
            raise ValueError(f"Qwen policy entry {key!r} label must be non-empty")
        if key != entry.label:
            raise ValueError(
                f"Qwen policy entry key {key!r} must match label {entry.label!r}"
            )
        if not entry.description.strip():
            raise ValueError(
                f"Qwen policy entry {entry.label!r} description must be non-empty"
            )
        if entry.dtype not in _ALLOWED_DTYPES:
            allowed = ", ".join(sorted(_ALLOWED_DTYPES))
            raise ValueError(
                f"Qwen policy entry {entry.label!r} dtype must be one of {{{allowed}}}"
            )
        if entry.device not in _ALLOWED_DEVICES:
            allowed = ", ".join(sorted(_ALLOWED_DEVICES))
            raise ValueError(
                f"Qwen policy entry {entry.label!r} device must be one of {{{allowed}}}"
            )
        if not isinstance(entry.trust_remote_code, bool):
            raise ValueError(
                f"Qwen policy entry {entry.label!r} trust_remote_code must be bool"
            )
        if not isinstance(entry.requires_manual_resolution, bool):
            raise ValueError(
                "Qwen policy entry "
                f"{entry.label!r} requires_manual_resolution must be bool"
            )
        if not entry.requires_manual_resolution and not _has_text(
            entry.resolved_model_id
        ):
            raise ValueError(
                "Qwen policy entry "
                f"{entry.label!r} resolved_model_id must be non-empty when "
                "requires_manual_resolution is false"
            )


def resolve_qwen_policy(
    label: str,
    *,
    policy_path: str | Path = "configs/qwen_policy.yaml",
    allow_unresolved: bool = False,
) -> QwenResolution:
    policy = load_qwen_policy(policy_path)
    return resolve_qwen_policy_map(policy, label, allow_unresolved=allow_unresolved)


def resolve_qwen_policy_map(
    policy: QwenPolicyMap,
    label: str,
    *,
    allow_unresolved: bool = False,
) -> QwenResolution:
    if label not in policy.policies:
        raise ValueError(f"Unknown Qwen policy label: {label!r}")

    entry = policy.policies[label]
    is_resolved = _has_text(entry.resolved_model_id)
    if not is_resolved and not allow_unresolved:
        raise ValueError(
            f"Qwen policy {label!r} is unresolved. Set resolved_model_id in config, "
            "update configs/qwen_policy.yaml, or pass --model-id. "
            "Use --allow-unresolved for dry-run inspection."
        )

    tokenizer_id = entry.tokenizer_id
    if tokenizer_id is None and entry.resolved_model_id is not None:
        tokenizer_id = entry.resolved_model_id

    return QwenResolution(
        label=entry.label,
        resolved_model_id=entry.resolved_model_id,
        tokenizer_id=tokenizer_id,
        trust_remote_code=entry.trust_remote_code,
        dtype=entry.dtype,
        device=entry.device,
        is_resolved=is_resolved,
        requires_manual_resolution=entry.requires_manual_resolution,
        notes=entry.notes,
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Qwen policy {name} must be bool")
    return value


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("Qwen policy notes must be a sequence")
    return tuple(str(item).strip() for item in value)


def _has_text(value: str | None) -> bool:
    return value is not None and bool(value.strip())
