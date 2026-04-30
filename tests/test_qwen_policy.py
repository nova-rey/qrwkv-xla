from __future__ import annotations

from pathlib import Path

import pytest

from qrwkv_xla.teacher_export.qwen_policy import (
    QwenPolicyEntry,
    QwenPolicyMap,
    load_qwen_policy,
    resolve_qwen_policy,
    resolve_qwen_policy_map,
    validate_qwen_policy,
)

ROOT = Path(__file__).resolve().parents[1]


def test_default_qwen_policy_loads_with_expected_labels() -> None:
    policy = load_qwen_policy(ROOT / "configs" / "qwen_policy.yaml")

    assert "Qwen3.latest" in policy.policies
    assert "Qwen3.0" in policy.policies
    assert "qwen-tiny-smoke" in policy.policies


def test_unresolved_policy_raises_without_allow() -> None:
    with pytest.raises(ValueError, match="is unresolved"):
        resolve_qwen_policy(
            "Qwen3.latest",
            policy_path=ROOT / "configs" / "qwen_policy.yaml",
        )


def test_unresolved_policy_can_be_inspected_with_allow() -> None:
    resolution = resolve_qwen_policy(
        "Qwen3.latest",
        policy_path=ROOT / "configs" / "qwen_policy.yaml",
        allow_unresolved=True,
    )

    assert resolution.is_resolved is False
    assert resolution.resolved_model_id is None
    assert resolution.requires_manual_resolution is True
    assert resolution.notes


def test_unknown_label_raises() -> None:
    with pytest.raises(ValueError, match="Unknown Qwen policy label"):
        resolve_qwen_policy(
            "Qwen3.missing",
            policy_path=ROOT / "configs" / "qwen_policy.yaml",
            allow_unresolved=True,
        )


def test_tokenizer_defaults_to_model_id() -> None:
    policy = QwenPolicyMap(
        schema_version="0.1",
        policies={
            "local": QwenPolicyEntry(
                label="local",
                description="local resolved policy",
                resolved_model_id="local/model",
                tokenizer_id=None,
                trust_remote_code=False,
                dtype="auto",
                device="cpu",
                requires_manual_resolution=False,
            )
        },
    )
    validate_qwen_policy(policy)

    resolution = resolve_qwen_policy_map(policy, "local")

    assert resolution.resolved_model_id == "local/model"
    assert resolution.tokenizer_id == "local/model"
    assert resolution.is_resolved is True


def test_invalid_policy_fields_raise() -> None:
    policy = QwenPolicyMap(
        schema_version="0.1",
        policies={
            "bad": QwenPolicyEntry(
                label="bad",
                description="bad policy",
                resolved_model_id=None,
                tokenizer_id=None,
                trust_remote_code=False,
                dtype="auto",
                device="cpu",
                requires_manual_resolution=False,
            )
        },
    )

    with pytest.raises(ValueError, match="resolved_model_id must be non-empty"):
        validate_qwen_policy(policy)


def test_invalid_policy_yaml_raises(tmp_path: Path) -> None:
    policy_path = tmp_path / "bad_policy.yaml"
    policy_path.write_text(
        """
schema_version: "0.1"
policies:
  Broken:
    description: ""
    resolved_model_id: null
    tokenizer_id: null
    trust_remote_code: true
    dtype: weird
    device: cpu
    requires_manual_resolution: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_qwen_policy(policy_path)
