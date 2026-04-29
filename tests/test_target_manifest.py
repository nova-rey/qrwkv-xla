from __future__ import annotations

import pytest

from qrwkv_xla.targets import manifest_from_dict, manifest_to_dict


def _valid_manifest_dict() -> dict:
    return {
        "schema_version": "0.1",
        "teacher_family": "qwen",
        "teacher_model_id": None,
        "teacher_policy_label": "Qwen3.latest",
        "fallback_policy_label": "Qwen3.0",
        "tokenizer_id": None,
        "sequence_length": 64,
        "hidden_size": 128,
        "num_layers": 2,
        "targets": {
            "input_ids": True,
            "attention_mask": True,
            "hidden_states": True,
            "logits": False,
            "attention_targets": False,
        },
        "dtype": "fp32",
        "created_by": "teacher_exporter",
        "notes": [],
    }


def test_valid_manifest_parses() -> None:
    manifest = manifest_from_dict(_valid_manifest_dict())
    assert manifest.teacher_family == "qwen"
    assert manifest.targets.hidden_states is True


def test_manifest_round_trip() -> None:
    manifest = manifest_from_dict(_valid_manifest_dict())
    payload = manifest_to_dict(manifest)
    assert payload["teacher_policy_label"] == "Qwen3.latest"
    assert payload["targets"]["input_ids"] is True


def test_invalid_dtype_raises() -> None:
    payload = _valid_manifest_dict()
    payload["dtype"] = "int8"
    with pytest.raises(ValueError, match="dtype"):
        manifest_from_dict(payload)


def test_invalid_sequence_length_raises() -> None:
    payload = _valid_manifest_dict()
    payload["sequence_length"] = 0
    with pytest.raises(ValueError, match="sequence_length"):
        manifest_from_dict(payload)


@pytest.mark.parametrize("policy_label", ["", "   "])
def test_policy_label_must_be_non_empty(policy_label: str) -> None:
    payload = _valid_manifest_dict()
    payload["teacher_policy_label"] = policy_label
    with pytest.raises(ValueError, match="teacher_policy_label"):
        manifest_from_dict(payload)
