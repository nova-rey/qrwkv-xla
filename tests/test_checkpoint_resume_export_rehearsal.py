from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest import mock

import jax.numpy as jnp
import pytest

from qrwkv_xla.checkpointing import (
    load_checkpoint,
    run_checkpoint_resume_export_rehearsal,
)
from qrwkv_xla.export import load_hf_safetensors_export

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_checkpoint_resume_export_rehearsal.py"


def test_checkpoint_resume_export_rehearsal_passes_with_safetensors(
    tmp_path: Path,
) -> None:
    pytest.importorskip("safetensors")

    result = run_checkpoint_resume_export_rehearsal(output_dir=tmp_path)

    assert result.status == "pass"
    assert result.checkpoint_roundtrip is True
    assert result.export_roundtrip is True
    assert result.output_match is True
    assert result.tensor_count > 0


def test_rehearsal_writes_checkpoint_manifest_and_params(tmp_path: Path) -> None:
    result = run_checkpoint_resume_export_rehearsal(output_dir=tmp_path)
    checkpoint_dir = Path(result.checkpoint_path)

    assert (checkpoint_dir / "checkpoint.json").is_file()
    assert (checkpoint_dir / "params.npz").is_file()
    loaded = load_checkpoint(checkpoint_dir)
    assert loaded.manifest.step == result.checkpoint_step
    assert loaded.manifest.student_architecture == "tiny_student"


def test_rehearsal_export_reload_preserves_outputs(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")

    result = run_checkpoint_resume_export_rehearsal(output_dir=tmp_path)
    checkpoint = load_checkpoint(result.checkpoint_path)
    loaded_export = load_hf_safetensors_export(result.export_path)
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    attention_mask = jnp.ones_like(input_ids)

    checkpoint_output = loaded_export.student.apply(
        checkpoint.params,
        input_ids,
        attention_mask,
    )
    export_output = loaded_export.student.apply(
        loaded_export.params,
        input_ids,
        attention_mask,
    )

    assert checkpoint_output.logits is not None
    assert export_output.logits is not None
    assert jnp.array_equal(checkpoint_output.hidden_states, export_output.hidden_states)
    assert jnp.array_equal(checkpoint_output.logits, export_output.logits)


def test_rehearsal_reports_unavailable_when_safetensors_is_missing(
    tmp_path: Path,
) -> None:
    with mock.patch(
        "qrwkv_xla.export.hf_safetensors._safetensors_numpy",
        side_effect=ImportError("safetensors unavailable for test"),
    ):
        result = run_checkpoint_resume_export_rehearsal(output_dir=tmp_path)

    assert result.status == "unavailable"
    assert result.checkpoint_roundtrip is True
    assert result.export_roundtrip is False
    assert result.output_match is True
    assert result.export_path is None
    assert "safetensors unavailable" in result.reason


def test_rehearsal_report_states_p108_scope_and_counts(tmp_path: Path) -> None:
    result = run_checkpoint_resume_export_rehearsal(output_dir=tmp_path)
    report = result.to_report()

    assert report["phase"] == "P108"
    assert report["scope"] == "checkpoint_resume_export_rehearsal"
    assert report["student_architecture"] == "tiny_student"
    assert report["checkpoint_step"] == 2
    assert "production_checkpointing_ready" in report["claims_not_made"]


def test_rehearsal_does_not_claim_training_qwen_or_tokenizer_remapping(
    tmp_path: Path,
) -> None:
    result = run_checkpoint_resume_export_rehearsal(output_dir=tmp_path)

    assert "training_ready" in result.claims_not_made
    assert "qwen_specific_support" in result.claims_not_made
    assert "tokenizer_remapping_supported" in result.claims_not_made


def test_checkpoint_resume_export_cli_writes_json_report(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(tmp_path / "artifacts" / "p108"),
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"phase": "P108"' in completed.stdout
    assert '"checkpoint_roundtrip": true' in completed.stdout
