from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from unittest import mock

import jax.numpy as jnp
import pytest

from qrwkv_xla.checkpointing import load_checkpoint
from qrwkv_xla.export import (
    SAFETENSORS_REQUIRED_MESSAGE,
    export_checkpoint_to_hf_safetensors,
    load_hf_safetensors_export,
)
from tests.generation_test_utils import write_generation_checkpoint

ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = ROOT / "scripts" / "export_student_hf_safetensors.py"
SMOKE_SCRIPT = ROOT / "scripts" / "run_export_smoke.py"


def test_export_checkpoint_to_hf_safetensors_round_trips_outputs(
    tmp_path: Path,
) -> None:
    pytest.importorskip("safetensors")
    checkpoint_dir = write_generation_checkpoint(tmp_path, emit_logits=True)
    export_dir = tmp_path / "hf_export"

    result = export_checkpoint_to_hf_safetensors(
        checkpoint_dir,
        export_dir,
        overwrite=True,
    )
    checkpoint = load_checkpoint(checkpoint_dir)
    loaded = load_hf_safetensors_export(export_dir)
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    attention_mask = jnp.ones_like(input_ids)

    original = loaded.student.apply(checkpoint.params, input_ids, attention_mask)
    reloaded = loaded.student.apply(loaded.params, input_ids, attention_mask)

    assert result.config_path.name == "config.json"
    assert (export_dir / "model.safetensors").is_file()
    assert (export_dir / "qrwkv_xla_export.json").is_file()
    assert (export_dir / "weight_map.json").is_file()
    assert loaded.config["model_type"] == "qrwkv_xla_student"
    assert loaded.metadata["student_architecture"] == "tiny_student"
    assert loaded.weight_map["embedding"] == "params.embedding"
    assert original.logits is not None
    assert reloaded.logits is not None
    assert jnp.array_equal(original.hidden_states, reloaded.hidden_states)
    assert jnp.array_equal(original.logits, reloaded.logits)


def test_export_student_hf_safetensors_cli_writes_expected_files(
    tmp_path: Path,
) -> None:
    pytest.importorskip("safetensors")
    checkpoint_dir = write_generation_checkpoint(tmp_path, emit_logits=True)
    export_dir = tmp_path / "cli_export"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXPORT_SCRIPT),
            "--checkpoint",
            str(checkpoint_dir),
            "--output-dir",
            str(export_dir),
            "--overwrite",
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (export_dir / "config.json").is_file()
    assert (export_dir / "model.safetensors").is_file()
    assert (export_dir / "qrwkv_xla_export.json").is_file()
    assert (export_dir / "weight_map.json").is_file()


def test_run_export_smoke_cli_writes_reports(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    output_dir = tmp_path / "artifacts" / "p41"

    completed = subprocess.run(
        [
            sys.executable,
            str(SMOKE_SCRIPT),
            "--output-dir",
            str(output_dir),
            "--overwrite",
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "config.json").is_file()
    assert (output_dir / "model.safetensors").is_file()
    assert (output_dir / "qrwkv_xla_export.json").is_file()
    assert (output_dir / "weight_map.json").is_file()
    assert (output_dir / "export_smoke_report.json").is_file()
    assert (output_dir / "P41_EXPORT_SMOKE_REPORT.md").is_file()
    assert '"passed": true' in completed.stdout


def test_safetensors_missing_message_is_clear() -> None:
    module = importlib.import_module("qrwkv_xla.export.hf_safetensors")
    real_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "safetensors.numpy":
            raise ImportError("missing test dependency")
        return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(ImportError, match="safetensors is required"):
            module._safetensors_numpy()

    with mock.patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(ImportError) as exc_info:
            module._safetensors_numpy()
    assert str(exc_info.value) == SAFETENSORS_REQUIRED_MESSAGE
