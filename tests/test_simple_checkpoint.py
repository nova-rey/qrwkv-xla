from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from qrwkv_xla.checkpointing import (
    checkpoint_exists,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_manifest,
)
from qrwkv_xla.students import create_student


def test_save_and_load_tiny_student_params(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "tiny"
    student = create_student(
        "tiny_student",
        vocab_size=32,
        hidden_size=8,
        num_layers=2,
    )
    params = student.init_params(jax.random.PRNGKey(0))

    save_checkpoint(
        checkpoint_dir,
        params,
        student_architecture="tiny_student",
        student_config={"vocab_size": 32, "hidden_size": 8, "num_layers": 2},
        step=7,
        learning_rate=0.1,
        loss_config={"hidden_mse": {"enabled": True, "weight": 1.0}},
        target_manifest={"hidden_size": 8, "num_layers": 2},
    )
    loaded = load_checkpoint(checkpoint_dir)

    assert checkpoint_exists(checkpoint_dir)
    assert loaded.checkpoint_dir == checkpoint_dir
    assert loaded.manifest.step == 7
    assert loaded.manifest.student_architecture == "tiny_student"
    np.testing.assert_allclose(loaded.params["embedding"], params["embedding"])


def test_load_checkpoint_allows_missing_lr_schedule_metadata(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "old"
    save_checkpoint(
        checkpoint_dir,
        {"x": np.ones((1,))},
        student_architecture="tiny_student",
        student_config={"vocab_size": 1, "hidden_size": 1, "num_layers": 1},
        step=1,
        learning_rate=0.1,
        loss_config={"hidden_mse": {"enabled": True, "weight": 1.0}},
        target_manifest={"hidden_size": 1, "num_layers": 1},
    )
    manifest_path = checkpoint_dir / "checkpoint.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("lr_schedule")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_checkpoint(checkpoint_dir)

    assert loaded.manifest.lr_schedule == {}


def test_save_and_load_rwkv7_reference_params(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "rwkv7"
    student = create_student(
        "rwkv7_reference",
        vocab_size=32,
        hidden_size=8,
        num_layers=2,
    )
    params = student.init_params(jax.random.PRNGKey(1))

    save_checkpoint(
        checkpoint_dir,
        params,
        student_architecture="rwkv7_reference",
        student_config={"vocab_size": 32, "hidden_size": 8, "num_layers": 2},
        step=2,
        learning_rate=0.01,
        loss_config={"hidden_mse": {"enabled": True, "weight": 1.0}},
        target_manifest={"hidden_size": 8, "num_layers": 2},
    )
    loaded = load_checkpoint(checkpoint_dir)

    assert loaded.manifest.student_architecture == "rwkv7_reference"
    assert set(loaded.params) == set(params)
    np.testing.assert_allclose(loaded.params["embedding"], params["embedding"])


def test_checkpoint_must_live_under_checkpoints(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="checkpoints"):
        save_checkpoint(
            tmp_path / "not_allowed",
            {"x": np.ones((1,))},
            student_architecture="tiny_student",
            student_config={"vocab_size": 1, "hidden_size": 1, "num_layers": 1},
            step=0,
            learning_rate=0.1,
            loss_config={},
            target_manifest={},
        )


def test_overwrite_false_rejects_existing_checkpoint_dir(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "existing"
    checkpoint_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        save_checkpoint(
            checkpoint_dir,
            {"x": np.ones((1,), dtype=np.float32)},
            student_architecture="tiny_student",
            student_config={"vocab_size": 1, "hidden_size": 1, "num_layers": 1},
            step=0,
            learning_rate=0.1,
            loss_config={"hidden_mse": {"enabled": True, "weight": 1.0}},
            target_manifest={"hidden_size": 1, "num_layers": 1},
        )


def test_load_rejects_missing_array(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "broken"
    save_checkpoint(
        checkpoint_dir,
        {"x": np.ones((1,))},
        student_architecture="tiny_student",
        student_config={"vocab_size": 1, "hidden_size": 1, "num_layers": 1},
        step=0,
        learning_rate=0.1,
        loss_config={"hidden_mse": {"enabled": True, "weight": 1.0}},
        target_manifest={"hidden_size": 1, "num_layers": 1},
    )
    manifest_path = checkpoint_dir / "checkpoint.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["param_tree"]["children"]["x"]["key"] = "arr_999999"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="missing array"):
        load_checkpoint(checkpoint_dir)


def test_validate_manifest_rejects_invalid_step(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "invalid_step"
    save_checkpoint(
        checkpoint_dir,
        {"x": np.ones((1,), dtype=np.float32)},
        student_architecture="tiny_student",
        student_config={"vocab_size": 1, "hidden_size": 1, "num_layers": 1},
        step=0,
        learning_rate=0.1,
        loss_config={"hidden_mse": {"enabled": True, "weight": 1.0}},
        target_manifest={"hidden_size": 1, "num_layers": 1},
    )
    loaded = load_checkpoint(checkpoint_dir)
    bad_manifest = loaded.manifest.__class__(**{**loaded.manifest.__dict__, "step": -1})
    with pytest.raises(ValueError, match="step"):
        validate_checkpoint_manifest(bad_manifest)
