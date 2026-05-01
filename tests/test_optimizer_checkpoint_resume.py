from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qrwkv_xla.checkpointing import load_checkpoint, save_checkpoint
from qrwkv_xla.optimizers import OptimizerConfig, init_optimizer_state, optimizer_update


def test_checkpoint_round_trips_adam_optimizer_state(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "adam"
    params = {"w": np.array([1.0, -2.0], dtype=np.float32)}
    grads = {"w": np.array([0.5, -0.25], dtype=np.float32)}
    config = OptimizerConfig(type="adamw", learning_rate=0.1, weight_decay=0.1)
    state = init_optimizer_state(params, config)
    new_params, new_state, _ = optimizer_update(params, grads, state, config)

    save_checkpoint(
        checkpoint_dir,
        new_params,
        student_architecture="tiny_student",
        student_config={"vocab_size": 1, "hidden_size": 1, "num_layers": 1},
        step=1,
        learning_rate=config.learning_rate,
        loss_config={"hidden_mse": {"enabled": True, "weight": 1.0}},
        target_manifest={"hidden_size": 1, "num_layers": 1},
        optimizer_config=config,
        optimizer_state=new_state,
    )

    loaded = load_checkpoint(checkpoint_dir)
    manifest = json.loads((checkpoint_dir / "checkpoint.json").read_text())

    assert manifest["optimizer_config"]["type"] == "adamw"
    assert loaded.optimizer_state is not None
    assert loaded.optimizer_state.type == "adamw"
    assert int(loaded.optimizer_state.step) == 1
    np.testing.assert_allclose(
        loaded.optimizer_state.slots["m"]["w"],
        new_state.slots["m"]["w"],
    )


def test_loaded_adam_state_continues_like_uninterrupted_update(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "adam_resume"
    params = {"w": np.array([1.0, -2.0], dtype=np.float32)}
    first_grad = {"w": np.array([0.5, -0.25], dtype=np.float32)}
    second_grad = {"w": np.array([0.1, 0.2], dtype=np.float32)}
    config = OptimizerConfig(type="adam", learning_rate=0.1)
    state = init_optimizer_state(params, config)
    once_params, once_state, _ = optimizer_update(params, first_grad, state, config)

    save_checkpoint(
        checkpoint_dir,
        once_params,
        student_architecture="tiny_student",
        student_config={"vocab_size": 1, "hidden_size": 1, "num_layers": 1},
        step=1,
        learning_rate=config.learning_rate,
        loss_config={"hidden_mse": {"enabled": True, "weight": 1.0}},
        target_manifest={"hidden_size": 1, "num_layers": 1},
        optimizer_config=config,
        optimizer_state=once_state,
    )
    loaded = load_checkpoint(checkpoint_dir)
    assert loaded.optimizer_state is not None

    resumed_params, resumed_state, _ = optimizer_update(
        loaded.params,
        second_grad,
        loaded.optimizer_state,
        config,
    )
    uninterrupted_params, uninterrupted_state, _ = optimizer_update(
        once_params,
        second_grad,
        once_state,
        config,
    )

    np.testing.assert_allclose(resumed_params["w"], uninterrupted_params["w"])
    np.testing.assert_allclose(
        resumed_state.slots["v"]["w"],
        uninterrupted_state.slots["v"]["w"],
    )
