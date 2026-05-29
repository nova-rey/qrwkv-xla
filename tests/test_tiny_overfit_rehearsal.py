from __future__ import annotations

import jax.numpy as jnp
import pytest

from qrwkv_xla.targets import mse_logits_loss
from qrwkv_xla.training import run_tiny_overfit_rehearsal


def test_tiny_overfit_rehearsal_returns_finite_losses() -> None:
    result = run_tiny_overfit_rehearsal()

    assert result.loss_finite is True
    assert jnp.isfinite(result.initial_loss)
    assert jnp.isfinite(result.final_loss)


def test_tiny_overfit_rehearsal_performs_requested_steps() -> None:
    result = run_tiny_overfit_rehearsal(steps=4)

    assert result.steps == 4


def test_tiny_overfit_rehearsal_moves_loss_down() -> None:
    result = run_tiny_overfit_rehearsal()

    assert result.loss_moved is True
    assert result.final_loss < result.initial_loss


def test_tiny_overfit_rehearsal_identifies_fallback_path() -> None:
    result = run_tiny_overfit_rehearsal()

    assert result.path_used == "tiny_trainable_logit_head"
    assert "training_ready" in result.claims_not_made


def test_tiny_overfit_rehearsal_report_is_cpu_and_offline_only() -> None:
    report = run_tiny_overfit_rehearsal().to_report()

    assert report["phase"] == "P96"
    assert report["status"] == "pass"
    assert report["live_teacher_required"] is False
    assert report["hf_or_qwen_required"] is False
    assert report["gpu_or_tpu_required"] is False
    assert report["training_kind"] == "tiny_controlled_rehearsal"


def test_p95_logits_shape_mismatch_still_fails() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        mse_logits_loss(jnp.zeros((1, 2, 3)), jnp.zeros((1, 2, 4)))
