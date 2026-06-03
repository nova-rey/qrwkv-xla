from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.checkpointing import (
    load_checkpoint,
    run_checkpoint_resume_update_rehearsal,
)


def test_resume_update_rehearsal_returns_finite_losses(tmp_path: Path) -> None:
    result = run_checkpoint_resume_update_rehearsal(output_dir=tmp_path)

    assert result.status == "pass"
    assert np.isfinite(result.initial_loss)
    assert np.isfinite(result.checkpoint_loss)
    assert np.isfinite(result.resumed_loss)
    assert result.initial_loss > result.checkpoint_loss


def test_resume_update_records_checkpoint_and_final_steps(tmp_path: Path) -> None:
    result = run_checkpoint_resume_update_rehearsal(
        output_dir=tmp_path,
        steps_before_checkpoint=3,
        steps_after_resume=2,
    )

    assert result.checkpoint_step == 3
    assert result.checkpoint_step > 0
    assert result.final_step == result.checkpoint_step + result.steps_after_resume
    assert result.final_step == 5


def test_resume_update_restored_matches_and_loss_is_finite(
    tmp_path: Path,
) -> None:
    result = run_checkpoint_resume_update_rehearsal(output_dir=tmp_path)

    assert result.restored_matches is True
    assert result.resumed_loss_finite is True


def test_loaded_checkpoint_arrays_match_expected_saved_arrays_exactly(
    tmp_path: Path,
) -> None:
    result = run_checkpoint_resume_update_rehearsal(output_dir=tmp_path)
    loaded = load_checkpoint(result.checkpoint_path)

    expected = _expected_checkpoint_weights(
        steps=result.checkpoint_step,
        learning_rate=loaded.manifest.learning_rate,
    )

    np.testing.assert_array_equal(loaded.params["weights"], expected)
    assert loaded.manifest.step == result.checkpoint_step
    assert loaded.manifest.loss_config["mse"]["checkpoint_loss"] == pytest.approx(
        result.checkpoint_loss
    )


def test_resumed_update_changes_params_after_reload(tmp_path: Path) -> None:
    result = run_checkpoint_resume_update_rehearsal(output_dir=tmp_path)
    checkpoint = load_checkpoint(result.checkpoint_path)

    resumed_weights = _expected_checkpoint_weights(
        steps=result.final_step,
        learning_rate=checkpoint.manifest.learning_rate,
    )

    assert result.params_changed_after_resume is True
    assert not np.array_equal(checkpoint.params["weights"], resumed_weights)


def test_resumed_loss_moves_down_or_stays_equal_for_deterministic_case(
    tmp_path: Path,
) -> None:
    result = run_checkpoint_resume_update_rehearsal(output_dir=tmp_path)

    assert result.resumed_loss <= result.checkpoint_loss


def test_resume_update_report_includes_path_and_claims(tmp_path: Path) -> None:
    report = run_checkpoint_resume_update_rehearsal(output_dir=tmp_path).to_report()

    assert report["phase"] == "P108.1"
    assert report["scope"] == "resume_update_closure"
    assert report["path_used"] == "tiny_deterministic_mse_update"
    assert "claims_not_made" in report


def test_resume_update_requires_no_hf_internet_accelerator_or_qwen(
    tmp_path: Path,
) -> None:
    result = run_checkpoint_resume_update_rehearsal(output_dir=tmp_path)

    assert result.status == "pass"
    assert "qwen_specific_support" in result.claims_not_made
    assert "hf_export_ready" in result.claims_not_made


def test_resume_update_does_not_claim_production_checkpoint_or_training(
    tmp_path: Path,
) -> None:
    result = run_checkpoint_resume_update_rehearsal(output_dir=tmp_path)

    assert "production_checkpointing_ready" in result.claims_not_made
    assert "distributed_training_ready" in result.claims_not_made
    assert "training_ready" in result.claims_not_made


def _expected_checkpoint_weights(
    *,
    steps: int,
    learning_rate: float,
) -> np.ndarray:
    weights = np.asarray([0.25, -0.5, 0.75, 1.25], dtype=np.float32)
    target = np.asarray([1.0, -1.0, 0.5, 0.0], dtype=np.float32)
    for _ in range(steps):
        grad = (2.0 / weights.size) * (weights - target)
        weights = weights - learning_rate * grad
    return np.asarray(weights, dtype=np.float32)
