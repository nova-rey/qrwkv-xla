from __future__ import annotations

from qrwkv_xla.burn import evaluate_distributed_training_readiness


def test_ready_when_all_training_sync_predicates_pass_without_checkpoint_gate() -> None:
    readiness = evaluate_distributed_training_readiness(
        distributed_example_sharding_verified=True,
        collective_sync_probe_verified=True,
        gradient_sync_enabled=True,
        gradient_sync_verified=True,
        parameter_sync_verified=True,
        optimizer_state_sync_verified=True,
        loss_is_global=True,
        checkpoint_fingerprint_match=False,
        checkpoint_fingerprint_match_required=False,
        process_count=4,
    )

    assert readiness.distributed_training_ready is True
    assert readiness.missing_predicates == ()


def test_not_ready_if_process_count_is_one() -> None:
    readiness = evaluate_distributed_training_readiness(
        distributed_example_sharding_verified=True,
        collective_sync_probe_verified=True,
        gradient_sync_enabled=True,
        gradient_sync_verified=True,
        parameter_sync_verified=True,
        optimizer_state_sync_verified=True,
        loss_is_global=True,
        checkpoint_fingerprint_match=False,
        checkpoint_fingerprint_match_required=False,
        process_count=1,
    )

    assert readiness.distributed_training_ready is False
    assert "jax_process_count_gt_1" in readiness.missing_predicates


def test_not_ready_if_gradient_sync_is_false() -> None:
    readiness = evaluate_distributed_training_readiness(
        distributed_example_sharding_verified=True,
        collective_sync_probe_verified=True,
        gradient_sync_enabled=False,
        gradient_sync_verified=False,
        parameter_sync_verified=True,
        optimizer_state_sync_verified=True,
        loss_is_global=True,
        checkpoint_fingerprint_match=False,
        checkpoint_fingerprint_match_required=False,
        process_count=4,
    )

    assert readiness.distributed_training_ready is False
    assert "gradient_sync_enabled" in readiness.missing_predicates
    assert "gradient_sync_verified" in readiness.missing_predicates


def test_not_ready_if_parameter_or_optimizer_sync_is_false() -> None:
    readiness = evaluate_distributed_training_readiness(
        distributed_example_sharding_verified=True,
        collective_sync_probe_verified=True,
        gradient_sync_enabled=True,
        gradient_sync_verified=True,
        parameter_sync_verified=False,
        optimizer_state_sync_verified=False,
        loss_is_global=True,
        checkpoint_fingerprint_match=False,
        checkpoint_fingerprint_match_required=False,
        process_count=4,
    )

    assert readiness.distributed_training_ready is False
    assert "parameter_sync_verified" in readiness.missing_predicates
    assert "optimizer_state_sync_verified" in readiness.missing_predicates


def test_not_ready_if_loss_is_local_only() -> None:
    readiness = evaluate_distributed_training_readiness(
        distributed_example_sharding_verified=True,
        collective_sync_probe_verified=True,
        gradient_sync_enabled=True,
        gradient_sync_verified=True,
        parameter_sync_verified=True,
        optimizer_state_sync_verified=True,
        loss_is_global=False,
        checkpoint_fingerprint_match=False,
        checkpoint_fingerprint_match_required=False,
        process_count=4,
    )

    assert readiness.distributed_training_ready is False
    assert "loss_is_global" in readiness.missing_predicates


def test_checkpoint_gate_can_be_required() -> None:
    readiness = evaluate_distributed_training_readiness(
        distributed_example_sharding_verified=True,
        collective_sync_probe_verified=True,
        gradient_sync_enabled=True,
        gradient_sync_verified=True,
        parameter_sync_verified=True,
        optimizer_state_sync_verified=True,
        loss_is_global=True,
        checkpoint_fingerprint_match=False,
        checkpoint_fingerprint_match_required=True,
        process_count=4,
    )

    assert readiness.distributed_training_ready is False
    assert "checkpoint_fingerprint_match" in readiness.missing_predicates
