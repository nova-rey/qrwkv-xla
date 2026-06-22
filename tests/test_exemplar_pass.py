from __future__ import annotations

import json
from pathlib import Path

import jax
import pytest

from qrwkv_xla.artifacts import (
    load_fingerprint_exemplars,
    summarize_fingerprint_artifact,
)
from qrwkv_xla.checkpointing import load_checkpoint, save_checkpoint
from qrwkv_xla.fingerprint.exemplar_pass import (
    ExemplarPassConfig,
    _create_backend,
    deterministic_record_order,
    record_order_sha256,
    run_exemplar_pass,
    validate_corridor_checkpoint_lineage,
)
from qrwkv_xla.fingerprint.provenance import file_sha256, parameter_fingerprint

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "behavioral_fingerprint"
    / "v0_1_with_exemplars_tiny"
)


def test_sequential_sampling_and_order_hash_are_deterministic() -> None:
    records = tuple(load_fingerprint_exemplars(FIXTURE).iter_records())
    first = deterministic_record_order(records, "sequential", 0)
    second = deterministic_record_order(records, "sequential", 99)
    assert first == second == records
    assert record_order_sha256(first) == record_order_sha256(second)


def test_non_corridor_parent_checkpoint_fails_closed(tmp_path: Path) -> None:
    checkpoint = _parent_checkpoint(tmp_path, loss_kind="fingerprint_exemplar")
    config = _config(tmp_path, checkpoint)
    with pytest.raises(ValueError, match="distill_mode_is_fingerprint_corridor"):
        validate_corridor_checkpoint_lineage(config, artifact_vocab_size=16)


def test_vocab_mismatch_fails_closed(tmp_path: Path) -> None:
    checkpoint = _parent_checkpoint(tmp_path)
    config = _config(tmp_path, checkpoint)
    with pytest.raises(ValueError, match="vocab_size_match"):
        validate_corridor_checkpoint_lineage(config, artifact_vocab_size=17)


def test_standalone_exemplar_pass_smoke(tmp_path: Path) -> None:
    checkpoint = _parent_checkpoint(tmp_path)
    parent = load_checkpoint(checkpoint)
    parent_fingerprint = parameter_fingerprint(parent.params)
    result = run_exemplar_pass(
        ExemplarPassConfig(
            corridor_checkpoint=checkpoint,
            fingerprint_artifact=FIXTURE,
            output_dir=tmp_path / "p154",
            student_backend="tiny_debug",
            steps=3,
            batch_size=2,
            optimizer="sgd",
            learning_rate=1e-2,
            eval_every=1,
            checkpoint_every=2,
            held_out_fingerprint_artifact=FIXTURE,
        )
    )
    report = _json(result.report_path)
    final = load_checkpoint(result.final_checkpoint)
    assert result.status == "pass"
    assert result.completed_steps == 3
    assert report["training_cycle"] == "exemplar"
    assert report["corridor_loss_enabled"] is False
    assert report["mixed_objective_enabled"] is False
    assert report["input_checkpoint_optimizer_state_loaded"] is False
    assert report["exemplar_optimizer_state_fresh"] is True
    assert report["exemplar_local_step_start"] == 0
    assert report["params_changed"] is True
    assert report["corridor_retention_evaluation_enabled"] is True
    assert report["initial_corridor_metrics"] is not None
    assert parameter_fingerprint(final.params) != parent_fingerprint
    assert (
        parameter_fingerprint(load_checkpoint(checkpoint).params) == parent_fingerprint
    )
    assert result.best_checkpoint.is_dir()
    assert result.final_checkpoint.is_dir()
    assert final.manifest.loss_config["kind"] == "fingerprint_exemplar"
    assert final.manifest.loss_config["cycle"] == 2
    for name in (
        "exemplar_pass_report.json",
        "exemplar_pass_summary.md",
        "exemplar_trajectory.jsonl",
        "exemplar_efficiency_metrics.json",
        "corridor_retention_report.json",
        "checkpoint_lineage_validation.json",
        "resource_accounting.json",
        "sampling_receipt.json",
        "resume_receipt.json",
    ):
        assert (result.output_dir / name).is_file()


def test_resume_restores_exemplar_optimizer_state(tmp_path: Path) -> None:
    checkpoint = _parent_checkpoint(tmp_path)
    first = run_exemplar_pass(
        ExemplarPassConfig(
            corridor_checkpoint=checkpoint,
            fingerprint_artifact=FIXTURE,
            output_dir=tmp_path / "first",
            student_backend="tiny_debug",
            steps=2,
            optimizer="adamw",
            learning_rate=1e-2,
            eval_every=1,
            checkpoint_every=1,
        )
    )
    resumed = run_exemplar_pass(
        ExemplarPassConfig(
            corridor_checkpoint=checkpoint,
            fingerprint_artifact=FIXTURE,
            output_dir=tmp_path / "resumed",
            student_backend="tiny_debug",
            steps=4,
            optimizer="adamw",
            learning_rate=1e-2,
            eval_every=1,
            checkpoint_every=1,
            resume_checkpoint=first.final_checkpoint,
        )
    )
    report = _json(resumed.report_path)
    receipt = _json(resumed.output_dir / "resume_receipt.json")
    assert resumed.completed_steps == 4
    assert report["exemplar_local_step_start"] == 2
    assert report["exemplar_optimizer_state_fresh"] is False
    assert receipt["resumed"] is True
    assert receipt["input_checkpoint_optimizer_state_loaded"] is True


def _config(tmp_path: Path, checkpoint: Path) -> ExemplarPassConfig:
    return ExemplarPassConfig(
        corridor_checkpoint=checkpoint,
        fingerprint_artifact=FIXTURE,
        output_dir=tmp_path / "output",
        student_backend="tiny_debug",
        steps=1,
    )


def _parent_checkpoint(
    tmp_path: Path, *, loss_kind: str = "fingerprint_corridor"
) -> Path:
    config = ExemplarPassConfig(
        corridor_checkpoint=tmp_path / "unused",
        fingerprint_artifact=FIXTURE,
        output_dir=tmp_path / "unused-output",
        student_backend="tiny_debug",
    )
    backend, student_config = _create_backend(
        config, summarize_fingerprint_artifact(FIXTURE)
    )
    path = tmp_path / "checkpoints" / "final"
    save_checkpoint(
        path,
        backend.init_params(jax.random.PRNGKey(0)),
        student_architecture="tiny_debug",
        student_config=student_config,
        step=3,
        learning_rate=1e-3,
        loss_config={"kind": loss_kind, "cycle": 1},
        target_manifest={
            "artifact_manifest_sha256": file_sha256(FIXTURE / "manifest.json")
        },
        optimizer_config={"type": "sgd", "learning_rate": 1e-3},
    )
    return path


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
