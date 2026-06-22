from __future__ import annotations

import json
from dataclasses import replace
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
    _retention_report,
    deterministic_record_order,
    record_order_sha256,
    run_exemplar_pass,
    validate_corridor_checkpoint_lineage,
)
from qrwkv_xla.fingerprint.provenance import (
    file_sha256,
    hash_checkpoint_bundle,
    parameter_fingerprint,
)

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "behavioral_fingerprint"
    / "v0_1_with_exemplars_tiny"
)
NO_EXEMPLARS_FIXTURE = (
    Path(__file__).parent / "fixtures" / "behavioral_fingerprint" / "v0_1_valid_tiny"
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
    assert report["exemplar_evaluation_split"] == "held_out"
    assert report["training_exemplar_fallback_used"] is False
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


def test_no_held_out_artifact_labels_training_evaluation(tmp_path: Path) -> None:
    checkpoint = _parent_checkpoint(tmp_path)
    result = run_exemplar_pass(
        replace(
            _config(tmp_path, checkpoint),
            output_dir=tmp_path / "training-eval",
            optimizer="sgd",
            learning_rate=1e-2,
        )
    )
    receipt = _json(result.output_dir / "held_out_evaluation_receipt.json")
    assert receipt["exemplar_evaluation_split"] == "training"
    assert receipt["held_out_artifact_supplied"] is False
    assert receipt["training_exemplar_fallback_used"] is False


def test_supplied_held_out_without_exemplars_fails_closed(tmp_path: Path) -> None:
    checkpoint = _parent_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="held_out_exemplar_reservoir_missing"):
        run_exemplar_pass(
            replace(
                _config(tmp_path, checkpoint),
                output_dir=tmp_path / "missing-held-out",
                held_out_fingerprint_artifact=NO_EXEMPLARS_FIXTURE,
            )
        )


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"seed": 9}, "resume_sampling_seed_mismatch"),
        ({"exemplar_max_records": 2}, "resume_record_limit_mismatch"),
        ({"batch_size": 2}, "resume_batch_size_mismatch"),
        ({"optimizer": "sgd"}, "resume_optimizer_mismatch"),
        ({"learning_rate": 2e-2}, "resume_learning_rate_mismatch"),
        ({"max_grad_norm": 0.5}, "resume_max_grad_norm_mismatch"),
    ],
)
def test_resume_configuration_mismatch_fails_closed(
    tmp_path: Path, changes: dict, error: str
) -> None:
    checkpoint = _parent_checkpoint(tmp_path)
    first = run_exemplar_pass(
        ExemplarPassConfig(
            corridor_checkpoint=checkpoint,
            fingerprint_artifact=FIXTURE,
            output_dir=tmp_path / "first",
            student_backend="tiny_debug",
            steps=1,
            optimizer="adamw",
            learning_rate=1e-2,
            max_grad_norm=1.0,
        )
    )
    resumed = ExemplarPassConfig(
        corridor_checkpoint=checkpoint,
        fingerprint_artifact=FIXTURE,
        output_dir=tmp_path / "resumed",
        student_backend="tiny_debug",
        steps=2,
        optimizer="adamw",
        learning_rate=1e-2,
        max_grad_norm=1.0,
        resume_checkpoint=first.final_checkpoint,
    )
    with pytest.raises(ValueError, match=error):
        run_exemplar_pass(replace(resumed, **changes))


def test_resume_record_order_mismatch_fails_closed(tmp_path: Path) -> None:
    checkpoint = _parent_checkpoint(tmp_path)
    first = run_exemplar_pass(
        replace(
            _config(tmp_path, checkpoint),
            output_dir=tmp_path / "first",
            optimizer="sgd",
            learning_rate=1e-2,
        )
    )
    manifest_path = first.final_checkpoint / "checkpoint.json"
    manifest = _json(manifest_path)
    manifest["loss_config"]["record_order_sha256"] = "sha256:wrong"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="resume_record_order_mismatch"):
        run_exemplar_pass(
            replace(
                _config(tmp_path, checkpoint),
                output_dir=tmp_path / "resumed",
                optimizer="sgd",
                learning_rate=1e-2,
                steps=2,
                resume_checkpoint=first.final_checkpoint,
            )
        )


def test_resume_artifact_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    checkpoint = _parent_checkpoint(tmp_path)
    first = run_exemplar_pass(
        replace(
            _config(tmp_path, checkpoint),
            output_dir=tmp_path / "first",
            optimizer="sgd",
            learning_rate=1e-2,
        )
    )
    manifest_path = first.final_checkpoint / "checkpoint.json"
    manifest = _json(manifest_path)
    manifest["target_manifest"]["exemplar_artifact_sha256"] = "sha256:wrong"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="resume_artifact_mismatch"):
        run_exemplar_pass(
            replace(
                _config(tmp_path, checkpoint),
                output_dir=tmp_path / "resumed",
                optimizer="sgd",
                learning_rate=1e-2,
                steps=2,
                resume_checkpoint=first.final_checkpoint,
            )
        )


def test_resume_parent_corridor_mismatch_fails_closed(tmp_path: Path) -> None:
    first_parent = _parent_checkpoint(tmp_path / "first-parent")
    second_parent = _parent_checkpoint(
        tmp_path / "second-parent", parent_learning_rate=2e-3
    )
    first = run_exemplar_pass(
        replace(
            _config(tmp_path, first_parent),
            output_dir=tmp_path / "first",
            optimizer="sgd",
            learning_rate=1e-2,
        )
    )
    with pytest.raises(ValueError, match="resume_parent_corridor_mismatch"):
        run_exemplar_pass(
            replace(
                _config(tmp_path, second_parent),
                output_dir=tmp_path / "resumed",
                optimizer="sgd",
                learning_rate=1e-2,
                steps=2,
                resume_checkpoint=first.final_checkpoint,
            )
        )


def test_calibration_receipt_binds_exact_parent(tmp_path: Path) -> None:
    checkpoint = _parent_checkpoint(tmp_path, selected_profile="ball_peen")
    loaded = load_checkpoint(checkpoint)
    receipt = tmp_path / "selection.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "pass",
                "selection_allowed": True,
                "winner_declared": True,
                "selected_profile": "ball_peen",
                "selected_profile_config_sha256": "sha256:config",
                "selected_corridor_checkpoint_bundle_sha256": hash_checkpoint_bundle(
                    checkpoint
                )["checkpoint_bundle_sha256"],
                "selected_corridor_parameter_fingerprint": parameter_fingerprint(
                    loaded.params
                ),
            }
        ),
        encoding="utf-8",
    )
    _, lineage = validate_corridor_checkpoint_lineage(
        replace(_config(tmp_path, checkpoint), selected_profile=receipt),
        artifact_vocab_size=16,
    )
    assert lineage["calibration_parent_binding_valid"] is True
    assert lineage["selected_profile_name"] == "ball_peen"


def test_calibration_receipt_parent_mismatch_fails(tmp_path: Path) -> None:
    checkpoint = _parent_checkpoint(tmp_path, selected_profile="ball_peen")
    receipt = tmp_path / "selection.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "pass",
                "selection_allowed": True,
                "winner_declared": True,
                "selected_profile": "ball_peen",
                "selected_profile_config_sha256": "sha256:config",
                "selected_corridor_checkpoint_bundle_sha256": "sha256:wrong",
                "selected_corridor_parameter_fingerprint": "sha256:wrong",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="calibration_parent_lineage_mismatch"):
        validate_corridor_checkpoint_lineage(
            replace(_config(tmp_path, checkpoint), selected_profile=receipt),
            artifact_vocab_size=16,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        {"selection_allowed": False},
        {"winner_declared": False},
        {"selected_profile": None},
        {"selected_profile": "rock_hammer"},
    ],
)
def test_invalid_calibration_selection_fails_closed(
    tmp_path: Path, mutation: dict
) -> None:
    checkpoint = _parent_checkpoint(tmp_path, selected_profile="ball_peen")
    loaded = load_checkpoint(checkpoint)
    payload = {
        "status": "pass",
        "selection_allowed": True,
        "winner_declared": True,
        "selected_profile": "ball_peen",
        "selected_profile_config_sha256": "sha256:config",
        "selected_corridor_checkpoint_bundle_sha256": hash_checkpoint_bundle(
            checkpoint
        )["checkpoint_bundle_sha256"],
        "selected_corridor_parameter_fingerprint": parameter_fingerprint(loaded.params),
    }
    payload.update(mutation)
    receipt = tmp_path / "invalid-selection.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration_parent_lineage_mismatch"):
        validate_corridor_checkpoint_lineage(
            replace(_config(tmp_path, checkpoint), selected_profile=receipt),
            artifact_vocab_size=16,
        )


def test_unrelated_p153_report_fails_binding(tmp_path: Path) -> None:
    checkpoint = _parent_checkpoint(tmp_path)
    report = tmp_path / "p153.json"
    report.write_text(
        json.dumps(
            {
                "status": "pass",
                "training_cycle": "corridor_only",
                "completed_steps": 3,
                "checkpoint": {
                    "checkpoint_bundle_sha256": "sha256:unrelated",
                    "final_parameter_fingerprint": "sha256:unrelated",
                },
                "lineage": {
                    "artifact_manifest_sha256": file_sha256(FIXTURE / "manifest.json")
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="p153_parent_binding_valid"):
        validate_corridor_checkpoint_lineage(
            replace(_config(tmp_path, checkpoint), p153_report=report),
            artifact_vocab_size=16,
        )


@pytest.mark.parametrize(
    ("rates", "degraded", "exited", "exit_step"),
    [
        ((0.50, 0.40), True, False, None),
        ((0.97, 0.90), True, True, 1),
        ((0.90, 0.97), False, False, None),
    ],
)
def test_corridor_degradation_and_exit_are_distinct(
    rates: tuple[float, float], degraded: bool, exited: bool, exit_step: int | None
) -> None:
    trajectory = [
        _corridor_point(0, rates[0]),
        _corridor_point(1, rates[1]),
    ]
    report = _retention_report(trajectory, threshold=0.95)
    assert report["corridor_retention_degraded"] is degraded
    assert report["corridor_exit_detected"] is exited
    assert report["first_corridor_exit_step"] == exit_step


def _config(tmp_path: Path, checkpoint: Path) -> ExemplarPassConfig:
    return ExemplarPassConfig(
        corridor_checkpoint=checkpoint,
        fingerprint_artifact=FIXTURE,
        output_dir=tmp_path / "output",
        student_backend="tiny_debug",
        steps=1,
    )


def _parent_checkpoint(
    tmp_path: Path,
    *,
    loss_kind: str = "fingerprint_corridor",
    selected_profile: str | None = None,
    parent_learning_rate: float = 1e-3,
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
        learning_rate=parent_learning_rate,
        loss_config={"kind": loss_kind, "cycle": 1},
        target_manifest={
            "artifact_manifest_sha256": file_sha256(FIXTURE / "manifest.json"),
            "selected_aggressiveness_profile": selected_profile,
            "selected_profile_config_sha256": (
                "sha256:config" if selected_profile is not None else None
            ),
        },
        optimizer_config={"type": "sgd", "learning_rate": parent_learning_rate},
    )
    return path


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _corridor_point(step: int, inside_rate: float) -> dict:
    return {
        "optimizer_step": step,
        "corridor_metrics": {
            "inside_all_rate": inside_rate,
            "mean_distance_outside_corridor": 1.0 - inside_rate,
            "corridor_loss": (1.0 - inside_rate) ** 2,
        },
    }
