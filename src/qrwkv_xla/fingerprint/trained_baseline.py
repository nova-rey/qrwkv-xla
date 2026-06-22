from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qrwkv_xla.artifacts import load_fingerprint_targets, summarize_fingerprint_artifact
from qrwkv_xla.artifacts._json import read_json_object, write_json
from qrwkv_xla.checkpointing import load_checkpoint, save_checkpoint
from qrwkv_xla.contracts import VocabContract
from qrwkv_xla.fingerprint.training_rehearsal import (
    RealTeacherFingerprintTrainingRehearsalConfig,
    run_real_teacher_fingerprint_training_rehearsal,
)
from qrwkv_xla.optimizers import OptimizerConfig, init_optimizer_state, optimizer_update
from qrwkv_xla.students import create_student_backend


@dataclass(frozen=True)
class FingerprintTrainedBaselineConfig:
    fingerprint_artifact: Path
    source_texts: Path
    output_dir: Path
    steps: int = 3
    batch_size: int = 2
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    seed: int = 0
    student_backend: str = "current_qrwkv"
    overwrite: bool = False


@dataclass(frozen=True)
class FingerprintTrainedBaselineResult:
    status: str
    output_dir: Path
    report_path: Path
    metrics_path: Path
    summary_path: Path


@dataclass(frozen=True)
class _SourceExample:
    example_id: str
    input_ids: np.ndarray
    attention_mask: np.ndarray


def masked_causal_lm_loss(
    logits: jax.Array,
    input_ids: jax.Array,
    attention_mask: jax.Array,
) -> jax.Array:
    """Mean next-token cross entropy over non-padding target positions."""
    target_logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    target_mask = attention_mask[:, 1:].astype(target_logits.dtype)
    token_loss = -jax.nn.log_softmax(target_logits, axis=-1)
    token_loss = jnp.take_along_axis(token_loss, labels[..., None], axis=-1)[..., 0]
    denominator = jnp.maximum(jnp.sum(target_mask), 1.0)
    return jnp.sum(token_loss * target_mask) / denominator


def parameter_fingerprint(params: Any) -> str:
    digest = hashlib.sha256()
    leaves, tree = jax.tree_util.tree_flatten(params)
    digest.update(str(tree).encode("utf-8"))
    for leaf in leaves:
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return f"sha256:{digest.hexdigest()}"


def run_fingerprint_trained_baseline_comparison(
    config: FingerprintTrainedBaselineConfig,
) -> FingerprintTrainedBaselineResult:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = summarize_fingerprint_artifact(config.fingerprint_artifact)
    dataset = load_fingerprint_targets(config.fingerprint_artifact, batch_size=1)
    examples = _unique_examples(tuple(dataset.iter_records()))
    source_ids = _validate_source_alignment(config.source_texts, examples)
    backend, student_config = _create_backend(config, artifact)
    shared_params = backend.init_params(jax.random.PRNGKey(config.seed))
    initial_fingerprint = parameter_fingerprint(shared_params)
    shared_checkpoint = config.output_dir / "shared" / "checkpoints" / "initial"
    optimizer_config = OptimizerConfig(
        type=config.optimizer,
        learning_rate=config.learning_rate,
    )
    save_checkpoint(
        shared_checkpoint,
        shared_params,
        student_architecture=config.student_backend,
        student_config=student_config,
        step=0,
        learning_rate=config.learning_rate,
        loss_config={"kind": "shared_initialization"},
        target_manifest={"artifact_dir": str(config.fingerprint_artifact)},
        optimizer_config=asdict(optimizer_config),
        optimizer_state=init_optimizer_state(shared_params, optimizer_config),
        notes=["P151 shared initialization for both trained arms"],
        overwrite=config.overwrite,
    )
    loaded_initial = load_checkpoint(shared_checkpoint)
    if parameter_fingerprint(loaded_initial.params) != initial_fingerprint:
        raise ValueError("shared initialization checkpoint changed parameter bytes")

    baseline = _run_baseline(
        config,
        backend=backend,
        student_config=student_config,
        initial_params=loaded_initial.params,
        optimizer_config=optimizer_config,
        examples=examples,
    )
    fingerprint = _run_fingerprint(
        config,
        shared_checkpoint=shared_checkpoint,
        initial_fingerprint=initial_fingerprint,
        source_ids=source_ids,
    )
    fairness = _fairness_contract(
        config,
        baseline=baseline,
        fingerprint=fingerprint,
        initial_fingerprint=initial_fingerprint,
        source_ids=source_ids,
        artifact=artifact,
    )
    claims = {
        "trained_baseline_available": baseline["status"] == "pass",
        "comparison_valid": fairness["comparison_valid"],
        "winner_declared": False,
        "general_quality_claim_made": False,
        "quality_per_byte_claim_made": False,
        "held_out_evaluation_available": False,
        "held_out_claim_made": False,
        "radlads_parity_claim_made": False,
        "scale_claim_made": False,
    }
    status = "pass" if fairness["comparison_valid"] else "fail"
    report = {
        "phase": "P151",
        "run_kind": "trained_baseline_vs_fingerprint_corridor",
        "status": status,
        "comparison_kind": "trained_baseline_vs_fingerprint_corridor",
        "artifact_dir": str(config.fingerprint_artifact),
        "source_texts": str(config.source_texts),
        "initial_parameter_fingerprint": initial_fingerprint,
        "baseline_final_parameter_fingerprint": baseline["final_parameter_fingerprint"],
        "fingerprint_final_parameter_fingerprint": fingerprint[
            "final_parameter_fingerprint"
        ],
        "baseline_param_delta_norm": baseline["param_delta_norm"],
        "fingerprint_param_delta_norm": fingerprint["param_delta_norm"],
        "fairness": fairness,
        "baseline": baseline,
        "fingerprint": fingerprint,
        "claims": claims,
        "limitations": [
            "The two arms optimize different training objectives; raw losses are "
            "not directly comparable.",
            "No shared held-out evaluation is available in P151.",
            "P151 covers Cycle 1 only; exemplar training is disabled.",
        ],
    }
    report_path = config.output_dir / "trained_baseline_comparison_report.json"
    metrics_path = config.output_dir / "trained_baseline_metrics.json"
    summary_path = config.output_dir / "trained_baseline_summary.md"
    write_json(report_path, report)
    write_json(metrics_path, {"baseline": baseline, "fingerprint": fingerprint})
    summary_path.write_text(_render_summary(report), encoding="utf-8")
    return FingerprintTrainedBaselineResult(
        status=status,
        output_dir=config.output_dir,
        report_path=report_path,
        metrics_path=metrics_path,
        summary_path=summary_path,
    )


def _validate_config(config: FingerprintTrainedBaselineConfig) -> None:
    if config.steps <= 0:
        raise ValueError("steps must be > 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if config.optimizer not in {"sgd", "adam", "adamw"}:
        raise ValueError("optimizer must be one of {'sgd', 'adam', 'adamw'}")
    if config.output_dir.exists() and not config.overwrite:
        raise FileExistsError(
            f"output exists: {config.output_dir}; pass overwrite=True"
        )


def _unique_examples(records: tuple[Any, ...]) -> tuple[_SourceExample, ...]:
    examples: dict[str, tuple[np.ndarray, int]] = {}
    for record in records:
        candidate = np.asarray(record.input_ids, dtype=np.int32)
        prior, max_position = examples.setdefault(
            record.example_id,
            (candidate, int(record.position)),
        )
        if not np.array_equal(prior, candidate):
            raise ValueError(
                f"example_id {record.example_id!r} has inconsistent input_ids"
            )
        examples[record.example_id] = (
            prior,
            max(max_position, int(record.position)),
        )
    if not examples:
        raise ValueError("fingerprint artifact contains zero source examples")
    output = []
    for example_id, (input_ids, max_position) in examples.items():
        valid_length = min(len(input_ids), max_position + 1)
        attention_mask = np.zeros_like(input_ids, dtype=np.float32)
        attention_mask[:valid_length] = 1.0
        output.append(_SourceExample(example_id, input_ids, attention_mask))
    return tuple(output)


def _validate_source_alignment(
    source_texts: Path,
    examples: tuple[_SourceExample, ...],
) -> tuple[str, ...]:
    rows = [
        json.loads(line)
        for line in source_texts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("source_texts contains zero records")
    ids = tuple(
        str(row.get("example_id", f"p145-real-teacher-{index:06d}"))
        for index, row in enumerate(rows)
    )
    artifact_ids = tuple(example.example_id for example in examples)
    if ids[: len(artifact_ids)] != artifact_ids or len(ids) < len(artifact_ids):
        raise ValueError(
            f"source example IDs do not align: source={ids} artifact={artifact_ids}"
        )
    return artifact_ids


def _create_backend(
    config: FingerprintTrainedBaselineConfig, artifact: Any
) -> tuple[Any, dict[str, Any]]:
    contract = VocabContract(
        tokenizer_id=artifact.tokenizer_name or "fingerprint-artifact",
        tokenizer_hash=artifact.tokenizer_name or None,
        vocab_size=artifact.vocab_size,
        model_id=artifact.teacher_model_name or None,
        model_family="behavioral_fingerprint",
    )
    backend = create_student_backend(
        vocab_contract=contract, architecture_id=config.student_backend
    )
    raw = getattr(getattr(backend, "student", None), "config", None)
    student_config = {
        "architecture_id": config.student_backend,
        "backend_name": type(backend).__name__,
        "architecture": type(raw).__name__,
        "vocab_size": int(getattr(raw, "vocab_size", artifact.vocab_size)),
        "hidden_size": int(getattr(raw, "hidden_size", 0)),
        "num_layers": int(getattr(raw, "num_layers", 0)),
        "num_heads": _optional_int(getattr(raw, "num_heads", None)),
        "num_kv_heads": _optional_int(getattr(raw, "num_kv_heads", None)),
        "emit_logits": bool(getattr(raw, "emit_logits", True)),
        "tie_embeddings": bool(getattr(raw, "tie_embeddings", False)),
        "emit_mixer_outputs": bool(getattr(raw, "emit_mixer_outputs", False)),
    }
    return backend, student_config


def _run_baseline(
    config: FingerprintTrainedBaselineConfig,
    *,
    backend: Any,
    student_config: dict[str, Any],
    initial_params: Any,
    optimizer_config: OptimizerConfig,
    examples: tuple[_SourceExample, ...],
) -> dict[str, Any]:
    started = time.perf_counter()
    params = jax.tree_util.tree_map(lambda value: jnp.array(value), initial_params)
    initial_params_copy = jax.tree_util.tree_map(lambda value: jnp.array(value), params)
    state = init_optimizer_state(params, optimizer_config)
    losses: list[float] = []
    grad_norm = 0.0

    def loss_fn(
        current_params: Any, input_ids: jax.Array, mask: jax.Array
    ) -> jax.Array:
        output, _ = backend.forward_full(current_params, input_ids, attention_mask=mask)
        return masked_causal_lm_loss(backend.logits(output), input_ids, mask)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    batches = _example_batches(examples, config.batch_size)
    tokens_consumed = 0
    for step in range(config.steps):
        input_ids, mask = batches[step % len(batches)]
        loss, grads = value_and_grad(params, input_ids, mask)
        grad_norm = _tree_norm(grads)
        params, state, _ = optimizer_update(params, grads, state, optimizer_config)
        losses.append(float(loss))
        tokens_consumed += int(np.asarray(mask[:, 1:]).sum())
    jax.block_until_ready(params)
    delta_norm = _tree_delta_norm(initial_params_copy, params)
    checkpoint = config.output_dir / "baseline" / "checkpoints" / "final"
    save_checkpoint(
        checkpoint,
        params,
        student_architecture=config.student_backend,
        student_config=student_config,
        step=config.steps,
        learning_rate=config.learning_rate,
        loss_config={"kind": "causal_language_model", "padding_masked": True},
        target_manifest={"source_texts": str(config.source_texts)},
        optimizer_config=asdict(optimizer_config),
        optimizer_state=state,
        notes=["P151 conventional trained causal-LM baseline"],
        overwrite=config.overwrite,
    )
    metrics = {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_delta": losses[-1] - losses[0],
        "optimizer_steps_completed": config.steps,
        "batches_consumed": config.steps,
        "grad_global_norm": grad_norm,
        "param_delta_norm": delta_norm,
        "params_changed": delta_norm > 1e-12,
        "tokens_consumed": tokens_consumed,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    status = "pass" if _arm_passes(metrics, config.steps) else "fail"
    payload = {
        "arm_id": "trained_causal_lm_baseline",
        "status": status,
        "source_example_ids": [example.example_id for example in examples],
        "student_backend": config.student_backend,
        "student_config": student_config,
        "initialization_seed": config.seed,
        "optimizer": config.optimizer,
        "learning_rate_schedule": "constant",
        "requested_steps": config.steps,
        "batch_size": config.batch_size,
        "sequence_length": int(examples[0].input_ids.shape[0]),
        "initial_parameter_fingerprint": parameter_fingerprint(initial_params_copy),
        "final_parameter_fingerprint": parameter_fingerprint(params),
        "checkpoint_dir": str(checkpoint),
        "checkpoint_written": _checkpoint_written(checkpoint),
        **metrics,
    }
    arm_dir = config.output_dir / "baseline"
    write_json(arm_dir / "metrics.json", metrics)
    write_json(arm_dir / "run_report.json", payload)
    return payload


def _run_fingerprint(
    config: FingerprintTrainedBaselineConfig,
    *,
    shared_checkpoint: Path,
    initial_fingerprint: str,
    source_ids: tuple[str, ...],
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = config.output_dir / "fingerprint"
    result = run_real_teacher_fingerprint_training_rehearsal(
        RealTeacherFingerprintTrainingRehearsalConfig(
            output_dir=output_dir,
            fingerprint_artifact=config.fingerprint_artifact,
            training_steps=config.steps,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            optimizer=config.optimizer,
            seed=config.seed,
            student_backend=config.student_backend,
            resume_from=shared_checkpoint,
            overwrite=config.overwrite,
        )
    )
    report = read_json_object(result.report_path)
    training = report["training"]
    runner_checkpoint = Path(training["checkpoint_dir"])
    loaded_final = load_checkpoint(runner_checkpoint)
    final_params = loaded_final.params
    final_metrics = read_json_object(output_dir / "training" / "metrics.json")["final"]
    final_checkpoint = output_dir / "checkpoints" / "final"
    save_checkpoint(
        final_checkpoint,
        final_params,
        student_architecture=loaded_final.manifest.student_architecture,
        student_config=loaded_final.manifest.student_config,
        step=loaded_final.manifest.step,
        learning_rate=loaded_final.manifest.learning_rate,
        loss_config=loaded_final.manifest.loss_config,
        target_manifest=loaded_final.manifest.target_manifest,
        optimizer_config=loaded_final.manifest.optimizer_config,
        optimizer_state=loaded_final.optimizer_state,
        lr_schedule=loaded_final.manifest.lr_schedule,
        gradients=loaded_final.manifest.gradients,
        notes=[*loaded_final.manifest.notes, "P151 canonical fingerprint arm copy"],
        overwrite=config.overwrite,
    )
    write_json(output_dir / "metrics.json", final_metrics)
    write_json(
        output_dir / "fingerprint_corridor_report.json",
        read_json_object(Path(training["runner_report_path"])),
    )
    payload = {
        "arm_id": "fingerprint_corridor",
        "status": result.status,
        "source_example_ids": list(source_ids),
        "student_backend": config.student_backend,
        "student_config": training["student"],
        "initialization_seed": config.seed,
        "optimizer": config.optimizer,
        "learning_rate_schedule": "constant",
        "requested_steps": config.steps,
        "batch_size": config.batch_size,
        "sequence_length": int(report["capture"]["max_seq_len"]),
        "initial_parameter_fingerprint": initial_fingerprint,
        "final_parameter_fingerprint": parameter_fingerprint(final_params),
        "initial_corridor_loss": training["initial_loss"],
        "final_corridor_loss": training["final_loss"],
        "corridor_loss_delta": training["loss_delta"],
        "inside_all_rate": final_metrics["fingerprint/corridor/inside_all_rate"],
        "inside_entropy_rate": final_metrics[
            "fingerprint/corridor/inside_entropy_rate"
        ],
        "inside_top1_margin_rate": final_metrics[
            "fingerprint/corridor/inside_top1_margin_rate"
        ],
        "inside_top8_mass_rate": final_metrics[
            "fingerprint/corridor/inside_top8_mass_rate"
        ],
        "inside_top32_mass_rate": final_metrics[
            "fingerprint/corridor/inside_top32_mass_rate"
        ],
        "inside_tail_mass_rate": final_metrics[
            "fingerprint/corridor/inside_tail_mass_rate"
        ],
        "optimizer_steps_completed": training["optimizer_steps_completed"],
        "batches_consumed": training["batches_consumed"],
        "grad_global_norm": final_metrics["grad_global_norm"],
        "param_delta_norm": training["param_delta_norm"],
        "params_changed": training["params_changed"],
        "records_consumed": _records_consumed(
            config.fingerprint_artifact,
            batch_size=config.batch_size,
            steps=config.steps,
        ),
        "wall_clock_seconds": time.perf_counter() - started,
        "checkpoint_dir": str(final_checkpoint),
        "checkpoint_written": _checkpoint_written(final_checkpoint),
        "teacher_required_during_training": training[
            "teacher_required_during_training"
        ],
        "exemplar_training_enabled": False,
    }
    return payload


def _fairness_contract(
    config: FingerprintTrainedBaselineConfig,
    *,
    baseline: dict[str, Any],
    fingerprint: dict[str, Any],
    initial_fingerprint: str,
    source_ids: tuple[str, ...],
    artifact: Any,
) -> dict[str, Any]:
    contract = {
        "comparison_kind": "trained_baseline_vs_fingerprint_corridor",
        "same_student_architecture": baseline["student_config"]
        == fingerprint["student_config"],
        "same_student_backend": baseline["student_backend"]
        == fingerprint["student_backend"],
        "same_initialization_seed": baseline["initialization_seed"]
        == fingerprint["initialization_seed"],
        "same_initial_parameter_fingerprint": baseline["initial_parameter_fingerprint"]
        == fingerprint["initial_parameter_fingerprint"]
        == initial_fingerprint,
        "same_source_example_ids": baseline["source_example_ids"]
        == fingerprint["source_example_ids"]
        == list(source_ids),
        "tokenizer_vocab_compatible": baseline["student_config"]["vocab_size"]
        == fingerprint["student_config"]["vocab_size"]
        == artifact.vocab_size,
        "deterministic_record_limit": baseline["source_example_ids"]
        == list(source_ids),
        "same_optimizer": baseline["optimizer"] == fingerprint["optimizer"],
        "same_learning_rate_schedule": baseline["learning_rate_schedule"]
        == fingerprint["learning_rate_schedule"],
        "same_requested_steps": baseline["requested_steps"]
        == fingerprint["requested_steps"],
        "same_completed_steps": baseline["optimizer_steps_completed"]
        == fingerprint["optimizer_steps_completed"]
        == config.steps,
        "same_batch_size": baseline["batch_size"] == fingerprint["batch_size"],
        "same_sequence_length": baseline["sequence_length"]
        == fingerprint["sequence_length"]
        == artifact.max_seq_len,
        "teacher_required_during_training": fingerprint[
            "teacher_required_during_training"
        ],
        "exemplar_training_enabled": fingerprint["exemplar_training_enabled"],
        "comparison_fairness": "matched_training_budget",
        "baseline_checkpoint_written": baseline["checkpoint_written"],
        "fingerprint_checkpoint_written": fingerprint["checkpoint_written"],
    }
    required = tuple(
        key
        for key, value in contract.items()
        if isinstance(value, bool)
        and key not in {"teacher_required_during_training", "exemplar_training_enabled"}
    )
    contract["comparison_valid"] = (
        all(contract[key] for key in required)
        and not contract["teacher_required_during_training"]
        and not contract["exemplar_training_enabled"]
        and baseline["status"] == fingerprint["status"] == "pass"
    )
    return contract


def _example_batches(
    examples: tuple[_SourceExample, ...], batch_size: int
) -> tuple[tuple[jax.Array, jax.Array], ...]:
    batches = []
    for start in range(0, len(examples), batch_size):
        rows = examples[start : start + batch_size]
        ids = jnp.asarray(np.stack([item.input_ids for item in rows]), dtype=jnp.int32)
        mask = jnp.asarray(
            np.stack([item.attention_mask for item in rows]), dtype=jnp.float32
        )
        batches.append((ids, mask))
    return tuple(batches)


def _records_consumed(artifact: Path, *, batch_size: int, steps: int) -> int:
    batches = tuple(
        load_fingerprint_targets(artifact, batch_size=batch_size).iter_batches()
    )
    return sum(len(batches[step % len(batches)].input_ids) for step in range(steps))


def _tree_norm(tree: Any) -> float:
    return float(
        jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(tree)))
    )


def _tree_delta_norm(before: Any, after: Any) -> float:
    return float(
        jnp.sqrt(
            sum(
                jnp.sum(jnp.square(b - a))
                for a, b in zip(
                    jax.tree_util.tree_leaves(before),
                    jax.tree_util.tree_leaves(after),
                    strict=True,
                )
            )
        )
    )


def _arm_passes(metrics: dict[str, Any], steps: int) -> bool:
    return bool(
        metrics["optimizer_steps_completed"] == steps
        and metrics["params_changed"]
        and all(
            math.isfinite(float(metrics[key]))
            for key in (
                "initial_loss",
                "final_loss",
                "grad_global_norm",
                "param_delta_norm",
            )
        )
    )


def _checkpoint_written(path: Path) -> bool:
    return (path / "checkpoint.json").is_file() and (path / "params.npz").is_file()


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _render_summary(report: dict[str, Any]) -> str:
    return "\n".join(
        (
            "# P151 Trained Baseline Comparison",
            "",
            f"- Status: {report['status']}",
            f"- Fairness: {report['fairness']['comparison_fairness']}",
            "- Comparison valid: "
            f"{str(report['fairness']['comparison_valid']).lower()}",
            f"- Baseline steps: {report['baseline']['optimizer_steps_completed']}",
            "- Fingerprint steps: "
            f"{report['fingerprint']['optimizer_steps_completed']}",
            "- Winner declared: false",
            "- Held-out evaluation available: false",
            "",
        )
    )
