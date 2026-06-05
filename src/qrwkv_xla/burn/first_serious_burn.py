from __future__ import annotations

import json
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qrwkv_xla.burn.config import FirstSeriousBurnConfig
from qrwkv_xla.checkpointing import run_checkpoint_resume_update_rehearsal
from qrwkv_xla.eval import (
    create_builtin_mini_eval_store,
    run_mini_eval_harness,
    write_mini_eval_report,
)
from qrwkv_xla.readiness import (
    ReadinessStatus,
    build_big_burn_readiness_report,
    write_big_burn_readiness_report,
)
from qrwkv_xla.targets import TeacherTargetStore, load_offline_target_batch
from qrwkv_xla.xla import run_runtime_environment_preflight

FIRST_SERIOUS_BURN_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "training_success_guaranteed",
    "model_quality_proven",
    "production_training_ready",
    "large_scale_performance_proven",
    "pallas_default_ready",
    "qwen_specific_support",
    "tokenizer_remapping_supported",
    "distributed_training_ready",
    "automatic_burn_launched",
)


@dataclass(frozen=True)
class FirstSeriousBurnResult:
    phase: str
    status: str
    mode: str
    dry_run: bool
    readiness_status: str | None
    steps_requested: int
    steps_completed: int
    output_dir: str
    readiness_report_path: str | None
    preflight_report_path: str | None
    checkpoint_path: str | None
    eval_report_path: str | None
    launch_commands_path: str | None
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    teacher_textbook_path: str | None = None
    teacher_target_type: str | None = None
    teacher_textbook_examples: int = 0
    teacher_textbook_sequence_length: int = 0
    teacher_textbook_vocab_size: int = 0
    max_steps_requested: int = 0
    batch_size: int = 1
    allow_textbook_reuse: bool = False
    examples_available: int = 0
    examples_consumed: int = 0
    unique_examples_consumed: int = 0
    reuse_count: int = 0
    epochs_completed_or_fractional: float = 0.0
    loss_initial: float | None = None
    loss_final: float | None = None
    loss_delta: float | None = None
    loss_trace_path: str | None = None
    checkpoint_written: bool = False
    checkpoint_nonzero: bool = False
    device_backend: str | None = None
    jax_devices: tuple[str, ...] = ()
    worker_id: str | None = None
    hostname: str | None = None
    real_training_executed: bool = False
    claims_not_made: tuple[str, ...] = FIRST_SERIOUS_BURN_CLAIMS_NOT_MADE

    def to_report(self) -> dict[str, Any]:
        return asdict(self)


def run_first_serious_burn(
    config: FirstSeriousBurnConfig,
    *,
    confirm_serious_burn: bool = False,
) -> FirstSeriousBurnResult:
    _validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    readiness = _resolve_readiness(config, output_dir=output_dir)
    launch_commands_path = write_launch_commands(output_dir / "launch_commands.md")
    readiness_blockers, readiness_warnings = _readiness_gate(config, readiness)
    if readiness_blockers:
        return _blocked_result(
            config=config,
            output_dir=output_dir,
            readiness_status=readiness.status,
            readiness_report_path=readiness.path,
            launch_commands_path=launch_commands_path,
            blockers=readiness_blockers,
            warnings=readiness_warnings,
        )

    if config.mode == "real":
        if not confirm_serious_burn:
            return _blocked_result(
                config=config,
                output_dir=output_dir,
                readiness_status=readiness.status,
                readiness_report_path=readiness.path,
                launch_commands_path=launch_commands_path,
                blockers=("real burn mode requires --confirm-serious-burn",),
                warnings=readiness_warnings,
            )
        return _run_real_training(
            config=config,
            output_dir=output_dir,
            readiness=readiness,
            launch_commands_path=launch_commands_path,
            warnings=readiness_warnings,
        )

    return _run_dry_run(
        config=config,
        output_dir=output_dir,
        readiness=readiness,
        launch_commands_path=launch_commands_path,
        warnings=readiness_warnings,
    )


def write_first_serious_burn_report(
    result: FirstSeriousBurnResult,
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_report(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_launch_commands(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_launch_commands_markdown(), encoding="utf-8")
    return output_path


@dataclass(frozen=True)
class _ReadinessGate:
    status: str | None
    path: Path | None
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]


def _run_dry_run(
    *,
    config: FirstSeriousBurnConfig,
    output_dir: Path,
    readiness: _ReadinessGate,
    launch_commands_path: Path,
    warnings: tuple[str, ...],
) -> FirstSeriousBurnResult:
    hugepage_path = output_dir / "transparent_hugepage_enabled"
    hugepage_path.write_text("[always] madvise never\n", encoding="utf-8")
    preflight = run_runtime_environment_preflight(
        hugepage_path=hugepage_path,
        require_tpu=False,
        enable_hugepages=False,
    )
    preflight_path = output_dir / "preflight_report.json"
    preflight_path.write_text(
        json.dumps(preflight.to_report(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checkpoint = run_checkpoint_resume_update_rehearsal(
        output_dir=output_dir / "checkpoint_rehearsal",
        steps_before_checkpoint=1,
        steps_after_resume=1,
    )
    checkpoint_path = Path(checkpoint.checkpoint_path)

    store = create_builtin_mini_eval_store(output_dir / "mini_eval_target_store")
    eval_result = run_mini_eval_harness(
        store=store,
        architecture_id=config.architecture_id,
        runtime=config.runtime,
    )
    eval_report_path = write_mini_eval_report(
        eval_result,
        output_dir / "mini_eval_report.json",
    )

    blockers = _dry_run_blockers(
        preflight_status=preflight.status,
        checkpoint_status=checkpoint.status,
        eval_status=eval_result.status,
    )
    return FirstSeriousBurnResult(
        phase=config.phase,
        status="blocked" if blockers else "dry_run_pass",
        mode=config.mode,
        dry_run=True,
        readiness_status=readiness.status,
        steps_requested=config.max_steps,
        steps_completed=0 if blockers else min(config.max_steps, 1),
        output_dir=str(output_dir),
        readiness_report_path=str(readiness.path) if readiness.path else None,
        preflight_report_path=str(preflight_path),
        checkpoint_path=str(checkpoint_path),
        eval_report_path=str(eval_report_path),
        launch_commands_path=str(launch_commands_path),
        blockers=blockers,
        warnings=warnings,
        max_steps_requested=config.max_steps,
        batch_size=config.batch_size,
        allow_textbook_reuse=config.allow_textbook_reuse,
        device_backend=_jax_backend(),
        jax_devices=_jax_devices(),
        worker_id="0",
        hostname=socket.gethostname(),
    )


@dataclass(frozen=True)
class _DenseTextbookArrays:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    teacher_logits: np.ndarray


@dataclass(frozen=True)
class _TrainingRun:
    steps_completed: int
    examples_available: int
    examples_consumed: int
    unique_examples_consumed: int
    reuse_count: int
    epochs_completed_or_fractional: float
    loss_trace: tuple[float, ...]
    checkpoint_path: Path
    loss_trace_path: Path
    checkpoint_written: bool
    checkpoint_nonzero: bool
    params_changed: bool


def _run_real_training(
    *,
    config: FirstSeriousBurnConfig,
    output_dir: Path,
    readiness: _ReadinessGate,
    launch_commands_path: Path,
    warnings: tuple[str, ...],
) -> FirstSeriousBurnResult:
    textbook_path = _teacher_textbook_path(config)
    if textbook_path is None:
        return _blocked_result(
            config=config,
            output_dir=output_dir,
            readiness_status=readiness.status,
            readiness_report_path=readiness.path,
            launch_commands_path=launch_commands_path,
            blockers=("real mode requires --teacher-textbook or target_store_path",),
            warnings=warnings,
        )
    store = TeacherTargetStore.open(textbook_path)
    store.validate()
    arrays = _load_dense_textbook_arrays(store)
    exhaustion_blocker = _exhaustion_blocker(
        examples_available=arrays.input_ids.shape[0],
        batch_size=config.batch_size,
        max_steps=config.max_steps,
        allow_reuse=config.allow_textbook_reuse,
    )
    if exhaustion_blocker is not None:
        return _blocked_result(
            config=config,
            output_dir=output_dir,
            readiness_status=readiness.status,
            readiness_report_path=readiness.path,
            launch_commands_path=launch_commands_path,
            blockers=(exhaustion_blocker,),
            warnings=warnings,
            teacher_textbook_path=str(textbook_path),
            teacher_target_type=store.metadata.target_type,
            teacher_textbook_examples=store.metadata.num_examples,
            teacher_textbook_sequence_length=store.metadata.sequence_length,
            teacher_textbook_vocab_size=store.metadata.vocab_size,
            examples_available=arrays.input_ids.shape[0],
        )

    training = _train_dense_textbook(
        arrays,
        output_dir=output_dir,
        config=config,
        teacher_textbook_path=textbook_path,
        target_type=store.metadata.target_type,
    )
    blockers = _real_training_blockers(training)
    loss_initial = training.loss_trace[0] if training.loss_trace else None
    loss_final = training.loss_trace[-1] if training.loss_trace else None
    return FirstSeriousBurnResult(
        phase="P117.1",
        status="fail" if blockers else "pass",
        mode=config.mode,
        dry_run=False,
        readiness_status=readiness.status,
        steps_requested=config.max_steps,
        steps_completed=training.steps_completed,
        output_dir=str(output_dir),
        readiness_report_path=str(readiness.path) if readiness.path else None,
        preflight_report_path=None,
        checkpoint_path=str(training.checkpoint_path),
        eval_report_path=None,
        launch_commands_path=str(launch_commands_path),
        blockers=blockers,
        warnings=warnings,
        teacher_textbook_path=str(textbook_path),
        teacher_target_type=store.metadata.target_type,
        teacher_textbook_examples=store.metadata.num_examples,
        teacher_textbook_sequence_length=store.metadata.sequence_length,
        teacher_textbook_vocab_size=store.metadata.vocab_size,
        max_steps_requested=config.max_steps,
        batch_size=config.batch_size,
        allow_textbook_reuse=config.allow_textbook_reuse,
        examples_available=training.examples_available,
        examples_consumed=training.examples_consumed,
        unique_examples_consumed=training.unique_examples_consumed,
        reuse_count=training.reuse_count,
        epochs_completed_or_fractional=training.epochs_completed_or_fractional,
        loss_initial=loss_initial,
        loss_final=loss_final,
        loss_delta=(
            None
            if loss_initial is None or loss_final is None
            else loss_final - loss_initial
        ),
        loss_trace_path=str(training.loss_trace_path),
        checkpoint_written=training.checkpoint_written,
        checkpoint_nonzero=training.checkpoint_nonzero,
        device_backend=_jax_backend(),
        jax_devices=_jax_devices(),
        worker_id="0",
        hostname=socket.gethostname(),
        real_training_executed=training.steps_completed > 0,
    )


def _resolve_readiness(
    config: FirstSeriousBurnConfig,
    *,
    output_dir: Path,
) -> _ReadinessGate:
    if config.readiness_report_path is not None:
        path = Path(config.readiness_report_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _ReadinessGate(
            status=payload.get("status"),
            path=path,
            blockers=tuple(payload.get("blockers", ())),
            warnings=tuple(payload.get("warnings", ())),
        )

    report = build_big_burn_readiness_report(work_dir=output_dir / "readiness")
    path = write_big_burn_readiness_report(
        report,
        output_dir / "readiness_report.json",
    )
    return _ReadinessGate(
        status=report.status.value,
        path=path,
        blockers=report.blockers,
        warnings=report.warnings,
    )


def _readiness_gate(
    config: FirstSeriousBurnConfig,
    readiness: _ReadinessGate,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    status = readiness.status
    blockers = tuple(readiness.blockers)
    warnings = tuple(readiness.warnings)
    if status == ReadinessStatus.FAIL.value:
        return (
            blockers or ("P111 readiness report status is fail",),
            warnings,
        )
    if status == ReadinessStatus.WARN.value:
        if config.require_readiness_pass and not _warnings_accepted(
            warnings,
            accepted=config.accepted_warnings,
        ):
            return (
                ("P111 readiness report status is warn and warnings were not accepted"),
            ), warnings
    if status != ReadinessStatus.PASS.value and status != ReadinessStatus.WARN.value:
        return ((f"unsupported readiness status: {status}",), warnings)
    return (), warnings


def _blocked_result(
    *,
    config: FirstSeriousBurnConfig,
    output_dir: Path,
    readiness_status: str | None,
    readiness_report_path: Path | None,
    launch_commands_path: Path,
    blockers: tuple[str, ...],
    warnings: tuple[str, ...],
    teacher_textbook_path: str | None = None,
    teacher_target_type: str | None = None,
    teacher_textbook_examples: int = 0,
    teacher_textbook_sequence_length: int = 0,
    teacher_textbook_vocab_size: int = 0,
    examples_available: int = 0,
) -> FirstSeriousBurnResult:
    return FirstSeriousBurnResult(
        phase="P112",
        status="blocked",
        mode=config.mode,
        dry_run=config.mode == "dry_run",
        readiness_status=readiness_status,
        steps_requested=config.max_steps,
        steps_completed=0,
        output_dir=str(output_dir),
        readiness_report_path=(
            str(readiness_report_path) if readiness_report_path else None
        ),
        preflight_report_path=None,
        checkpoint_path=None,
        eval_report_path=None,
        launch_commands_path=str(launch_commands_path),
        blockers=blockers,
        warnings=warnings,
        teacher_textbook_path=teacher_textbook_path,
        teacher_target_type=teacher_target_type,
        teacher_textbook_examples=teacher_textbook_examples,
        teacher_textbook_sequence_length=teacher_textbook_sequence_length,
        teacher_textbook_vocab_size=teacher_textbook_vocab_size,
        max_steps_requested=config.max_steps,
        batch_size=config.batch_size,
        allow_textbook_reuse=config.allow_textbook_reuse,
        examples_available=examples_available,
        device_backend=_jax_backend(),
        jax_devices=_jax_devices(),
        worker_id="0",
        hostname=socket.gethostname(),
    )


def _dry_run_blockers(
    *,
    preflight_status: str,
    checkpoint_status: str,
    eval_status: str,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if preflight_status == "fail":
        blockers.append("runtime environment preflight failed")
    if checkpoint_status != "pass":
        blockers.append(f"checkpoint/resume update status={checkpoint_status}")
    if eval_status != "pass":
        blockers.append(f"mini eval status={eval_status}")
    return tuple(blockers)


def _warnings_accepted(
    warnings: tuple[str, ...],
    *,
    accepted: tuple[str, ...],
) -> bool:
    if not warnings:
        return True
    if "*" in accepted:
        return True
    return set(warnings).issubset(set(accepted))


def _validate_config(config: FirstSeriousBurnConfig) -> None:
    if config.phase not in {"P112", "P117.1"}:
        raise ValueError(f"phase must be 'P112' or 'P117.1', got {config.phase!r}")
    if config.mode not in {"dry_run", "real"}:
        raise ValueError("mode must be 'dry_run' or 'real'")
    if config.max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if config.sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")
    if config.checkpoint_every_steps <= 0:
        raise ValueError("checkpoint_every_steps must be > 0")
    if config.eval_every_steps <= 0:
        raise ValueError("eval_every_steps must be > 0")
    if config.allow_downloads and config.local_files_only:
        raise ValueError("allow_downloads=True conflicts with local_files_only=True")


def _teacher_textbook_path(config: FirstSeriousBurnConfig) -> Path | None:
    raw_path = config.teacher_textbook_path or config.target_store_path
    return None if raw_path is None else Path(raw_path)


def _load_dense_textbook_arrays(store: TeacherTargetStore) -> _DenseTextbookArrays:
    if store.metadata.target_type not in {"full_logits", "synthetic"}:
        raise ValueError(
            "P117.1 real training supports dense TeacherTextbook target types "
            f"{{'full_logits', 'synthetic'}}, got {store.metadata.target_type!r}"
        )
    batches = [
        load_offline_target_batch(store, shard_id=shard_id)
        for shard_id in range(store.metadata.shard_count)
    ]
    return _DenseTextbookArrays(
        input_ids=np.concatenate([batch.input_ids for batch in batches], axis=0),
        attention_mask=np.concatenate(
            [batch.attention_mask for batch in batches],
            axis=0,
        ),
        teacher_logits=np.concatenate(
            [batch.teacher_logits for batch in batches],
            axis=0,
        ),
    )


def _exhaustion_blocker(
    *,
    examples_available: int,
    batch_size: int,
    max_steps: int,
    allow_reuse: bool,
) -> str | None:
    if allow_reuse:
        return None
    examples_required = max_steps * batch_size
    if examples_required <= examples_available:
        return None
    return (
        f"requested {max_steps} steps with batch_size {batch_size} requires "
        f"{examples_required} examples, but textbook has {examples_available} "
        "examples and allow_textbook_reuse=false"
    )


def _train_dense_textbook(
    arrays: _DenseTextbookArrays,
    *,
    output_dir: Path,
    config: FirstSeriousBurnConfig,
    teacher_textbook_path: Path,
    target_type: str,
) -> _TrainingRun:
    input_ids = jnp.asarray(arrays.input_ids, dtype=jnp.int32)
    attention_mask = jnp.asarray(arrays.attention_mask, dtype=jnp.float32)
    teacher_logits = jnp.asarray(arrays.teacher_logits, dtype=jnp.float32)
    vocab_size = int(teacher_logits.shape[-1])
    params = {
        "vocab_bias": jnp.zeros((vocab_size,), dtype=jnp.float32),
        "token_scale": jnp.asarray(0.01, dtype=jnp.float32),
    }
    initial_params = params
    loss_and_grad = jax.value_and_grad(_dense_distill_loss)
    loss_trace: list[float] = []
    seen: list[int] = []

    for step in range(config.max_steps):
        indices = _batch_indices(
            step=step,
            batch_size=config.batch_size,
            examples_available=arrays.input_ids.shape[0],
            allow_reuse=config.allow_textbook_reuse,
        )
        batch_input_ids = input_ids[jnp.asarray(indices)]
        batch_attention_mask = attention_mask[jnp.asarray(indices)]
        batch_teacher_logits = teacher_logits[jnp.asarray(indices)]
        loss, grads = loss_and_grad(
            params,
            batch_input_ids,
            batch_attention_mask,
            batch_teacher_logits,
        )
        params = jax.tree_util.tree_map(
            lambda param, grad: param - jnp.asarray(0.05, dtype=jnp.float32) * grad,
            params,
            grads,
        )
        loss_trace.append(float(loss))
        seen.extend(indices)

    output_dir.mkdir(parents=True, exist_ok=True)
    loss_trace_path = output_dir / "loss_trace.json"
    loss_trace_path.write_text(
        json.dumps(
            [{"step": step + 1, "loss": loss} for step, loss in enumerate(loss_trace)],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_path = output_dir / "checkpoint.json"
    params_changed = _params_changed(initial_params, params)
    checkpoint_payload = {
        "step": len(loss_trace),
        "model_config": {
            "kind": "tiny_dense_logit_bias_student",
            "vocab_size": vocab_size,
            "sequence_length": int(input_ids.shape[1]),
        },
        "optimizer_config": {"type": "sgd", "learning_rate": 0.05},
        "param_summary": {
            "vocab_bias_mean": float(jnp.mean(params["vocab_bias"])),
            "vocab_bias_std": float(jnp.std(params["vocab_bias"])),
            "vocab_bias_first8": [
                float(value) for value in np.asarray(params["vocab_bias"][:8])
            ],
            "token_scale": float(params["token_scale"]),
            "params_changed": params_changed,
        },
        "rng_state": None,
        "teacher_textbook_path": str(teacher_textbook_path),
        "teacher_target_type": target_type,
        "batch_size": config.batch_size,
        "max_steps": config.max_steps,
        "allow_textbook_reuse": config.allow_textbook_reuse,
        "loss_trace": loss_trace,
    }
    checkpoint_path.write_text(
        json.dumps(checkpoint_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    unique_seen = len(set(seen))
    return _TrainingRun(
        steps_completed=len(loss_trace),
        examples_available=arrays.input_ids.shape[0],
        examples_consumed=len(seen),
        unique_examples_consumed=unique_seen,
        reuse_count=len(seen) - unique_seen,
        epochs_completed_or_fractional=len(seen) / arrays.input_ids.shape[0],
        loss_trace=tuple(loss_trace),
        checkpoint_path=checkpoint_path,
        loss_trace_path=loss_trace_path,
        checkpoint_written=checkpoint_path.is_file(),
        checkpoint_nonzero=checkpoint_path.is_file()
        and checkpoint_path.stat().st_size > 0,
        params_changed=params_changed,
    )


def _batch_indices(
    *,
    step: int,
    batch_size: int,
    examples_available: int,
    allow_reuse: bool,
) -> list[int]:
    start = step * batch_size
    if allow_reuse:
        return [
            int((start + offset) % examples_available) for offset in range(batch_size)
        ]
    return [int(start + offset) for offset in range(batch_size)]


def _dense_distill_loss(
    params: dict[str, jax.Array],
    input_ids: jax.Array,
    attention_mask: jax.Array,
    teacher_logits: jax.Array,
) -> jax.Array:
    student_logits = params["vocab_bias"][None, None, :] + params[
        "token_scale"
    ] * jax.nn.one_hot(input_ids, teacher_logits.shape[-1])
    teacher_log_probs = jax.nn.log_softmax(teacher_logits, axis=-1)
    teacher_probs = jnp.exp(teacher_log_probs)
    student_log_probs = jax.nn.log_softmax(student_logits, axis=-1)
    token_kl = jnp.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1)
    numerator = jnp.sum(token_kl * attention_mask)
    denominator = jnp.maximum(
        jnp.sum(attention_mask), jnp.asarray(1.0, dtype=jnp.float32)
    )
    return numerator / denominator


def _params_changed(
    before: dict[str, jax.Array],
    after: dict[str, jax.Array],
) -> bool:
    return any(
        bool(jnp.any(jnp.asarray(before[name]) != jnp.asarray(after[name])))
        for name in before
    )


def _real_training_blockers(training: _TrainingRun) -> tuple[str, ...]:
    blockers: list[str] = []
    if training.steps_completed < 1:
        blockers.append("real mode executed zero train steps")
    if (
        not training.loss_trace
        or not np.isfinite(np.asarray(training.loss_trace)).all()
    ):
        blockers.append("real mode training loss was not finite")
    if not training.checkpoint_written:
        blockers.append("real mode did not write a checkpoint")
    if not training.checkpoint_nonzero:
        blockers.append("real mode checkpoint is empty")
    if not training.params_changed:
        blockers.append("real mode optimizer did not change parameters")
    return tuple(blockers)


def _jax_backend() -> str:
    try:
        return str(jax.default_backend())
    except Exception:
        return "unavailable"


def _jax_devices() -> tuple[str, ...]:
    try:
        return tuple(str(device) for device in jax.devices())
    except Exception:
        return ()


def _launch_commands_markdown() -> str:
    return """# P112 First Serious Compute Burn Launch Commands

Review readiness before attempting the serious burn. The real command must be
run manually and includes `--confirm-serious-burn`.

```bash
python scripts/run_big_burn_readiness_report.py \\
  --output artifacts/p111_big_burn_readiness/readiness_report.json

python scripts/run_runtime_environment_preflight.py \\
  --output artifacts/p109_runtime_environment/runtime_environment_report.json

python scripts/run_first_serious_burn.py \\
  --config artifacts/p112_first_serious_burn/burn_config.json \\
  --output artifacts/p112_first_serious_burn/dry_run \\
  --mode dry_run

python scripts/run_first_serious_burn.py \\
  --config artifacts/p112_first_serious_burn/burn_config.json \\
  --output artifacts/p112_first_serious_burn/run_001 \\
  --mode real \\
  --confirm-serious-burn
```

Real mode should only be run after reviewing P111 readiness output and
explicitly accepting any warnings.
"""
