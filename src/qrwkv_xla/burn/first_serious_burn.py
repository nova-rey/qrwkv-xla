from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
        return FirstSeriousBurnResult(
            phase="P112",
            status="pass",
            mode=config.mode,
            dry_run=False,
            readiness_status=readiness.status,
            steps_requested=config.max_steps,
            steps_completed=0,
            output_dir=str(output_dir),
            readiness_report_path=str(readiness.path) if readiness.path else None,
            preflight_report_path=None,
            checkpoint_path=None,
            eval_report_path=None,
            launch_commands_path=str(launch_commands_path),
            blockers=(),
            warnings=readiness_warnings
            + (
                "real mode gate passed; P112 harness did not execute expensive "
                "training in baseline path",
            ),
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
        phase="P112",
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
    if config.phase != "P112":
        raise ValueError(f"phase must be 'P112', got {config.phase!r}")
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
