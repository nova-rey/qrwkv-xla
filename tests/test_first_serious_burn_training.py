from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from qrwkv_xla.burn import (
    FirstSeriousBurnConfig,
    default_first_serious_burn_config,
    run_first_serious_burn,
    write_first_serious_burn_config,
)
from qrwkv_xla.targets import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
    TargetStoreMetadata,
    TeacherTargetStore,
)

ROOT = Path(__file__).resolve().parents[1]


def test_real_mode_runs_8_unique_train_steps_without_reuse(tmp_path: Path) -> None:
    store = _dense_textbook(tmp_path, examples=8)
    result = run_first_serious_burn(
        _real_config(tmp_path, store=store, max_steps=8, batch_size=1),
        confirm_serious_burn=True,
    )

    assert result.status == "pass"
    assert result.real_training_executed is True
    assert result.steps_completed == 8
    assert result.examples_consumed == 8
    assert result.unique_examples_consumed == 8
    assert result.reuse_count == 0
    assert result.checkpoint_written is True
    assert result.checkpoint_path is not None
    assert Path(result.checkpoint_path).is_file()
    assert result.loss_trace_path is not None
    assert Path(result.loss_trace_path).is_file()
    assert result.loss_initial is not None
    assert result.loss_final is not None


def test_no_reuse_allows_exact_floor_steps(tmp_path: Path) -> None:
    store = _dense_textbook(tmp_path, examples=8)
    result = run_first_serious_burn(
        _real_config(tmp_path, store=store, max_steps=4, batch_size=2),
        confirm_serious_burn=True,
    )

    assert result.status == "pass"
    assert result.steps_completed == 4
    assert result.examples_consumed == 8


def test_no_reuse_fails_before_training_when_examples_are_insufficient(
    tmp_path: Path,
) -> None:
    store = _dense_textbook(tmp_path, examples=8)
    result = run_first_serious_burn(
        _real_config(tmp_path, store=store, max_steps=5, batch_size=2),
        confirm_serious_burn=True,
    )

    assert result.status == "blocked"
    assert result.steps_completed == 0
    assert result.real_training_executed is False
    assert "requires 10 examples" in result.blockers[0]


def test_reuse_mode_cycles_examples_for_longer_smoke(tmp_path: Path) -> None:
    store = _dense_textbook(tmp_path, examples=8)
    result = run_first_serious_burn(
        _real_config(
            tmp_path,
            store=store,
            max_steps=100,
            batch_size=4,
            allow_reuse=True,
        ),
        confirm_serious_burn=True,
    )

    assert result.status == "pass"
    assert result.steps_completed == 100
    assert result.examples_consumed == 400
    assert result.unique_examples_consumed == 8
    assert result.reuse_count == 392
    assert result.epochs_completed_or_fractional == 50.0


def test_real_mode_without_textbook_cannot_pass_zero_steps(tmp_path: Path) -> None:
    result = run_first_serious_burn(
        replace(
            default_first_serious_burn_config(output_dir=tmp_path, mode="real"),
            readiness_report_path=str(_readiness_report(tmp_path, status="pass")),
        ),
        confirm_serious_burn=True,
    )

    assert result.status == "blocked"
    assert result.steps_completed == 0
    assert result.real_training_executed is False
    assert result.blockers == (
        "real mode requires --teacher-textbook or target_store_path",
    )


def test_real_mode_checkpoint_records_updated_parameter_summary(
    tmp_path: Path,
) -> None:
    store = _dense_textbook(tmp_path, examples=2)
    result = run_first_serious_burn(
        _real_config(tmp_path, store=store, max_steps=1, batch_size=1),
        confirm_serious_burn=True,
    )

    checkpoint = json.loads(Path(result.checkpoint_path).read_text(encoding="utf-8"))

    assert checkpoint["step"] == 1
    assert checkpoint["batch_size"] == 1
    assert checkpoint["max_steps"] == 1
    assert checkpoint["allow_textbook_reuse"] is False
    assert checkpoint["param_summary"]["params_changed"] is True
    assert checkpoint["teacher_textbook_path"] == str(store.root)


def test_cli_parses_train_step_knobs_for_dry_run(tmp_path: Path) -> None:
    config_path = tmp_path / "burn_config.json"
    output_dir = tmp_path / "cli"
    write_first_serious_burn_config(
        replace(
            default_first_serious_burn_config(output_dir=output_dir),
            readiness_report_path=str(_readiness_report(tmp_path, status="pass")),
        ),
        config_path,
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_first_serious_burn.py"),
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
            "--max-steps",
            "8",
            "--batch-size",
            "1",
            "--no-allow-textbook-reuse",
        ],
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads((output_dir / "burn_report.json").read_text(encoding="utf-8"))
    assert report["max_steps_requested"] == 8
    assert report["batch_size"] == 1
    assert report["allow_textbook_reuse"] is False


def test_cli_parses_allow_textbook_reuse(tmp_path: Path) -> None:
    config_path = tmp_path / "burn_config.json"
    output_dir = tmp_path / "cli_reuse"
    write_first_serious_burn_config(
        replace(
            default_first_serious_burn_config(output_dir=output_dir),
            readiness_report_path=str(_readiness_report(tmp_path, status="pass")),
        ),
        config_path,
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_first_serious_burn.py"),
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
            "--allow-textbook-reuse",
        ],
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads((output_dir / "burn_report.json").read_text(encoding="utf-8"))
    assert report["allow_textbook_reuse"] is True


def _real_config(
    tmp_path: Path,
    *,
    store: TeacherTargetStore,
    max_steps: int,
    batch_size: int,
    allow_reuse: bool = False,
) -> FirstSeriousBurnConfig:
    return replace(
        default_first_serious_burn_config(output_dir=tmp_path / "burn", mode="real"),
        phase="P117.1",
        readiness_report_path=str(_readiness_report(tmp_path, status="pass")),
        teacher_textbook_path=str(store.root),
        max_steps=max_steps,
        batch_size=batch_size,
        allow_textbook_reuse=allow_reuse,
    )


def _dense_textbook(tmp_path: Path, *, examples: int) -> TeacherTargetStore:
    store = TeacherTargetStore.create(
        tmp_path / "teacher_textbook",
        TargetStoreMetadata(
            schema_version=TEACHER_TARGET_STORE_SCHEMA_VERSION,
            target_store_version=TEACHER_TARGET_STORE_VERSION,
            model_id="unit-dense-teacher",
            model_family="synthetic",
            tokenizer_id="unit-tokenizer",
            tokenizer_hash=None,
            vocab_size=5,
            target_type="synthetic",
            dtype="float32",
            sequence_length=3,
            num_examples=examples,
            shard_count=1,
            created_by="test",
            created_at="2026-06-05T00:00:00Z",
            source={"kind": "unit"},
            provenance={"phase": "P117.1"},
        ),
        overwrite=True,
    )
    input_ids = (np.arange(examples * 3, dtype=np.int32) % 5).reshape(examples, 3)
    vocab = np.arange(5, dtype=np.float32)
    logits = input_ids[:, :, None].astype(np.float32) * 0.1 + vocab[None, None, :]
    store.write_shard(
        0,
        {
            "input_ids": input_ids,
            "attention_mask": np.ones_like(input_ids, dtype=np.int32),
            "logits": logits.astype(np.float32),
        },
    )
    return TeacherTargetStore.open(store.root)


def _readiness_report(tmp_path: Path, *, status: str) -> Path:
    path = tmp_path / f"readiness_{status}.json"
    path.write_text(
        json.dumps(
            {
                "phase": "P111",
                "status": status,
                "blockers": [],
                "warnings": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
