from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from qrwkv_xla.tracking import (
    create_run_context,
    get_git_metadata,
    make_run_id,
    write_run_metadata,
    write_run_summary,
)
from qrwkv_xla.tracking.json_io import to_jsonable, write_json


def test_make_run_id_uses_utc_timestamp_and_slug() -> None:
    run_id = make_run_id(
        stage=0,
        student_architecture="rwkv7_reference",
        run_name="Stage 0 Smoke!",
        now=datetime(2026, 4, 30, 12, 5, 6, tzinfo=UTC),
    )

    assert run_id == "20260430_120506_stage_0_smoke"


def test_create_run_context_creates_expected_paths(tmp_path: Path) -> None:
    context = create_run_context(
        run_root=tmp_path / "runs",
        stage=0,
        student_architecture="tiny_student",
        run_name="unit",
        command=["python", "scripts/run_distill_stage.py"],
        git={"available": False},
        environment={"python_version": "3.11"},
        distillation={"stage": 0},
        teacher_target={"targets_dir": "bundle"},
        student={"architecture": "tiny_student"},
        checkpoint={"resume_from": None},
    )

    assert context.paths.run_dir.is_dir()
    assert context.paths.run_json == context.paths.run_dir / "run.json"
    assert context.paths.metrics_jsonl == context.paths.run_dir / "metrics.jsonl"
    assert context.paths.summary_json == context.paths.run_dir / "summary.json"
    assert context.paths.checkpoints_dir == context.paths.run_dir / "checkpoints"


def test_write_run_metadata_and_summary_are_valid_json(tmp_path: Path) -> None:
    context = create_run_context(
        run_root=tmp_path / "runs",
        stage=0,
        student_architecture="tiny_student",
        run_name="unit",
        command=["python", "scripts/run_distill_stage.py"],
        git={"available": False},
        environment={"python_version": "3.11"},
        distillation={"stage": 0, "config": {"value": 3}},
        teacher_target={"targets_dir": "bundle"},
        student={"architecture": "tiny_student"},
        checkpoint={"resume_from": None},
        tags=["smoke"],
        notes=["local only"],
    )

    write_run_metadata(context)
    write_run_summary(
        context=context,
        summary={"status": "completed", "final_loss": 1.23},
    )

    run_text = context.paths.run_json.read_text(encoding="utf-8")
    run_payload = json.loads(run_text)
    summary_payload = json.loads(context.paths.summary_json.read_text(encoding="utf-8"))

    assert "\n  " in run_text
    assert run_payload["run_name"] == "unit"
    assert run_payload["tags"] == ["smoke"]
    assert run_payload["distillation"]["config"]["value"] == 3
    assert summary_payload["summary"]["final_loss"] == 1.23


def test_overwrite_false_rejects_existing_run_dir(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    existing = run_root / "20260430_120506_unit"
    existing.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        create_run_context(
            run_root=run_root,
            stage=0,
            student_architecture="tiny_student",
            run_name="unit",
            command=[],
            git={"available": False},
            environment={},
            distillation={},
            teacher_target={},
            student={},
            checkpoint={},
            now=datetime(2026, 4, 30, 12, 5, 6, tzinfo=UTC),
        )


def test_jsonable_converts_paths_and_objects(tmp_path: Path) -> None:
    payload = to_jsonable({"config": {"output": tmp_path / "out", "value": 3}})
    assert payload["config"]["output"] == str(tmp_path / "out")
    written = write_json(tmp_path / "runs" / "unit" / "payload.json", payload)
    assert json.loads(written.read_text(encoding="utf-8")) == payload


def test_git_metadata_is_best_effort_for_missing_repo(tmp_path: Path) -> None:
    metadata = get_git_metadata(tmp_path / "missing")
    assert metadata["available"] is False
