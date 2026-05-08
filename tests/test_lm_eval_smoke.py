from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from qrwkv_xla.eval.exported_student import (
    ExportedStudentEvalAdapter,
    load_toy_continuation_task,
    run_toy_exported_student_eval,
)
from scripts.run_export_smoke import run_export_smoke

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_lm_eval_smoke.py"
TASK = ROOT / "tests" / "fixtures" / "eval" / "p42_toy_continuations.jsonl"


def test_exported_student_adapter_scores_toy_task(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    export_dir = tmp_path / "p41_export"
    run_export_smoke(export_dir, overwrite=True)
    examples = load_toy_continuation_task(TASK)

    adapter = ExportedStudentEvalAdapter(export_dir)
    score = adapter.loglikelihood(examples[0])
    result = run_toy_exported_student_eval(export_dir=export_dir, task_path=TASK)

    assert score.num_tokens_scored == 2
    assert result.num_examples == 2
    assert result.num_tokens_scored == 3
    assert result.mean_neg_loglikelihood > 0.0
    assert result.perplexity > 0.0
    assert result.official_lm_eval["integrated"] is False


def test_run_lm_eval_smoke_cli_writes_results_and_bundle(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    export_dir = tmp_path / "artifacts" / "p41"
    out_dir = tmp_path / "artifacts" / "eval" / "p42"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--export-dir",
            str(export_dir),
            "--out",
            str(out_dir),
            "--overwrite",
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["passed"] is True
    assert report["export_created"] is True
    assert report["num_examples"] == 2
    assert report["num_tokens_scored"] == 3
    assert (out_dir / "results.json").is_file()
    assert (out_dir / "P42_RESULTS.md").is_file()
    bundle = out_dir / "p42_results_bundle.tar.gz"
    assert bundle.is_file()
    with tarfile.open(bundle, "r:gz") as tar:
        assert sorted(tar.getnames()) == ["P42_RESULTS.md", "results.json"]


def test_run_lm_eval_smoke_fails_for_partial_export(tmp_path: Path) -> None:
    partial_export = tmp_path / "partial"
    partial_export.mkdir()
    (partial_export / "config.json").write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--export-dir",
            str(partial_export),
            "--out",
            str(out_dir),
            "--overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "missing required files" in completed.stderr
