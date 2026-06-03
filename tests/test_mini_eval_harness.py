from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from qrwkv_xla.contracts import vocab_contract_from_metadata
from qrwkv_xla.eval import (
    create_builtin_mini_eval_store,
    run_mini_eval_harness,
    write_mini_eval_report,
)
from qrwkv_xla.students import TINY_DEBUG_ARCHITECTURE_ID

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_mini_eval_harness.py"


def test_mini_eval_runs_on_tiny_two_shard_target_store(tmp_path: Path) -> None:
    result = run_mini_eval_harness(
        store=_store(tmp_path),
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
    )

    assert result.status == "pass"
    assert result.shard_count == 2


def test_registry_selected_student_backend_participates(tmp_path: Path) -> None:
    result = run_mini_eval_harness(
        store=_store(tmp_path),
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
    )

    assert result.architecture_id == TINY_DEBUG_ARCHITECTURE_ID
    assert result.runtime == "reference"


def test_compatibility_gate_passes_before_metrics(tmp_path: Path) -> None:
    result = run_mini_eval_harness(
        store=_store(tmp_path),
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
    )

    assert result.compatibility_status == "compatible"
    assert result.compatibility_reason.startswith("compatible:")
    assert result.mean_mse_loss is not None


def test_compatibility_mismatch_blocks_eval_clearly(tmp_path: Path) -> None:
    store = _store(tmp_path)
    contract = vocab_contract_from_metadata(store.metadata)

    result = run_mini_eval_harness(
        store=store,
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
        student_vocab_contract=replace(contract, vocab_size=9),
    )

    assert result.status == "incompatible"
    assert result.compatibility_status == "incompatible"
    assert "vocab_size mismatch" in result.compatibility_reason
    assert result.mean_mse_loss is None
    assert result.examples_evaluated == 0


def test_mean_mse_loss_is_finite(tmp_path: Path) -> None:
    result = run_mini_eval_harness(
        store=_store(tmp_path),
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
    )

    assert result.mean_mse_loss is not None
    assert np.isfinite(result.mean_mse_loss)
    assert result.loss_finite is True


def test_eval_counts_match_tiny_shape_and_metadata(tmp_path: Path) -> None:
    store = _store(tmp_path)

    result = run_mini_eval_harness(
        store=store,
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
    )

    assert result.examples_evaluated == store.metadata.num_examples
    assert result.shard_count == store.metadata.shard_count
    assert result.tokens_evaluated == 12
    assert result.elements_evaluated == 96


def test_top1_agreement_is_toy_metric_between_zero_and_one(tmp_path: Path) -> None:
    result = run_mini_eval_harness(
        store=_store(tmp_path),
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
    )

    assert result.top1_agreement is not None
    assert 0.0 <= result.top1_agreement <= 1.0


def test_json_report_is_written_and_inspectable(tmp_path: Path) -> None:
    result = run_mini_eval_harness(
        store=_store(tmp_path),
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
    )
    report_path = write_mini_eval_report(result, tmp_path / "mini_eval.json")

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["phase"] == "P110"
    assert report["status"] == "pass"
    assert report["examples_evaluated"] == 4


def test_report_includes_claims_not_made(tmp_path: Path) -> None:
    result = run_mini_eval_harness(
        store=_store(tmp_path),
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
    )
    report = result.to_report()

    assert "claims_not_made" in report
    assert "benchmark_complete" in report["claims_not_made"]


def test_no_hf_internet_accelerator_or_qwen_is_required(tmp_path: Path) -> None:
    result = run_mini_eval_harness(
        store=_store(tmp_path),
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
    )

    assert result.status == "pass"
    assert "qwen_specific_support" in result.claims_not_made


def test_no_training_optimizer_or_quality_claims_are_made(tmp_path: Path) -> None:
    result = run_mini_eval_harness(
        store=_store(tmp_path),
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
    )

    assert "training_ready" in result.claims_not_made
    assert "model_quality_proven" in result.claims_not_made
    assert "production_eval_ready" in result.claims_not_made


def test_mini_eval_cli_writes_report_with_builtin_store(tmp_path: Path) -> None:
    output_path = tmp_path / "artifacts" / "mini_eval.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(output_path),
            "--architecture-id",
            TINY_DEBUG_ARCHITECTURE_ID,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["phase"] == "P110"
    assert report["status"] == "pass"


def _store(tmp_path: Path):
    return create_builtin_mini_eval_store(tmp_path / "targets")
