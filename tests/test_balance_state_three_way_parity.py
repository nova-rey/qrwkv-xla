from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from qrwkv_xla.parity.radlads_balance_state_three_way import (
    BALANCE_STATE_THREE_WAY_SCHEMA,
    run_balance_state_three_way,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_balance_state_radlads_three_way.py"


def test_three_way_report_writes_required_artifacts(tmp_path: Path) -> None:
    report = run_balance_state_three_way(out_dir=tmp_path, overwrite=True)

    assert report["schema"] == BALANCE_STATE_THREE_WAY_SCHEMA
    assert report["phase"] == "P66"
    assert report["strict_real_artifacts"] is True
    assert report["synthetic_fallback_used"] is False
    assert report["default_behavior_preserved"] is True
    assert report["balance_state_mode_promoted"] is False
    assert report["p58_log_w_preserved"] is True
    assert report["recommendations"] == [report["recommendation"]]
    assert report["recommendation"] in {
        "P67 promote/harden balance-state compatibility path",
        "P67 targeted k/v update_outer_product parity fix",
        "P67 targeted balance-state formula/shape fix",
        "P67 residual-impact / kernel-readiness gate",
        "P67 Pallas prototype behind known-caveat flag",
    }
    assert "experimental_closer_to_radlads" in report
    assert "stage_summary" in report
    assert "cases" in report

    for name in (
        "radlads_update_boundary_trace.jsonl",
        "qrwkv_off_update_boundary_trace.jsonl",
        "qrwkv_experimental_update_boundary_trace.jsonl",
        "three_way_trace_radlads.jsonl",
        "three_way_trace_qrwkv_off.jsonl",
        "three_way_trace_qrwkv_experimental.jsonl",
        "THREE_WAY_PARITY.md",
        "UPDATE_BOUNDARY_PARITY.md",
        "BALANCE_STATE_DECISION.md",
        "P66_RESULTS.md",
        "three_way_parity_report.json",
    ):
        assert (tmp_path / name).is_file()

    payload = json.loads((tmp_path / "three_way_parity_report.json").read_text())
    assert payload["schema"] == BALANCE_STATE_THREE_WAY_SCHEMA
    assert payload["rows"]
    assert {
        "radlads_vs_qrwkv_off",
        "radlads_vs_qrwkv_experimental",
        "qrwkv_off_vs_qrwkv_experimental",
    } <= payload["rows"][0].keys()


def test_decision_report_emits_exactly_one_recommendation(tmp_path: Path) -> None:
    run_balance_state_three_way(out_dir=tmp_path, overwrite=True)

    text = (tmp_path / "BALANCE_STATE_DECISION.md").read_text(encoding="utf-8")
    assert text.count("recommendation:") == 1


def test_trace_jsonl_contains_raw_values(tmp_path: Path) -> None:
    run_balance_state_three_way(out_dir=tmp_path, overwrite=True)

    first = json.loads(
        (tmp_path / "qrwkv_experimental_update_boundary_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert first["schema"] == "qrwkv_xla.p66_balance_state_trace.v1"
    assert first["side"] == "qrwkv_experimental"
    assert first["array"] is not None
    assert first["comparison_label"] in {
        "state_before",
        "decay_value",
        "decayed_state",
        "update_outer_product",
        "balance_state_term",
        "composite_update_term",
        "final_update_term",
        "state_after",
        "composite_balance_update_term",
        "state_after_from_full_source_formula",
        "residual_after_composite_term",
    }


def test_script_help_and_write(tmp_path: Path) -> None:
    help_result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "P66" in help_result.stdout
    assert "three-way" in help_result.stdout

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out-dir",
            str(tmp_path / "p66"),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "recommendation=" in result.stdout
