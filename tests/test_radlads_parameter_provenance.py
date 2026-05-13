"""Tests for RADLADS parameter provenance audit functionality."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from qrwkv_xla.parity.audit_radlads_parameter_provenance import (
    audit_radlads_parameter_provenance,
    write_provenance_audit_report,
)
from qrwkv_xla.parity.radlads_fixture_validation import (
    analyze_array,
    audit_parameter_payload,
    compute_sha256_from_array,
    to_audit_report,
    validate_parameter_payload,
    write_audit_report,
)

ROOT = Path(__file__).resolve().parents[1]


def test_compute_sha256_from_array_is_deterministic() -> None:
    """Test that SHA256 computation is deterministic."""
    arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    arr2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    arr3 = np.array([3.0, 2.0, 1.0], dtype=np.float32)

    hash1 = compute_sha256_from_array(arr1)
    hash2 = compute_sha256_from_array(arr2)
    hash3 = compute_sha256_from_array(arr3)

    assert hash1 == hash2, "Same array should produce same hash"
    assert hash1 != hash3, "Different array should produce different hash"


def test_analyze_array_detects_finite_values() -> None:
    """Test analysis of finite values."""
    result = analyze_array(
        "test.weight",
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        stage="test_stage",
    )

    assert result.name == "test.weight"
    assert result.stage == "test_stage"
    assert result.shape == [2, 2]
    assert result.dtype == "float32"
    assert result.finite_count == 4
    assert result.nan_count == 0
    assert result.abs_max == 4.0
    assert result.status == "finite_ok"


def test_analyze_array_detects_nan() -> None:
    """Test detection of NaN values."""
    result = analyze_array(
        "bad.weight",
        np.array([1.0, np.nan, 3.0], dtype=np.float32),
        stage="test_stage",
    )

    assert result.nan_count == 1
    assert result.finite_count == 2
    assert result.status == "non_finite"


def test_analyze_array_detects_inf() -> None:
    """Test detection of Inf values."""
    result = analyze_array(
        "inf.weight",
        np.array([1.0, np.inf, -np.inf], dtype=np.float32),
        stage="test_stage",
    )

    assert result.posinf_count == 1
    assert result.neginf_count == 1
    assert result.finite_count == 1
    assert result.status == "non_finite"


def test_analyze_array_detects_extreme_values() -> None:
    """Test detection of extreme values above threshold."""
    result = analyze_array(
        "huge.weight",
        np.array([1e9], dtype=np.float32),
        stage="test_stage",
        extreme_threshold=1e6,
    )

    assert result.abs_max == 1e9
    assert result.status == "extreme_value"


def test_analyze_array_all_zero_flag() -> None:
    """Test all_zero flag detection."""
    result = analyze_array(
        "zeros",
        np.zeros((3, 4), dtype=np.float32),
        stage="test_stage",
    )

    assert result.all_zero is True
    assert result.status == "finite_ok"


def test_analyze_array_all_same_flag() -> None:
    """Test all_same flag detection."""
    result = analyze_array(
        "uniform",
        np.ones((3, 4), dtype=np.float32) * 5.0,
        stage="test_stage",
    )

    assert result.all_same is True


def test_audit_parameter_payload_batch_audit() -> None:
    """Test batch auditing of multiple parameters."""
    params = {
        "ok.weight": np.array([1.0, 2.0], dtype=np.float32),
        "nan.weight": np.array([np.nan], dtype=np.float32),
        "huge.weight": np.array([1e9], dtype=np.float32),
    }

    results = audit_parameter_payload(
        params,
        stage="pre_save",
        extreme_threshold=1e6,
        seed=4949,
    )

    assert len(results) == 3
    by_name = {r.name: r for r in results}

    assert by_name["huge.weight"].status == "extreme_value"
    assert by_name["nan.weight"].status == "non_finite"
    assert by_name["ok.weight"].status == "finite_ok"


def test_validate_parameter_payload() -> None:
    """Test validation of audit results."""
    results = [
        analyze_array("ok", np.array([1.0]), stage="test"),
        analyze_array("nan", np.array([np.nan]), stage="test"),
    ]

    is_valid, blocking = validate_parameter_payload(results)

    assert is_valid is False
    assert len(blocking) == 1
    assert blocking[0].name == "nan"


def test_to_audit_report_json_serializable() -> None:
    """Test that audit report is JSON serializable."""
    results = [
        analyze_array("test", np.array([1.0, 2.0]), stage="test"),
    ]

    report = to_audit_report(results)

    # Should be JSON serializable
    json_str = json.dumps(report)
    assert "parameter_count" in json_str
    assert "summary" in json_str


def test_write_audit_report_files_created(tmp_path: Path) -> None:
    """Test that audit report writes files."""
    results = [
        analyze_array("test", np.array([1.0, 2.0]), stage="test"),
        analyze_array("nan", np.array([np.nan]), stage="test"),
    ]

    write_audit_report(
        to_audit_report(results),
        tmp_path,
    )

    assert (tmp_path / "parameter_provenance_report.json").is_file()
    assert (tmp_path / "P52_PARAMETER_PROVENANCE.md").is_file()


def test_audit_with_existing_fixture(tmp_path: Path) -> None:
    """Test audit with existing RADLADS fixture."""
    manifest_path = (
        ROOT
        / "artifacts"
        / "p49_radlads_numerical_parity"
        / "radlads_fixtures"
        / "manifest.json"
    )
    parameters_path = (
        ROOT
        / "artifacts"
        / "p49_radlads_numerical_parity"
        / "radlads_fixtures"
        / "radlads_parameters.npz"
    )

    if not manifest_path.exists() or not parameters_path.exists():
        # Skip if fixtures not available
        return

    audit_result = audit_radlads_parameter_provenance(
        manifest_path=manifest_path,
        parameters_path=parameters_path,
        seed=4949,
    )

    # Should detect non-finite parameters
    blocking = audit_result.get("blocking_issues", [])
    assert len(blocking) > 0, "Should detect non-finite parameters in fixtures"

    # Check summary structure
    summary = audit_result.get("summary", {})
    assert "aggregated_by_status" in summary

    # Check recommendations
    recommendations = audit_result.get("recommendations", [])
    assert len(recommendations) > 0


def test_provenance_audit_report_writing(tmp_path: Path) -> None:
    """Test full provenance audit report writing."""
    manifest_path = (
        ROOT
        / "artifacts"
        / "p49_radlads_numerical_parity"
        / "radlads_fixtures"
        / "manifest.json"
    )
    parameters_path = (
        ROOT
        / "artifacts"
        / "p49_radlads_numerical_parity"
        / "radlads_fixtures"
        / "radlads_parameters.npz"
    )

    if not manifest_path.exists() or not parameters_path.exists():
        return

    audit_result = audit_radlads_parameter_provenance(
        manifest_path=manifest_path,
        parameters_path=parameters_path,
        seed=4949,
    )

    write_provenance_audit_report(audit_result, tmp_path)

    assert (tmp_path / "P52_PROVENANCE_AUDIT.md").is_file()
    assert (tmp_path / "provenance_audit.json").is_file()


def test_audit_detects_specific_suspicious_parameters() -> None:
    """Test that audit detects the specific suspicious parameters from P51."""
    # These are the parameters P51 flagged
    suspicious_params = {
        "layers.0.self_attn.w1": np.array([np.nan, np.nan], dtype=np.float32),
        "layers.0.self_attn.w2": np.array([np.nan], dtype=np.float32),
        "layers.0.self_attn.v1": np.array([1.79e35], dtype=np.float32),
        "layers.0.self_attn.v2": np.array([1.79e35], dtype=np.float32),
        "layers.0.self_attn.a0": np.array([1.79e35], dtype=np.float32),
        "layers.0.self_attn.r_k": np.array([1.79e35], dtype=np.float32),
        "layers.0.self_attn.g1": np.array([1.79e35], dtype=np.float32),
        "layers.0.self_attn.g2": np.array([1.79e35], dtype=np.float32),
    }

    results = audit_parameter_payload(
        suspicious_params,
        stage="saved_npz",
        extreme_threshold=1e6,
        seed=4949,
    )

    by_name = {r.name: r for r in results}

    # w1 should be non_finite (NaN)
    assert by_name["layers.0.self_attn.w1"].status == "non_finite"
    assert by_name["layers.0.self_attn.w2"].status == "non_finite"

    # v1, v2, a0, r_k, g1, g2 should be extreme_value (> 1e6)
    assert by_name["layers.0.self_attn.v1"].status == "extreme_value"
    assert by_name["layers.0.self_attn.v2"].status == "extreme_value"
    assert by_name["layers.0.self_attn.a0"].status == "extreme_value"
    assert by_name["layers.0.self_attn.r_k"].status == "extreme_value"
    assert by_name["layers.0.self_attn.g1"].status == "extreme_value"
    assert by_name["layers.0.self_attn.g2"].status == "extreme_value"


def test_cli_help() -> None:
    """Test that CLI help is available."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_radlads_fixture_parameter_provenance.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "P52" in result.stdout
    assert "--radlads-repo" in result.stdout
    assert "--manifest" in result.stdout
    assert "--parameters" in result.stdout
    assert "--out" in result.stdout
