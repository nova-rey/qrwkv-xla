from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.parity.radlads_clean_loader import export_qrwkv_clean_payload_outputs
from qrwkv_xla.parity.radlads_numerical_fixtures import (
    generate_radlads_tiny_numerical_fixtures,
)
from qrwkv_xla.parity.radlads_wkv_state_convention import (
    REFERENCE_STATE_EXPORT_PATH,
    REFERENCE_STATE_IMPORT_PATH,
    WKV_STATE_CONVENTION_REPORT_SCHEMA,
    WKV_STATE_EXPORT_SCHEMA,
    WKV_STATE_SLOT_AUDIT_SCHEMA,
    compare_wkv_matrix_state_conventions,
    export_reference_state_object,
    import_reference_state_object,
    normalize_qrwkv_wkv_matrix_state,
    normalize_radlads_wkv_matrix_state,
)

ROOT = Path(__file__).resolve().parents[1]
INSPECT_SCRIPT = ROOT / "scripts" / "inspect_radlads_qrwkv_wkv_state_slots.py"
COMPARE_SCRIPT = (
    ROOT / "scripts" / "compare_radlads_qrwkv_head_to_head_normalized_state.py"
)


def _prepare_fixture_bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    fixture_root = tmp_path / "p54_confirmation" / "fixtures"
    generate_radlads_tiny_numerical_fixtures(
        fixture_root,
        overwrite=True,
        init_policy="deterministic_finite",
    )
    parameter_payload = fixture_root / "radlads_parameters.npz"

    qrwkv_out = tmp_path / "qrwkv_outputs"
    export_qrwkv_clean_payload_outputs(parameter_payload, qrwkv_out, overwrite=True)

    radlads_out = tmp_path / "radlads_outputs"
    radlads_out.mkdir(parents=True, exist_ok=True)
    for path in qrwkv_out.glob("*.npz"):
        with np.load(path) as arrays:
            renamed = {
                (
                    key.replace("qrwkv_", "radlads_")
                    if key.startswith("qrwkv_")
                    else key
                ): arrays[key]
                for key in arrays.files
            }
        np.savez(radlads_out / path.name, **renamed)

    manifest = json.loads((qrwkv_out / "manifest.json").read_text())
    manifest["side"] = "radlads"
    manifest["overall_status"] = "pass"
    (radlads_out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    p60_report = tmp_path / "p60_report.json"
    p60_report.write_text(
        json.dumps(
            {
                "first_real_divergence_case": "tiny_attention_mask",
                "hidden_states_explained": "comparison_only",
                "regenerated_live_outputs": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return fixture_root / "manifest.json", radlads_out, qrwkv_out, p60_report


def test_slot_audit_report_schema_validates(tmp_path: Path) -> None:
    fixture_manifest, radlads_out, qrwkv_out, p60_report = _prepare_fixture_bundle(
        tmp_path
    )
    out = tmp_path / "slot_audit"
    subprocess.run(
        [
            sys.executable,
            str(INSPECT_SCRIPT),
            "--fixture-manifest",
            str(fixture_manifest),
            "--radlads-outputs",
            str(radlads_out),
            "--qrwkv-outputs",
            str(qrwkv_out),
            "--p60-report",
            str(p60_report),
            "--out",
            str(out),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((out / "wkv_state_slot_audit.json").read_text())
    assert report["schema"] == WKV_STATE_SLOT_AUDIT_SCHEMA
    assert report["wkv_matrix_state_slot_match"] is True
    assert (out / "state_slot_samples.npz").is_file()


def test_normalization_refuses_unknown_slot_layout() -> None:
    with pytest.raises(KeyError):
        normalize_radlads_wkv_matrix_state({}, slot_name="missing")


def test_as_is_normalization_preserves_values() -> None:
    value = np.arange(12, dtype=np.float32).reshape(1, 1, 3, 2, 2)
    result = normalize_radlads_wkv_matrix_state(value)
    assert np.array_equal(result["source_array"], value)
    assert np.array_equal(result["normalized_array"], value)


def test_reference_state_export_import_roundtrip_is_source_backed() -> None:
    state = {
        "wkv_matrix_state": np.arange(4, dtype=np.float32).reshape(1, 1, 1, 2, 2),
        "shift_state": np.zeros((1, 1, 2), dtype=np.float32),
        "next_position": np.array(2, dtype=np.int32),
    }

    exported = export_reference_state_object(state)
    imported = import_reference_state_object(exported)

    assert exported["schema"] == WKV_STATE_EXPORT_SCHEMA
    assert exported["export_path"] == REFERENCE_STATE_EXPORT_PATH
    assert REFERENCE_STATE_IMPORT_PATH.endswith("import_reference_state_object")
    assert exported["representation"] == "reference_state_slots"
    assert np.array_equal(
        imported["wkv_matrix_state"],  # type: ignore[index]
        state["wkv_matrix_state"],
    )


def test_axis_normalization_works_on_synthetic_known_layout() -> None:
    value = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.float32).reshape(2, 3, 4, 5, 6)
    result = normalize_qrwkv_wkv_matrix_state(
        value, normalization="transpose_matrix_axes"
    )
    expected = np.swapaxes(value, -1, -2)
    assert np.array_equal(result["normalized_array"], expected)


def test_slot_swap_normalization_works_on_synthetic_known_layout() -> None:
    source = {
        "state_slots": {
            "wkv_matrix_state": np.zeros((1, 1, 1, 2, 2), dtype=np.float32),
            "shift_state": np.ones((1, 1, 2), dtype=np.float32),
        }
    }
    result = normalize_qrwkv_wkv_matrix_state(
        source,
        slot_name="wkv_matrix_state",
        alternative_slot="shift_state",
        normalization="swap_state_slots",
    )
    assert np.array_equal(
        result["normalized_array"], np.ones((1, 1, 2), dtype=np.float32)
    )


def test_pre_post_convention_candidate_report_is_deterministic() -> None:
    left = np.zeros((1, 1, 1, 2, 2), dtype=np.float32)
    right = np.zeros((1, 1, 1, 2, 2), dtype=np.float32)
    first = compare_wkv_matrix_state_conventions(left, right)
    second = compare_wkv_matrix_state_conventions(left, right)
    assert first["candidate_normalizations"] == second["candidate_normalizations"]
    assert first["schema"] == WKV_STATE_CONVENTION_REPORT_SCHEMA


def test_normalized_comparison_reports_raw_and_normalized_errors() -> None:
    left = np.zeros((1, 1, 1, 2, 2), dtype=np.float32)
    right = left.copy()
    right[0, 0, 0, 0, 0] = 0.25
    report = compare_wkv_matrix_state_conventions(
        left,
        right,
        normalization="as_is",
    )
    assert report["raw_wkv_matrix_state_error"]["max_abs_error"] == 0.25
    assert report["normalized_wkv_matrix_state_error"]["max_abs_error"] == 0.25
    assert report["normalization_applied"] == "as_is"


def test_script_help_works_for_inspection_and_normalized_comparison_scripts() -> None:
    for script in (INSPECT_SCRIPT, COMPARE_SCRIPT):
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "P61" in result.stdout
        assert "WKV" in result.stdout


def test_normalized_comparison_end_to_end(tmp_path: Path) -> None:
    fixture_manifest, radlads_out, qrwkv_out, p60_report = _prepare_fixture_bundle(
        tmp_path
    )
    slot_audit_dir = tmp_path / "slot_audit"
    subprocess.run(
        [
            sys.executable,
            str(INSPECT_SCRIPT),
            "--fixture-manifest",
            str(fixture_manifest),
            "--radlads-outputs",
            str(radlads_out),
            "--qrwkv-outputs",
            str(qrwkv_out),
            "--p60-report",
            str(p60_report),
            "--out",
            str(slot_audit_dir),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    comparison_out = tmp_path / "comparison"
    subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            "--manifest",
            str(fixture_manifest),
            "--radlads-outputs",
            str(radlads_out),
            "--qrwkv-outputs",
            str(qrwkv_out),
            "--slot-audit",
            str(slot_audit_dir / "wkv_state_slot_audit.json"),
            "--p60-report",
            str(p60_report),
            "--out",
            str(comparison_out),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(
        (comparison_out / "head_to_head_normalized_state_report.json").read_text()
    )
    assert report["schema"] == "radlads_qrwkv_wkv_state_convention_report.v1"
    assert "raw_wkv_matrix_state_error" in report
    assert "normalized_wkv_matrix_state_error" in report
    assert (comparison_out / "P61_SURFACE_COMPARISON.md").is_file()
