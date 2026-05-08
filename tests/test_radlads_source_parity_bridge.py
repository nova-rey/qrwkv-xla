from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.parity import (
    REQUIRED_CASE_NAMES,
    build_parameter_surface_map,
    compare_manifest,
    load_case_arrays,
    validate_manifest,
    write_comparison_reports,
    write_parameter_surface_map_reports,
)
from qrwkv_xla.parity.radlads_source import hash_arrays
from scripts.import_radlads_source_fixtures import write_current_behavior_fixtures

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "radlads_source_parity"


def test_checked_in_radlads_source_manifest_is_valid_and_honest() -> None:
    manifest = validate_manifest(FIXTURE_DIR)

    assert manifest["schema"] == "radlads_source_parity.v1"
    assert manifest["backend"] == "rwkv7_qwen_reference"
    assert manifest["source"]["head"] == "1b362eb"
    assert manifest["source"]["generation_mode"] == "qrwkv_current_behavior_only"
    assert {case["name"] for case in manifest["cases"]} == set(REQUIRED_CASE_NAMES)
    assert {case["status"] for case in manifest["cases"]} == {"unsupported"}
    assert "not RADLADS outputs" in manifest["claim"]

    for case in manifest["cases"]:
        arrays = load_case_arrays(FIXTURE_DIR / "manifest.json", case)
        assert case["payload_sha256"] == hash_arrays(arrays)
        assert arrays["input_ids"].shape == (2, 5)
        assert arrays["qrwkv_hidden_states"].shape == (2, 2, 5, 8)
        assert arrays["qrwkv_logits"].shape == (2, 5, 32)
        assert arrays["qrwkv_mixer_outputs"].shape == (2, 2, 5, 8)


def test_current_behavior_fixture_generation_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    write_current_behavior_fixtures(first, seed=4040, overwrite=True)
    write_current_behavior_fixtures(second, seed=4040, overwrite=True)

    assert _stable_manifest(first) == _stable_manifest(second)
    for case in validate_manifest(first)["cases"]:
        first_arrays = load_case_arrays(first / "manifest.json", case)
        second_arrays = load_case_arrays(
            second / "manifest.json",
            _case_by_name(second, case["name"]),
        )
        assert hash_arrays(first_arrays) == hash_arrays(second_arrays)


def test_manifest_validation_rejects_missing_arrays(tmp_path: Path) -> None:
    manifest_dir = _write_minimal_manifest(
        tmp_path,
        status="pass",
        arrays_for_first={
            "input_ids": np.array([[1, 2]], dtype=np.int32),
            "radlads_hidden_states": np.ones((1, 2, 3), dtype=np.float32),
        },
    )

    with pytest.raises(ValueError, match="missing array qrwkv_hidden_states"):
        validate_manifest(manifest_dir)


def test_manifest_validation_rejects_shape_mismatches(tmp_path: Path) -> None:
    manifest_dir = _write_minimal_manifest(
        tmp_path,
        status="pass",
        arrays_for_first={
            "input_ids": np.array([[1, 2]], dtype=np.int32),
            "radlads_hidden_states": np.ones((1, 2, 3), dtype=np.float32),
            "qrwkv_hidden_states": np.ones((1, 2, 4), dtype=np.float32),
        },
    )

    with pytest.raises(ValueError, match="comparison shape mismatch"):
        validate_manifest(manifest_dir)


def test_comparison_math_reports_pass_fail_and_unsupported(tmp_path: Path) -> None:
    manifest_dir = _write_mixed_comparison_manifest(tmp_path)

    report = compare_manifest(manifest_dir / "manifest.json")

    assert report["overall_status"] == "fail"
    assert report["counts"] == {"fail": 1, "pass": 1, "unsupported": 1}
    by_name = {case["name"]: case for case in report["cases"]}
    assert by_name["tiny_no_mask"]["status"] == "pass"
    assert by_name["tiny_attention_mask"]["status"] == "fail"
    failed = by_name["tiny_attention_mask"]["comparisons"][0]
    assert failed["max_abs"] == pytest.approx(0.25)
    assert failed["max_rel"] == pytest.approx(0.25)
    assert by_name["tiny_prefix_padding_or_left_padding"]["status"] == "unsupported"


def test_report_writing(tmp_path: Path) -> None:
    report = write_comparison_reports(FIXTURE_DIR / "manifest.json", tmp_path)

    assert report["overall_status"] == "unsupported"
    report_json = json.loads((tmp_path / "parity_report.json").read_text())
    report_md = (tmp_path / "P40_PARITY_REPORT.md").read_text(encoding="utf-8")
    assert report_json["counts"]["unsupported"] == 3
    assert "P40 RADLADS Source Parity Report" in report_md
    assert "tiny_no_mask" in report_md


def test_parameter_mapping_report_marks_p48_surfaces(tmp_path: Path) -> None:
    report = write_parameter_surface_map_reports(tmp_path)

    assert report == build_parameter_surface_map()
    assert report["counts"]["unsupported"] >= 1
    rows = {row["radlads"]: row for row in report["mappings"]}
    assert rows["layers.*.self_attn.q_proj.weight"]["status"].startswith("direct")
    assert rows["layers.*.self_attn.w0/w1/w2"]["status"] == ("represented_flagged_math")
    assert rows["layers.*.self_attn.k_k/k_a/r_k"]["status"] == (
        "partially_represented_flagged_math"
    )
    assert (tmp_path / "parameter_surface_map.json").is_file()
    assert "w0/w1/w2" in (tmp_path / "P40_PARAMETER_SURFACE_MAP.md").read_text(
        encoding="utf-8"
    )


def test_import_and_compare_scripts_smoke(tmp_path: Path) -> None:
    fixture_out = tmp_path / "fixtures"
    import_result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "import_radlads_source_fixtures.py"),
            "--out",
            str(fixture_out),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "unsupported current-behavior fixtures" in import_result.stdout
    validate_manifest(fixture_out)

    report_out = tmp_path / "reports"
    compare_result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_radlads_source_fixtures.py"),
            "--manifest",
            str(fixture_out / "manifest.json"),
            "--out-dir",
            str(report_out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "overall_status=unsupported" in compare_result.stdout
    assert (report_out / "parity_report.json").is_file()

    map_result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "map_radlads_parameter_surface.py"),
            "--out-dir",
            str(report_out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "parameter surface map" in map_result.stdout
    assert (report_out / "parameter_surface_map.json").is_file()


def _stable_manifest(path: Path) -> dict:
    manifest = validate_manifest(path)
    return {key: value for key, value in manifest.items() if key not in {"cases"}} | {
        "cases": [
            {key: value for key, value in case.items() if key != "payload_sha256"}
            for case in manifest["cases"]
        ]
    }


def _case_by_name(path: Path, name: str) -> dict:
    return next(
        case for case in validate_manifest(path)["cases"] if case["name"] == name
    )


def _write_mixed_comparison_manifest(tmp_path: Path) -> Path:
    out = tmp_path / "mixed"
    out.mkdir()
    case_payloads = {
        "tiny_no_mask": {
            "status": "pass",
            "arrays": {
                "input_ids": np.array([[1, 2]], dtype=np.int32),
                "radlads_hidden_states": np.array([[1.0, 2.0]], dtype=np.float32),
                "qrwkv_hidden_states": np.array([[1.0, 2.0]], dtype=np.float32),
            },
        },
        "tiny_attention_mask": {
            "status": "pass",
            "arrays": {
                "input_ids": np.array([[1, 2]], dtype=np.int32),
                "attention_mask": np.array([[1, 0]], dtype=np.int32),
                "radlads_hidden_states": np.array([[1.0, 1.0]], dtype=np.float32),
                "qrwkv_hidden_states": np.array([[1.25, 1.0]], dtype=np.float32),
            },
        },
        "tiny_prefix_padding_or_left_padding": {
            "status": "unsupported",
            "arrays": {
                "input_ids": np.array([[0, 1]], dtype=np.int32),
                "attention_mask": np.array([[0, 1]], dtype=np.int32),
            },
        },
    }
    cases = []
    for name, payload in case_payloads.items():
        arrays = payload["arrays"]
        np.savez(out / f"{name}.npz", **arrays)
        cases.append(_case(name, payload["status"], arrays))
    _write_manifest(out, cases)
    return out


def _write_minimal_manifest(
    tmp_path: Path,
    *,
    status: str,
    arrays_for_first: dict[str, np.ndarray],
) -> Path:
    out = tmp_path / "minimal"
    out.mkdir()
    cases = []
    for index, name in enumerate(REQUIRED_CASE_NAMES):
        arrays = (
            arrays_for_first
            if index == 0
            else {"input_ids": np.array([[1, 2]], dtype=np.int32)}
        )
        np.savez(out / f"{name}.npz", **arrays)
        cases.append(_case(name, status if index == 0 else "unsupported", arrays))
    _write_manifest(out, cases)
    return out


def _case(name: str, status: str, arrays: dict[str, np.ndarray]) -> dict:
    has_mask = "attention_mask" in arrays
    case = {
        "name": name,
        "description": name,
        "status": status,
        "unsupported_reason": "" if status != "unsupported" else "test unsupported",
        "payload": f"{name}.npz",
        "payload_sha256": hash_arrays(arrays),
        "input_shape": list(arrays["input_ids"].shape),
        "attention_mask": {
            "present": has_mask,
            "kind": "test",
            "shape": list(arrays["attention_mask"].shape) if has_mask else None,
        },
        "comparisons": [],
    }
    if status != "unsupported":
        case["comparisons"] = [
            {
                "name": "hidden_states",
                "left": "qrwkv_hidden_states",
                "right": "radlads_hidden_states",
                "atol": 1e-5,
                "rtol": 1e-5,
            }
        ]
    return case


def _write_manifest(out: Path, cases: list[dict]) -> None:
    manifest = {
        "fixture_version": 1,
        "schema": "radlads_source_parity.v1",
        "backend": "rwkv7_qwen_reference",
        "claim": "test manifest",
        "source": {"name": "RADLADS"},
        "cases": cases,
    }
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
