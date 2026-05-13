from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.parity import (
    REQUIRED_NUMERICAL_CASE_NAMES,
    compare_numerical_manifest,
    compare_parameter_surfaces,
    generate_radlads_tiny_numerical_fixtures,
    load_numerical_case_arrays,
    validate_numerical_manifest,
    write_current_behavior_numerical_fixtures,
    write_numerical_comparison_reports,
)
from qrwkv_xla.parity.radlads_numerical_fixtures import (
    PARAMETER_EXTREME_THRESHOLD,
    hash_numerical_arrays,
    load_parameter_arrays,
)

ROOT = Path(__file__).resolve().parents[1]


def test_current_behavior_manifest_is_valid_and_honest(tmp_path: Path) -> None:
    out = tmp_path / "fixtures"
    manifest = write_current_behavior_numerical_fixtures(out, overwrite=True)

    assert manifest["schema"] == "radlads_tiny_numerical_parity.v1"
    assert manifest["phase"] == "P49"
    assert manifest["real_radlads_fixture_status"] == "source_unavailable"
    assert set(case["name"] for case in manifest["cases"]) == set(
        REQUIRED_NUMERICAL_CASE_NAMES
    )
    assert {case["status"] for case in manifest["cases"]} == {"missing_source"}
    assert "not RADLADS outputs" in manifest["claim"]

    for case in manifest["cases"]:
        arrays = load_numerical_case_arrays(out / "manifest.json", case)
        assert case["payload_sha256"] == hash_numerical_arrays(arrays)
        assert arrays["input_ids"].shape == (2, 4)


def test_deterministic_finite_init_writes_clean_parameter_payload(
    tmp_path: Path,
) -> None:
    out = tmp_path / "fixtures"

    manifest = generate_radlads_tiny_numerical_fixtures(
        out,
        overwrite=True,
        init_policy="deterministic_finite",
    )
    parameters = load_parameter_arrays(out / "manifest.json")
    max_abs = max(float(np.max(np.abs(value))) for value in parameters.values())

    assert manifest["parameter_payload"] == "radlads_parameters.npz"
    assert manifest["parameter_payload_init_policy"] == "deterministic_finite"
    assert manifest["parameter_payload_source"] == "deterministic_finite"
    assert manifest["parameter_payload_validation"]["status"] == "clean"
    assert manifest["parameter_payload_validation"]["deterministic"] is True
    assert manifest["parameter_payload_validation"]["finite"] is True
    assert parameters
    assert all(np.all(np.isfinite(value)) for value in parameters.values())
    assert max_abs < PARAMETER_EXTREME_THRESHOLD


def test_parameter_mapping_statuses() -> None:
    report = compare_parameter_surfaces(
        {
            "same.weight": (2, 2),
            "embed_tokens.weight": (3, 4),
            "bad_shape.weight": (1,),
            "missing.weight": (5,),
            "layers.*.self_attn.gate": (6,),
        },
        {
            "same.weight": (2, 2),
            "token_embedding.weight": (3, 4),
            "bad_shape.weight": (2,),
            "qrwkv_only.weight": (7,),
        },
    )

    rows = {
        (row.get("radlads"), row.get("qrwkv")): row["status"]
        for row in report["mappings"]
    }
    assert rows[("same.weight", "same.weight")] == "mapped_exact"
    assert rows[("embed_tokens.weight", "token_embedding.weight")] == "mapped_renamed"
    assert rows[("bad_shape.weight", "bad_shape.weight")] == "shape_mismatch"
    assert rows[("missing.weight", "missing.weight")] == "missing_in_qrwkv"
    assert rows[("layers.*.self_attn.gate", "layers.*.self_attn.gate")] == "unsupported"
    assert rows[(None, "qrwkv_only.weight")] == "missing_in_radlads"
    assert compare_parameter_surfaces(None, {})["overall_status"] == "source_not_found"


def test_comparator_pass_fail_and_missing_source(tmp_path: Path) -> None:
    manifest_dir = _write_comparison_fixture_set(tmp_path)

    report = compare_numerical_manifest(manifest_dir / "manifest.json")

    assert report["overall_status"] == "fail"
    assert report["counts"] == {
        "pass": 1,
        "fail": 1,
        "unsupported": 1,
        "missing_source": 2,
    }
    by_name = {case["name"]: case for case in report["cases"]}
    assert by_name["tiny_no_mask"]["status"] == "pass"
    assert by_name["tiny_attention_mask"]["status"] == "fail"
    assert by_name["tiny_prefix_or_left_padding"]["status"] == "unsupported"
    assert by_name["tiny_stepwise_state"]["status"] == "missing_source"


def test_report_writing_outputs_required_files(tmp_path: Path) -> None:
    manifest_dir = _write_comparison_fixture_set(tmp_path / "fixture_root")
    out = tmp_path / "reports"

    report = write_numerical_comparison_reports(manifest_dir / "manifest.json", out)

    assert report["overall_status"] == "fail"
    assert (out / "P49_RESULTS.md").is_file()
    assert (out / "numerical_parity_report.json").is_file()
    assert (out / "P49_SURFACE_COMPARISON.md").is_file()
    assert (out / "surface_comparison.json").is_file()
    assert (
        json.loads((out / "numerical_parity_report.json").read_text())["counts"]["fail"]
        == 1
    )


def test_cli_help_and_smoke(tmp_path: Path) -> None:
    for script in (
        "generate_radlads_tiny_numerical_fixtures.py",
        "import_radlads_tiny_numerical_fixtures.py",
        "compare_radlads_tiny_numerical_fixtures.py",
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "P49" in result.stdout or "RADLADS" in result.stdout

    fixture_out = tmp_path / "fixtures"
    import_result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "import_radlads_tiny_numerical_fixtures.py"),
            "--out",
            str(fixture_out),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "missing_source synthetic P49 fixtures" in import_result.stdout
    validate_numerical_manifest(fixture_out)

    report_out = tmp_path / "reports"
    compare_result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_radlads_tiny_numerical_fixtures.py"),
            "--manifest",
            str(fixture_out / "manifest.json"),
            "--out-dir",
            str(report_out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "overall_status=source_unavailable" in compare_result.stdout
    assert (report_out / "P49_RESULTS.md").is_file()


def test_env_gated_live_path_skips_without_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("QRWKV_RUN_RADLADS_LIVE", raising=False)

    manifest = generate_radlads_tiny_numerical_fixtures(
        tmp_path / "fixtures",
        overwrite=True,
        radlads_source_path=tmp_path / "does-not-exist",
    )

    assert os.environ.get("QRWKV_RUN_RADLADS_LIVE") != "1"
    assert manifest["real_radlads_fixture_status"] == "source_unavailable"
    assert {case["status"] for case in manifest["cases"]} == {"missing_source"}


def _write_comparison_fixture_set(tmp_path: Path) -> Path:
    out = tmp_path / "mixed"
    out.mkdir(parents=True)
    cases = []
    specs = {
        "tiny_no_mask": ("pass", 0.0, None),
        "tiny_attention_mask": ("pass", 0.25, None),
        "tiny_prefix_or_left_padding": ("unsupported", 0.0, "no supported source"),
        "tiny_stepwise_state": ("missing_source", 0.0, "source unavailable"),
        "tiny_all_radlads_math_enabled": ("missing_source", 0.0, "source unavailable"),
    }
    for name, (status, delta, reason) in specs.items():
        arrays = {
            "input_ids": np.array([[1, 2]], dtype=np.int32),
            "radlads_hidden_states": np.array([[1.0, 2.0]], dtype=np.float32),
            "qrwkv_hidden_states": np.array([[1.0 + delta, 2.0]], dtype=np.float32),
        }
        payload = f"{name}.npz"
        np.savez(out / payload, **arrays)
        case = {
            "name": name,
            "description": name,
            "surface_names": ["hidden"],
            "status": status,
            "payload": payload,
            "payload_sha256": hash_numerical_arrays(arrays),
            "input_shape": [1, 2],
            "attention_mask": {"present": False, "kind": "none", "shape": None},
            "shapes": {key: list(value.shape) for key, value in arrays.items()},
            "dtypes": {key: str(value.dtype) for key, value in arrays.items()},
            "comparisons": [
                {
                    "name": "hidden",
                    "left": "radlads_hidden_states",
                    "right": "qrwkv_hidden_states",
                    "atol": 1e-5,
                    "rtol": 1e-5,
                }
            ]
            if status == "pass"
            else [],
        }
        if status == "missing_source":
            case["missing_source_reason"] = reason
        if status == "unsupported":
            case["unsupported_reason"] = reason
        cases.append(case)

    manifest = {
        "schema_version": 1,
        "schema": "radlads_tiny_numerical_parity.v1",
        "phase": "P49",
        "source": "radlads",
        "backend": "rwkv7_qwen_reference",
        "radlads_commit": "test",
        "radlads_source_path": "test",
        "generation_script": "test",
        "created_at_utc": "2026-05-08T00:00:00Z",
        "dtype_policy": {"default": "float32"},
        "seed": 1,
        "required_cases": list(REQUIRED_NUMERICAL_CASE_NAMES),
        "real_radlads_fixture_status": "generated",
        "claim": "test",
        "parameter_mapping": compare_parameter_surfaces(
            {"embed_tokens.weight": (2, 2)},
            {"token_embedding.weight": (2, 2), "qrwkv_only.weight": (1,)},
        ),
        "cases": cases,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_numerical_manifest(out)
    return out
