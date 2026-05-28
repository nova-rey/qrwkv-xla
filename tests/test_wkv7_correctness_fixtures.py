from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from qrwkv_xla.kernels import (
    compare_wkv7_manifest,
    generate_wkv7_fixture_bundle,
    load_wkv7_case,
    validate_wkv7_manifest,
    wkv7_reference_full_scan,
    wkv7_reference_stepwise,
    write_wkv7_comparison_reports,
)
from qrwkv_xla.kernels.wkv7_fixtures import DEFAULT_CASES, hash_arrays
from qrwkv_xla.students import build_pallas_runtime_probe

ROOT = Path(__file__).resolve().parents[1]


def test_wkv7_fixture_generation_schema_and_payloads(tmp_path: Path) -> None:
    out = tmp_path / "wkv7"
    manifest = generate_wkv7_fixture_bundle(out, overwrite=True)
    validated = validate_wkv7_manifest(out / "manifest.json")

    assert validated["schema_version"] == "0.1"
    assert validated["phase"] == "P43"
    assert validated["fixture_set"] == "tiny_wkv7_correctness"
    assert validated["surface"]["implementation_surface"] == "wkv7_recurrence_core"
    assert validated["source"]["implementation"] == "rwkv7_qwen_reference"
    assert {case["case_id"] for case in manifest["cases"]} == set(DEFAULT_CASES)
    assert (out / "P43_WKV7_FIXTURE_SUMMARY.md").is_file()

    for case in validated["cases"]:
        inputs, expected = load_wkv7_case(out / "manifest.json", case)
        assert case["inputs_sha256"] == hash_arrays(inputs)
        assert case["expected_sha256"] == hash_arrays(expected)
        assert expected["output"].shape == tuple(case["shapes"]["output"])
        assert expected["next_state"].shape == tuple(case["shapes"]["next_state"])
        assert np.all(np.isfinite(expected["output"]))
        assert np.all(np.isfinite(expected["next_state"]))
        assert case["full_scan_vs_stepwise"]["status"] == "pass"


def test_wkv7_fixture_generation_is_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    manifest_a = generate_wkv7_fixture_bundle(out_a, overwrite=True)
    manifest_b = generate_wkv7_fixture_bundle(out_b, overwrite=True)

    by_case_a = {case["case_id"]: case for case in manifest_a["cases"]}
    by_case_b = {case["case_id"]: case for case in manifest_b["cases"]}
    assert set(by_case_a) == set(by_case_b)
    for case_id in by_case_a:
        assert (
            by_case_a[case_id]["inputs_sha256"] == by_case_b[case_id]["inputs_sha256"]
        )
        assert (
            by_case_a[case_id]["expected_sha256"]
            == by_case_b[case_id]["expected_sha256"]
        )


def test_wkv7_reference_full_scan_matches_stepwise(tmp_path: Path) -> None:
    out = tmp_path / "wkv7"
    manifest = generate_wkv7_fixture_bundle(out, overwrite=True)

    for case in manifest["cases"]:
        inputs, expected = load_wkv7_case(out / "manifest.json", case)
        full = wkv7_reference_full_scan(inputs)
        stepwise = wkv7_reference_stepwise(inputs)
        np.testing.assert_allclose(full["output"], expected["output"], atol=1e-5)
        np.testing.assert_allclose(
            full["next_state"], expected["next_state"], atol=1e-5
        )
        np.testing.assert_allclose(
            full["output"], stepwise["stepwise_output"], atol=1e-5
        )
        np.testing.assert_allclose(
            full["next_state"], stepwise["stepwise_next_state"], atol=1e-5
        )


def test_wkv7_reference_candidate_comparison_passes(tmp_path: Path) -> None:
    out = tmp_path / "wkv7"
    generate_wkv7_fixture_bundle(out, overwrite=True)

    report = write_wkv7_comparison_reports(
        out / "manifest.json", out, candidate="reference", overwrite=True
    )

    assert report["overall_status"] == "pass"
    assert report["num_cases"] == len(DEFAULT_CASES)
    assert report["counts"]["pass"] == len(DEFAULT_CASES)
    assert (out / "comparison_report.json").is_file()
    assert (out / "P43_WKV7_COMPARISON_REPORT.md").is_file()
    persisted = json.loads((out / "comparison_report.json").read_text())
    assert persisted["status_vocabulary"] == [
        "candidate_error",
        "dtype_mismatch",
        "fail",
        "missing_fixture",
        "non_finite",
        "pass",
        "shape_mismatch",
        "unsupported",
    ]


def test_wkv7_comparator_reports_fail_for_perturbed_expected_arrays(
    tmp_path: Path,
) -> None:
    out = tmp_path / "wkv7"
    manifest = generate_wkv7_fixture_bundle(out, overwrite=True)
    case = manifest["cases"][0]
    _, expected = load_wkv7_case(out / "manifest.json", case)
    expected["output"][0, 0, 0, 0] += 0.5
    np.savez(out / case["paths"]["expected"], **expected)

    report = compare_wkv7_manifest(out / "manifest.json", candidate="reference")

    assert report["overall_status"] == "fail"
    assert report["counts"]["fail"] == 1
    failed = next(item for item in report["cases"] if item["status"] == "fail")
    assert failed["output"]["max_abs_error"] > 0.1


def test_wkv7_comparator_reports_shape_mismatch(tmp_path: Path) -> None:
    out = tmp_path / "wkv7"
    manifest = generate_wkv7_fixture_bundle(out, overwrite=True)
    case = manifest["cases"][0]
    _, expected = load_wkv7_case(out / "manifest.json", case)
    expected["output"] = expected["output"][..., :3]
    np.savez(out / case["paths"]["expected"], **expected)

    report = compare_wkv7_manifest(out / "manifest.json", candidate="reference")

    assert report["overall_status"] == "shape_mismatch"
    mismatch = next(
        item for item in report["cases"] if item["status"] == "shape_mismatch"
    )
    assert mismatch["output"]["status"] == "shape_mismatch"


def test_wkv7_comparator_reports_non_finite(tmp_path: Path) -> None:
    out = tmp_path / "wkv7"
    manifest = generate_wkv7_fixture_bundle(out, overwrite=True)
    case = manifest["cases"][0]
    _, expected = load_wkv7_case(out / "manifest.json", case)
    expected["next_state"][0, 0, 0, 0] = np.nan
    np.savez(out / case["paths"]["expected"], **expected)

    report = compare_wkv7_manifest(out / "manifest.json", candidate="reference")

    assert report["overall_status"] == "non_finite"
    non_finite = next(
        item for item in report["cases"] if item["status"] == "non_finite"
    )
    assert non_finite["next_state"]["status"] == "non_finite"


def test_wkv7_missing_fixture_is_reported(tmp_path: Path) -> None:
    out = tmp_path / "wkv7"
    manifest = generate_wkv7_fixture_bundle(out, overwrite=True)
    case = manifest["cases"][0]
    (out / case["paths"]["expected"]).unlink()

    report = compare_wkv7_manifest(out / "manifest.json", candidate="reference")

    assert report["overall_status"] == "missing_fixture"
    assert report["counts"]["missing_fixture"] == 1


def test_wkv7_pallas_candidate_is_explicitly_unsupported(tmp_path: Path) -> None:
    out = tmp_path / "wkv7"
    generate_wkv7_fixture_bundle(out, overwrite=True)

    report = compare_wkv7_manifest(out / "manifest.json", candidate="pallas")

    assert report["overall_status"] == "unsupported"
    assert report["counts"]["unsupported"] == len(DEFAULT_CASES)
    assert all(case["status"] == "unsupported" for case in report["cases"])
    assert "not implemented yet" in report["cases"][0]["reason"]


def test_p82_pallas_probe_does_not_claim_kernel_parity() -> None:
    probe = build_pallas_runtime_probe(requested="pallas")

    assert probe["schema"] == "qrwkv_xla.p82_pallas_runtime_probe.v1"
    assert probe["phase"] == "P82"
    assert probe["default_runtime"] == "reference"
    assert probe["allowed_runtimes"] == ["reference", "pallas"]
    assert probe["wkv_runtime_requested"] == "pallas"
    assert probe["fallback_used"] is False
    assert probe["prototype_status"] in {"pass", "unavailable", "failed"}
    assert probe["kernel_parity_claimed"] is False
    if probe["prototype_status"] == "pass":
        assert probe["wkv_runtime_effective"] == "pallas"
        assert probe["pallas_available"] is True
        assert probe["finite"] is True
        assert probe["probe_shapes"]["state"] == [1, 1, 2, 2]
    else:
        assert probe["wkv_runtime_effective"] == "unavailable"


def test_wkv7_script_help_and_smoke(tmp_path: Path) -> None:
    generate_help = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_wkv7_correctness_fixtures.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    compare_help = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_wkv7_correctness_fixtures.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (
        "Generate P43 tiny deterministic WKV7 correctness fixtures"
        in generate_help.stdout
    )
    assert (
        "Compare a WKV7 candidate against P43 correctness fixtures"
        in compare_help.stdout
    )

    out = tmp_path / "scripted"
    generate = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_wkv7_correctness_fixtures.py"),
            "--out",
            str(out),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert f"wrote {len(DEFAULT_CASES)} WKV7 correctness cases" in generate.stdout

    compare = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_wkv7_correctness_fixtures.py"),
            "--manifest",
            str(out / "manifest.json"),
            "--candidate",
            "reference",
            "--out",
            str(out),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "overall_status=pass" in compare.stdout
