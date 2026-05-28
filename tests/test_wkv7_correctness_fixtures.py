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
from qrwkv_xla.students import (
    build_pallas_runtime_probe,
    pallas_wkv_sequence_parity_cases,
    pallas_wkv_sequence_update_fused_or_scan,
    pallas_wkv_sequence_update_repeated,
    pallas_wkv_shape_dtype_parity_cases,
    pallas_wkv_update,
    reference_wkv_sequence_update,
    reference_wkv_update,
    run_pallas_wkv_fused_sequence_parity_matrix,
    run_pallas_wkv_parity_probe,
    run_pallas_wkv_sequence_parity_matrix,
    run_pallas_wkv_shape_dtype_parity_matrix,
)

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


def test_p83_reference_wkv_update_matches_formula() -> None:
    state = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    k = np.array([[[2.0, 3.0]]], dtype=np.float32)
    v = np.array([[[5.0, 7.0]]], dtype=np.float32)
    decay = np.array([[[0.5, 0.25]]], dtype=np.float32)

    expected = state * decay[..., None, :] + k[..., :, None] * v[..., None, :]
    np.testing.assert_allclose(reference_wkv_update(state, k, v, decay), expected)


def test_p83_pallas_update_matches_reference_or_reports_unavailable() -> None:
    state = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    k = np.array([[[2.0, 3.0]]], dtype=np.float32)
    v = np.array([[[5.0, 7.0]]], dtype=np.float32)
    decay = np.array([[[0.5, 0.25]]], dtype=np.float32)

    probe = run_pallas_wkv_parity_probe()
    assert probe["parity_scope"] == "tiny_one_step_wkv_update"
    assert probe["kernel_parity_claimed"] is (probe["parity_status"] == "pass")
    if probe["parity_status"] == "pass":
        np.testing.assert_allclose(
            pallas_wkv_update(state, k, v, decay),
            reference_wkv_update(state, k, v, decay),
            atol=probe["atol"],
            rtol=probe["rtol"],
        )
    else:
        assert probe["reason"]


def test_p83_pallas_probe_claims_parity_only_on_pass() -> None:
    probe = build_pallas_runtime_probe(requested="pallas")

    assert probe["schema"] == "qrwkv_xla.p83_pallas_wkv_parity_probe.v1"
    assert probe["phase"] == "P83"
    assert probe["default_runtime"] == "reference"
    assert probe["allowed_runtimes"] == ["reference", "pallas"]
    assert probe["wkv_runtime_requested"] == "pallas"
    assert probe["fallback_used"] is False
    assert probe["prototype_status"] in {"pass", "unavailable", "failed"}
    assert probe["parity_status"] in {"pass", "unavailable", "failed"}
    assert probe["parity_scope"] == "tiny_one_step_wkv_update"
    matrix = probe["p86_pallas_fused_sequence_parity_matrix"]
    assert (
        probe["kernel_parity_claimed"] is (matrix["summary"]["kernel_parity_claimed"])
    )
    if probe["prototype_status"] == "pass":
        assert probe["wkv_runtime_effective"] == "pallas"
        assert probe["pallas_available"] is True
        assert probe["finite"] is True
        assert probe["shape_match"] is True
        assert probe["max_abs_error"] <= probe["atol"]
        assert probe["max_rel_error"] <= probe["rtol"]
        assert probe["probe_shapes"]["state"] == [1, 1, 2, 2]
    else:
        assert probe["wkv_runtime_effective"] == "unavailable"


def test_p84_shape_dtype_parity_cases_include_required_float32_surface() -> None:
    cases = pallas_wkv_shape_dtype_parity_cases()
    required = {case.case_id for case in cases if case.required}

    assert required == {
        "float32_b1_h1_d2",
        "float32_b1_h2_d2",
        "float32_b2_h1_d2",
        "float32_b1_h1_d4",
        "float32_b2_h2_d4",
    }
    assert {case.dtype for case in cases if not case.required} == {"bfloat16"}


def test_p84_shape_dtype_parity_matrix_claims_only_required_passes() -> None:
    matrix = run_pallas_wkv_shape_dtype_parity_matrix()

    assert matrix["schema"] == "qrwkv_xla.p84_pallas_shape_dtype_parity_matrix.v1"
    assert matrix["phase"] == "P84"
    assert matrix["parity_scope"] == "broader_one_step_wkv_shape_dtype"
    assert matrix["default_runtime"] == "reference"
    assert matrix["wkv_runtime_requested"] == "pallas"
    assert matrix["summary"]["cases_total"] == len(matrix["cases"])
    assert (
        matrix["kernel_parity_claimed"]
        is (matrix["summary"]["all_required_cases_pass"])
    )
    assert (
        matrix["summary"]["kernel_parity_claimed"]
        is (matrix["summary"]["all_required_cases_pass"])
    )

    required_rows = [case for case in matrix["cases"] if case["required"]]
    assert {case["case_id"] for case in required_rows} == {
        "float32_b1_h1_d2",
        "float32_b1_h2_d2",
        "float32_b2_h1_d2",
        "float32_b1_h1_d4",
        "float32_b2_h2_d4",
    }
    for case in matrix["cases"]:
        assert case["state_shape"] == [
            case["batch"],
            case["heads"],
            case["dim"],
            case["dim"],
        ]
        assert case["k_shape"] == [case["batch"], case["heads"], case["dim"]]
        assert case["v_shape"] == [case["batch"], case["heads"], case["dim"]]
        assert case["decay_shape"] == [case["batch"], case["heads"], case["dim"]]
        assert case["parity_status"] in {"pass", "fail", "unavailable"}
        assert "finite" in case
        assert "shape_match" in case
        assert "max_abs_error" in case
        assert "max_rel_error" in case
        assert "atol" in case
        assert "rtol" in case
        if case["required"] and matrix["pallas_available"]:
            assert case["dtype"] == "float32"
            assert case["parity_status"] == "pass"
            assert case["finite"] is True
            assert case["shape_match"] is True
            assert case["max_abs_error"] <= case["atol"]
            assert case["max_rel_error"] <= case["rtol"]
        if case["dtype"] == "bfloat16" and case["parity_status"] == "unavailable":
            assert case["reason"]


def test_p85_reference_sequence_update_matches_repeated_formula() -> None:
    initial_state = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    k_seq = np.array([[[[1.0, 2.0]]], [[[3.0, 4.0]]]], dtype=np.float32)
    v_seq = np.array([[[[2.0, 3.0]]], [[[4.0, 5.0]]]], dtype=np.float32)
    decay_seq = np.array([[[[0.5, 0.25]]], [[[0.75, 0.5]]]], dtype=np.float32)

    first = initial_state * decay_seq[0, ..., None, :] + (
        k_seq[0, ..., :, None] * v_seq[0, ..., None, :]
    )
    second = first * decay_seq[1, ..., None, :] + (
        k_seq[1, ..., :, None] * v_seq[1, ..., None, :]
    )
    result = reference_wkv_sequence_update(initial_state, k_seq, v_seq, decay_seq)

    np.testing.assert_allclose(result["final_state"], second)
    np.testing.assert_allclose(result["per_step_states"], np.stack([first, second]))


def test_p85_pallas_sequence_update_matches_reference_or_reports_unavailable() -> None:
    initial_state = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    k_seq = np.array([[[[1.0, 2.0]]], [[[3.0, 4.0]]]], dtype=np.float32)
    v_seq = np.array([[[[2.0, 3.0]]], [[[4.0, 5.0]]]], dtype=np.float32)
    decay_seq = np.array([[[[0.5, 0.25]]], [[[0.75, 0.5]]]], dtype=np.float32)
    matrix = run_pallas_wkv_sequence_parity_matrix()

    if matrix["pallas_available"]:
        pallas = pallas_wkv_sequence_update_repeated(
            initial_state,
            k_seq,
            v_seq,
            decay_seq,
        )
        reference = reference_wkv_sequence_update(
            initial_state,
            k_seq,
            v_seq,
            decay_seq,
        )
        np.testing.assert_allclose(
            pallas["final_state"],
            reference["final_state"],
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            pallas["per_step_states"],
            reference["per_step_states"],
            atol=1e-6,
            rtol=1e-6,
        )
    else:
        assert matrix["reason"]


def test_p85_sequence_parity_cases_include_required_float32_surface() -> None:
    cases = pallas_wkv_sequence_parity_cases()
    required = {case.case_id for case in cases if case.required}

    assert required == {
        "float32_b1_h1_d2_t2",
        "float32_b1_h1_d2_t4",
        "float32_b1_h2_d2_t4",
        "float32_b2_h2_d4_t4",
    }
    assert {case.dtype for case in cases if not case.required} == {"bfloat16"}


def test_p85_sequence_parity_matrix_claims_only_required_passes() -> None:
    matrix = run_pallas_wkv_sequence_parity_matrix()

    assert matrix["schema"] == "qrwkv_xla.p85_pallas_sequence_parity_matrix.v1"
    assert matrix["phase"] == "P85"
    assert matrix["parity_scope"] == "short_sequence_repeated_one_step_wkv"
    assert matrix["sequence_method"] == "repeated_one_step_pallas"
    assert matrix["default_runtime"] == "reference"
    assert matrix["wkv_runtime_requested"] == "pallas"
    assert matrix["summary"]["cases_total"] == len(matrix["cases"])
    assert (
        matrix["kernel_parity_claimed"]
        is (matrix["summary"]["all_required_cases_pass"])
    )

    required_rows = [case for case in matrix["cases"] if case["required"]]
    assert {case["case_id"] for case in required_rows} == {
        "float32_b1_h1_d2_t2",
        "float32_b1_h1_d2_t4",
        "float32_b1_h2_d2_t4",
        "float32_b2_h2_d4_t4",
    }
    for case in matrix["cases"]:
        assert case["initial_state_shape"] == [
            case["batch"],
            case["heads"],
            case["dim"],
            case["dim"],
        ]
        seq_shape = [case["seq_len"], case["batch"], case["heads"], case["dim"]]
        assert case["k_seq_shape"] == seq_shape
        assert case["v_seq_shape"] == seq_shape
        assert case["decay_seq_shape"] == seq_shape
        assert case["parity_status"] in {"pass", "fail", "unavailable"}
        for field in {
            "final_max_abs_error",
            "final_max_rel_error",
            "worst_step_max_abs_error",
            "worst_step_max_rel_error",
            "atol",
            "rtol",
        }:
            assert field in case
        if case["required"] and matrix["pallas_available"]:
            assert case["dtype"] == "float32"
            assert case["parity_status"] == "pass"
            assert case["final_shape_match"] is True
            assert case["per_step_shape_match"] is True
            assert case["final_state_finite"] is True
            assert case["per_step_finite"] is True
            assert case["final_max_abs_error"] <= case["atol"]
            assert case["final_max_rel_error"] <= case["rtol"]
            assert case["worst_step_max_abs_error"] <= case["atol"]
            assert case["worst_step_max_rel_error"] <= case["rtol"]
        if case["dtype"] == "bfloat16" and case["parity_status"] == "unavailable":
            assert case["reason"]


def test_p86_fused_or_scan_sequence_update_matches_reference_or_unavailable() -> None:
    initial_state = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2) / 3.0
    k_seq = np.array([[[[0.1, 0.2]]], [[[0.3, 0.4]]]], dtype=np.float32)
    v_seq = np.array([[[[0.2, 0.3]]], [[[0.4, 0.5]]]], dtype=np.float32)
    decay_seq = np.array([[[[0.5, 0.25]]], [[[0.75, 0.5]]]], dtype=np.float32)
    matrix = run_pallas_wkv_fused_sequence_parity_matrix()

    if matrix["pallas_available"]:
        pallas = pallas_wkv_sequence_update_fused_or_scan(
            initial_state,
            k_seq,
            v_seq,
            decay_seq,
        )
        reference = reference_wkv_sequence_update(
            initial_state,
            k_seq,
            v_seq,
            decay_seq,
        )
        assert pallas["sequence_method"] == "jax_scan_pallas_step_scaffold"
        assert pallas["fused_sequence_kernel_status"] == "scan_scaffold_pass"
        np.testing.assert_allclose(
            pallas["final_state"],
            reference["final_state"],
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            pallas["per_step_states"],
            reference["per_step_states"],
            atol=1e-6,
            rtol=1e-6,
        )
    else:
        assert matrix["reason"]


def test_p86_fused_sequence_parity_matrix_claims_only_required_passes() -> None:
    matrix = run_pallas_wkv_fused_sequence_parity_matrix()

    assert matrix["schema"] == "qrwkv_xla.p86_pallas_fused_sequence_parity_matrix.v1"
    assert matrix["phase"] == "P86"
    assert matrix["parity_scope"] == "fused_or_scan_style_wkv_sequence"
    assert matrix["sequence_method"] in {
        "jax_scan_pallas_step_scaffold",
        "fused_sequence_scaffold_unavailable",
    }
    assert matrix["default_runtime"] == "reference"
    assert matrix["wkv_runtime_requested"] == "pallas"
    assert matrix["summary"]["cases_total"] == len(matrix["cases"])
    assert (
        matrix["kernel_parity_claimed"]
        is (matrix["summary"]["all_required_cases_pass"])
    )
    assert matrix["p85_repeated_step_parity"]["schema"] == (
        "qrwkv_xla.p85_pallas_sequence_parity_matrix.v1"
    )

    required_rows = [case for case in matrix["cases"] if case["required"]]
    assert {case["case_id"] for case in required_rows} == {
        "float32_b1_h1_d2_t2",
        "float32_b1_h1_d2_t4",
        "float32_b1_h2_d2_t4",
        "float32_b2_h2_d4_t4",
    }
    for case in matrix["cases"]:
        assert case["sequence_method"] in {
            "jax_scan_pallas_step_scaffold",
            "fused_sequence_scaffold_unavailable",
        }
        assert case["fused_sequence_kernel_status"] in {
            "scan_scaffold_pass",
            "fused_sequence_scaffold_unavailable",
        }
        assert case["initial_state_shape"] == [
            case["batch"],
            case["heads"],
            case["dim"],
            case["dim"],
        ]
        seq_shape = [case["seq_len"], case["batch"], case["heads"], case["dim"]]
        assert case["k_seq_shape"] == seq_shape
        assert case["v_seq_shape"] == seq_shape
        assert case["decay_seq_shape"] == seq_shape
        assert case["parity_status"] in {"pass", "fail", "unavailable"}
        for field in {
            "final_max_abs_error",
            "final_max_rel_error",
            "worst_step_max_abs_error",
            "worst_step_max_rel_error",
            "atol",
            "rtol",
        }:
            assert field in case
        if case["required"] and matrix["pallas_available"]:
            assert case["dtype"] == "float32"
            assert case["parity_status"] == "pass"
            assert case["final_shape_match"] is True
            assert case["per_step_shape_match"] is True
            assert case["final_state_finite"] is True
            assert case["per_step_finite"] is True
            assert case["final_max_abs_error"] <= case["atol"]
            assert case["final_max_rel_error"] <= case["rtol"]
            assert case["worst_step_max_abs_error"] <= case["atol"]
            assert case["worst_step_max_rel_error"] <= case["rtol"]
        if case["dtype"] == "bfloat16" and case["parity_status"] == "unavailable":
            assert case["reason"]


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
