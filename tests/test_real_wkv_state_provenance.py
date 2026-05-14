from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from qrwkv_xla.parity.radlads_clean_loader import export_qrwkv_clean_payload_outputs
from qrwkv_xla.parity.radlads_numerical_fixtures import (
    generate_radlads_tiny_numerical_fixtures,
)
from qrwkv_xla.parity.radlads_wkv_state_provenance import load_provenance_jsonl

ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "scripts" / "run_real_radlads_qrwkv_wkv_state_provenance.py"
COMPARE_SCRIPT = ROOT / "scripts" / "compare_real_radlads_qrwkv_wkv_state_provenance.py"


def _prepare_p60_fixture_dirs(tmp_path: Path) -> tuple[Path, Path]:
    fixture_root = tmp_path / "p54_confirmation" / "fixtures"
    generate_radlads_tiny_numerical_fixtures(
        fixture_root,
        overwrite=True,
        init_policy="deterministic_finite",
    )
    parameter_payload = fixture_root / "radlads_parameters.npz"

    qrwkv_out = tmp_path / "qrwkv_outputs"
    export_qrwkv_clean_payload_outputs(
        parameter_payload,
        qrwkv_out,
        overwrite=True,
    )

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

    qrwkv_manifest = json.loads((qrwkv_out / "manifest.json").read_text())
    qrwkv_manifest["side"] = "radlads"
    qrwkv_manifest["overall_status"] = "pass"
    qrwkv_manifest["notes"] = [
        "Synthesized RADLADS-shaped outputs for CI-only P60 test coverage.",
    ]
    (radlads_out / "manifest.json").write_text(
        json.dumps(qrwkv_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return radlads_out, qrwkv_out


def _run_p60(
    tmp_path: Path,
    radlads_out: Path,
    qrwkv_out: Path,
    *extra: str,
    modes: tuple[str, ...] = ("stepwise", "mask", "final"),
) -> Path:
    out = tmp_path / "p60"
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--out",
            str(out),
            "--overwrite",
            "--radlads-outputs",
            str(radlads_out),
            "--qrwkv-outputs",
            str(qrwkv_out),
            *[item for mode in modes for item in ("--mode", mode)],
            *extra,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "P60 real WKV state provenance" in result.stdout
    return out


def _compare_p60(out: Path, *extra: str) -> Path:
    comparison = out / "comparison"
    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            "--radlads-trace",
            str(out / "real_wkv_state_provenance_radlads.jsonl"),
            "--qrwkv-trace",
            str(out / "real_wkv_state_provenance_qrwkv.jsonl"),
            "--out",
            str(comparison),
            "--overwrite",
            *extra,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "P60 real WKV state provenance comparison" in result.stdout
    return comparison


def test_p60_runner_labels_cached_real_artifacts_and_valid_schema(
    tmp_path: Path,
) -> None:
    radlads_out, qrwkv_out = _prepare_p60_fixture_dirs(tmp_path)
    out = _run_p60(tmp_path, radlads_out, qrwkv_out)
    metadata = json.loads((out / "real_provenance_metadata.json").read_text())
    assert metadata["real_artifact_trace"] is True
    assert metadata["synthetic_trace"] is False
    assert metadata["derived_from_cached_outputs"] is True
    assert metadata["regenerated_live_outputs"] is False

    for filename in [
        "real_wkv_state_provenance_radlads.jsonl",
        "real_wkv_state_provenance_qrwkv.jsonl",
    ]:
        rows = load_provenance_jsonl(out / filename)
        assert rows
        assert all(row["real_artifact_trace"] is True for row in rows)
        assert all(row["synthetic_trace"] is False for row in rows)
        assert all(row["derived_from_cached_outputs"] is True for row in rows)

    for filename in [
        "P60_RESULTS.md",
        "TRACE_PROVENANCE.md",
        "TINY_NO_MASK_REAL_STATE.md",
        "TINY_STEPWISE_REAL_STATE.md",
        "REAL_MASK_PADDING_STATE.md",
        "HIDDEN_STATE_DEPENDENCY.md",
        "p60_real_state_provenance_report.json",
    ]:
        assert (out / filename).is_file()
    assert not (out / "P60_FIX_NOTE.md").exists()


def test_p60_strict_mode_fails_instead_of_synthetic_fallback(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--out",
            str(tmp_path / "strict"),
            "--strict-real-artifacts",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "will not synthesize replacements" in result.stderr


def test_p60_missing_path_failure(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--out",
            str(tmp_path / "missing"),
            "--radlads-outputs",
            str(tmp_path / "missing-radlads"),
            "--qrwkv-outputs",
            str(tmp_path / "missing-qrwkv"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "missing real artifact paths" in result.stderr


def test_p60_compare_deterministic_first_divergence(tmp_path: Path) -> None:
    radlads_out, qrwkv_out = _prepare_p60_fixture_dirs(tmp_path)
    out = _run_p60(tmp_path, radlads_out, qrwkv_out)
    comparison = _compare_p60(out)
    report = json.loads(
        (comparison / "p60_real_wkv_state_provenance_report.json").read_text()
    )
    assert report["overall_status"] == "pass"
    assert report["first_divergence"] is None
    assert report["first_mismatch"] is None
    assert (comparison / "P60_REAL_WKV_STATE_PROVENANCE.md").is_file()


def test_p60_case_and_mode_filtering(tmp_path: Path) -> None:
    radlads_out, qrwkv_out = _prepare_p60_fixture_dirs(tmp_path)
    out = _run_p60(
        tmp_path,
        radlads_out,
        qrwkv_out,
        "--case",
        "tiny_stepwise_state",
        "--mode",
        "stepwise",
        modes=("stepwise",),
    )
    radlads = load_provenance_jsonl(out / "real_wkv_state_provenance_radlads.jsonl")
    qrwkv = load_provenance_jsonl(out / "real_wkv_state_provenance_qrwkv.jsonl")
    assert {row["case"] for row in radlads + qrwkv} == {"tiny_stepwise_state"}
    assert {row["comparison"] for row in radlads + qrwkv} == {"full_vs_stepwise"}

    comparison = _compare_p60(out, "--case", "tiny_stepwise_state")
    report = json.loads(
        (comparison / "p60_real_wkv_state_provenance_report.json").read_text()
    )
    assert report["overall_status"] == "pass"
    assert report["case_counts"] == {"tiny_stepwise_state": report["row_count"]}


def test_p60_hidden_state_dependency_schema(tmp_path: Path) -> None:
    radlads_out, qrwkv_out = _prepare_p60_fixture_dirs(tmp_path)
    out = _run_p60(
        tmp_path,
        radlads_out,
        qrwkv_out,
        "--case",
        "tiny_stepwise_state",
        "--mode",
        "stepwise",
        modes=("stepwise",),
    )
    payload = json.loads((out / "hidden_state_dependency.json").read_text())
    assert payload["schema"] == "radlads_qrwkv_p60_hidden_state_dependency.v1"
    assert payload["hidden_state_rows"] == 2
    required = set(payload["required_fields"])
    assert {"real_artifact_trace", "derived_from_cached_outputs"} <= required
    assert all(row["state_name"] == "hidden_states" for row in payload["rows"])


def test_p60_script_help() -> None:
    for script in [RUN_SCRIPT, COMPARE_SCRIPT]:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "P60" in result.stdout
        assert "provenance" in result.stdout
