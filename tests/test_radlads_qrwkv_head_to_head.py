from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from qrwkv_xla.parity.radlads_head_to_head import (
    HEAD_TO_HEAD_SCHEMA,
    compare_head_to_head_manifest,
    compare_surface_arrays,
    generate_head_to_head_fixtures,
    write_head_to_head_reports,
)

ROOT = Path(__file__).resolve().parents[1]


def test_head_to_head_generation_uses_clean_payload(tmp_path: Path) -> None:
    out = tmp_path / "p53"

    manifest = generate_head_to_head_fixtures(
        out,
        overwrite=True,
        radlads_source_path=tmp_path / "missing-radlads",
    )

    assert manifest["schema"] == HEAD_TO_HEAD_SCHEMA
    assert manifest["phase"] == "P53"
    assert manifest["seed"] == 5353
    assert manifest["parameter_validation"]["all_finite"] is True
    assert manifest["parameter_validation"]["non_finite_count"] == 0
    assert manifest["parameter_validation"]["extreme_count"] == 0
    assert manifest["radlads"]["available"] is False
    assert manifest["radlads"]["blocker"]
    assert manifest["qrwkv"]["available"] is True
    assert manifest["qrwkv"]["parameter_import"]["overall_status"] == "pass"
    assert (out / "fixtures_clean" / "manifest.json").is_file()
    assert (out / "radlads_parameters.npz").is_file()
    assert (out / "tiny_no_mask.npz").is_file()


def test_head_to_head_compare_reports_blocked_radlads_honestly(tmp_path: Path) -> None:
    out = tmp_path / "p53"
    generate_head_to_head_fixtures(
        out,
        overwrite=True,
        radlads_source_path=tmp_path / "missing-radlads",
    )

    report = compare_head_to_head_manifest(
        out / "manifest.json", out_dir=out / "comparison"
    )

    assert report["overall_status"] == "unsupported"
    assert report["attempted_comparisons"] == 0
    assert report["radlads_blocker"]
    assert report["surface_status_counts"]["unsupported"] > 0
    assert (out / "comparison" / "head_to_head_comparison_report.json").is_file()
    assert (out / "comparison" / "P53_RESULTS.md").is_file()
    assert (out / "comparison" / "P53_SURFACE_COMPARISON.md").is_file()
    payload = json.loads(
        (out / "comparison" / "head_to_head_comparison_report.json").read_text()
    )
    assert payload["overall_status"] == "unsupported"


def test_surface_comparison_helper_covers_core_statuses() -> None:
    a = np.ones((2, 3), dtype=np.float32)
    b = a.copy()
    fail = a + 1e-2

    assert compare_surface_arrays("pass", a, b)["status"] == "pass"
    assert compare_surface_arrays("fail", a, fail)["status"] == "fail"
    assert (
        compare_surface_arrays("shape", a, np.ones((3, 2), dtype=np.float32))["status"]
        == "shape_mismatch"
    )
    assert (
        compare_surface_arrays("dtype", a, b.astype(np.float16))["status"]
        == "dtype_mismatch"
    )
    assert (
        compare_surface_arrays(
            "left-nan",
            np.array([[np.nan]], dtype=np.float32),
            np.array([[1.0]], dtype=np.float32),
        )["status"]
        == "non_finite_radlads"
    )
    assert (
        compare_surface_arrays(
            "right-nan",
            np.array([[1.0]], dtype=np.float32),
            np.array([[np.nan]], dtype=np.float32),
        )["status"]
        == "non_finite_qrwkv"
    )


def test_report_writer_emits_json_and_markdown(tmp_path: Path) -> None:
    report = {
        "schema": HEAD_TO_HEAD_SCHEMA,
        "overall_status": "unsupported",
        "attempted_comparisons": 0,
        "radlads_blocker": "blocked",
        "best_passing_surface": None,
        "largest_failure": None,
        "surface_status_counts": {"unsupported": 1},
        "cases": [],
    }
    write_head_to_head_reports(report, tmp_path)

    assert (tmp_path / "head_to_head_comparison_report.json").is_file()
    assert (tmp_path / "P53_RESULTS.md").is_file()
    assert (tmp_path / "P53_SURFACE_COMPARISON.md").is_file()
    assert (
        json.loads((tmp_path / "head_to_head_comparison_report.json").read_text())[
            "schema"
        ]
        == HEAD_TO_HEAD_SCHEMA
    )


def test_head_to_head_cli_help() -> None:
    for script in (
        "generate_radlads_qrwkv_head_to_head_fixtures.py",
        "compare_radlads_qrwkv_head_to_head.py",
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "P53" in result.stdout
