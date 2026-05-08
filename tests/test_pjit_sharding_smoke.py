from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import jax
import pytest

from qrwkv_xla.sharding.mesh import create_named_mesh
from qrwkv_xla.sharding.reports import write_p46_reports
from qrwkv_xla.sharding.smoke import run_pjit_sharding_smoke
from qrwkv_xla.sharding.specs import get_sharding_policy

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_mesh_info_records_jax_topology() -> None:
    _mesh, info = create_named_mesh(mesh_axis="data")

    assert info.backend == jax.default_backend()
    assert info.device_count >= 1
    assert info.local_device_count >= 1
    assert info.mesh_axis_names == ("data",)
    assert info.mesh_shape == (info.device_count,)
    assert info.device_kinds
    assert info.multi_device is (info.device_count >= 2)


def test_single_device_fallback_is_honest_when_applicable() -> None:
    _mesh, info = create_named_mesh(mesh_axis="data")

    if info.device_count == 1:
        assert info.multi_device is False
        assert info.fallback_reason == "single_device_fallback"
    else:
        assert info.multi_device is True
        assert info.fallback_reason is None


def test_require_multi_device_failure_on_single_device() -> None:
    if jax.device_count() >= 2:
        pytest.skip("requires a single-device JAX runtime")

    with pytest.raises(RuntimeError, match="requires at least 2 JAX devices"):
        create_named_mesh(mesh_axis="data", require_multi_device=True)


def test_policy_metadata_for_data_parallel_single_axis() -> None:
    policy = get_sharding_policy("data_parallel_single_axis", mesh_axis="dp")

    assert policy.supported is True
    assert policy.name == "data_parallel_single_axis"
    assert policy.mesh_axis == "dp"
    assert policy.param_partition == ()
    assert policy.batch_partition == ("dp", None)


def test_placeholder_policy_reports_unsupported() -> None:
    policy = get_sharding_policy("model_parallel_placeholder", mesh_axis="data")

    assert policy.supported is False
    assert "unsupported" in policy.notes[0]


def test_unknown_policy_raises() -> None:
    with pytest.raises(ValueError, match="unsupported sharding policy"):
        get_sharding_policy("not_a_policy", mesh_axis="data")


def test_pjit_sharding_smoke_finite_loss() -> None:
    result = run_pjit_sharding_smoke(batch_size=2, seq_len=8)

    assert result.status == "passed"
    assert result.compile_api in {"jit_with_shardings", "pjit"}
    assert result.policy.name == "data_parallel_single_axis"
    assert math.isfinite(result.initial_loss)
    assert math.isfinite(result.final_loss)
    assert result.finite_loss is True
    assert result.multi_device_execution is (jax.device_count() >= 2)


def test_pjit_sharding_smoke_rejects_unsupported_policy() -> None:
    with pytest.raises(ValueError, match="unsupported in P46"):
        run_pjit_sharding_smoke(policy_name="fsdp_placeholder")


def test_report_writing(tmp_path: Path) -> None:
    result = run_pjit_sharding_smoke(batch_size=2, seq_len=8, skip_update=True)
    paths = write_p46_reports(result, out_dir=tmp_path)

    assert paths["json"].is_file()
    assert paths["markdown"].is_file()
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["phase"] == "P46"
    assert payload["update_ran"] is False
    assert "P46 pjit / Sharding Compile Smoke" in paths["markdown"].read_text(
        encoding="utf-8"
    )


def test_script_help() -> None:
    result = _run("scripts/run_pjit_sharding_smoke.py", "--help")

    assert result.returncode == 0
    assert "--require-multi-device" in result.stdout
    assert "--compile-api" in result.stdout


def test_script_json_output(tmp_path: Path) -> None:
    result = _run(
        "scripts/run_pjit_sharding_smoke.py",
        "--out",
        str(tmp_path),
        "--overwrite",
        "--json",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "passed"
    assert (tmp_path / "P46_RESULTS.md").is_file()
    assert (tmp_path / "pjit_sharding_smoke_report.json").is_file()
