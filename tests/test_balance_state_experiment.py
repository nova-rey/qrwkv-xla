from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from qrwkv_xla.parity.radlads_balance_state_experiment import (
    BALANCE_STATE_EXPERIMENT_SCHEMA,
    BALANCE_STATE_STABILITY_SCHEMA,
    base_balance_state_experiment_config,
    experimental_balance_state_config,
    run_balance_state_experiment,
    run_balance_state_stability_smoke,
)
from qrwkv_xla.students import RWKV7QwenReferenceConfig, RWKV7QwenReferenceStudent

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_SCRIPT = ROOT / "scripts" / "run_balance_state_experiment.py"
STABILITY_SCRIPT = ROOT / "scripts" / "run_balance_state_stability_smoke.py"


def test_experimental_config_is_explicit_and_default_is_off() -> None:
    off = base_balance_state_experiment_config()
    experimental = experimental_balance_state_config(off)

    assert not off.radlads_balance_state_terms
    assert not off.radlads_balance_state
    assert not off.use_radlads_balance_state_terms
    assert experimental.radlads_balance_state_terms
    assert experimental.radlads_balance_state
    assert experimental.use_radlads_balance_state_terms


def test_default_off_behavior_matches_explicit_off() -> None:
    default_student = RWKV7QwenReferenceStudent(
        RWKV7QwenReferenceConfig(
            vocab_size=32,
            hidden_size=8,
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            emit_logits=True,
        )
    )
    explicit_off = RWKV7QwenReferenceStudent(
        RWKV7QwenReferenceConfig(
            vocab_size=32,
            hidden_size=8,
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            emit_logits=True,
            radlads_balance_state_terms=False,
            radlads_balance_state=False,
        )
    )
    params = default_student.init_params(jax.random.PRNGKey(65))
    input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)

    default_output, default_state = default_student.apply_with_state(params, input_ids)
    explicit_output, explicit_state = explicit_off.apply_with_state(params, input_ids)

    assert _max_abs(default_output.hidden_states, explicit_output.hidden_states) == 0.0
    assert (
        _max_abs(default_state.wkv_matrix_state, explicit_state.wkv_matrix_state) == 0.0
    )
    assert _max_abs(default_state.shift_state, explicit_state.shift_state) == 0.0


def test_experimental_path_requires_terms_flag() -> None:
    with pytest.raises(ValueError, match="radlads_balance_state requires"):
        RWKV7QwenReferenceConfig(
            hidden_size=8,
            num_heads=2,
            radlads_balance_state=True,
        )


def test_experiment_report_writes_required_surfaces(tmp_path: Path) -> None:
    report = run_balance_state_experiment(out_dir=tmp_path, overwrite=True)

    assert report["schema"] == BALANCE_STATE_EXPERIMENT_SCHEMA
    assert report["default_behavior_preserved"] is True
    assert report["off_mode_status"] == "off"
    assert report["experimental_mode_status"] == "experimental"
    assert report["experimental_flags"]["radlads_balance_state_terms"] is True
    assert report["experimental_flags"]["radlads_balance_state"] is True
    assert (tmp_path / "P65_RESULTS.md").is_file()
    assert (tmp_path / "balance_state_experiment_report.json").is_file()
    assert (tmp_path / "DIFF_SUMMARY.md").is_file()
    assert (tmp_path / "OFF_VS_EXPERIMENTAL.md").is_file()
    assert (tmp_path / "off" / "mode_report.json").is_file()
    assert (tmp_path / "experimental" / "mode_report.json").is_file()

    payload = json.loads(
        (tmp_path / "balance_state_experiment_report.json").read_text()
    )
    assert payload["schema"] == BALANCE_STATE_EXPERIMENT_SCHEMA
    first = payload["cases"][0]
    surfaces = {item["surface"] for item in first["comparisons"]}
    assert {
        "log_w",
        "logits",
        "shift_state",
        "wkv_matrix_state",
        "hidden_states",
    } <= surfaces
    assert "first_divergent_stage" in first


def test_stability_smoke_report_writes_required_shape(tmp_path: Path) -> None:
    report = run_balance_state_stability_smoke(out_dir=tmp_path, overwrite=True)

    assert report["schema"] == BALANCE_STATE_STABILITY_SCHEMA
    assert report["status"] == "pass"
    assert report["experimental_flags"]["radlads_balance_state"] is True
    assert (tmp_path / "STABILITY_SMOKE.md").is_file()
    payload = json.loads((tmp_path / "stability" / "stability_report.json").read_text())
    assert payload["schema"] == BALANCE_STATE_STABILITY_SCHEMA
    assert payload["surfaces"]["wkv_matrix_state"]["shape"] == [2, 2, 2, 4, 4]


def test_scripts_help() -> None:
    for script in (EXPERIMENT_SCRIPT, STABILITY_SCRIPT):
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "P65" in result.stdout
        assert "balance-state" in result.stdout


def test_scripts_write_reports(tmp_path: Path) -> None:
    experiment = subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT_SCRIPT),
            "--out-dir",
            str(tmp_path / "experiment"),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "default_behavior_preserved=True" in experiment.stdout
    smoke = subprocess.run(
        [
            sys.executable,
            str(STABILITY_SCRIPT),
            "--out-dir",
            str(tmp_path / "experiment"),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "status=pass" in smoke.stdout


def _max_abs(left: object, right: object) -> float:
    return float(jnp.max(jnp.abs(jnp.asarray(left) - jnp.asarray(right))))
