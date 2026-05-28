from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_pallas_tpu_smoke.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_pallas_tpu_smoke", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p88_smoke_module_imports_and_inspects_environment() -> None:
    smoke = _module()

    info = smoke.inspect_jax_environment()

    assert "jax_version" in info
    assert "jaxlib_version" in info
    assert isinstance(info["devices"], list)
    assert isinstance(info["tpu_devices_detected"], bool)


def test_p88_no_tpu_report_is_unavailable(monkeypatch) -> None:
    smoke = _module()
    monkeypatch.setattr(
        smoke,
        "inspect_jax_environment",
        lambda: {
            "jax_import_ok": True,
            "jax_version": "test-jax",
            "jaxlib_version": "test-jaxlib",
            "platform": "cpu",
            "default_backend": "cpu",
            "devices": [{"id": "0", "platform": "cpu", "device_kind": "cpu"}],
            "tpu_devices_detected": False,
            "error_type": None,
            "error_message": None,
        },
    )

    report = smoke.run_smoke()

    assert report["phase"] == "P88"
    assert report["status"] == "unavailable"
    assert report["reason"] == "no_tpu_devices_detected"
    assert report["jit_lowering_attempted"] is False
    assert report["execution_attempted"] is False
    assert "production_pallas_ready" in report["claims_not_made"]
    assert "training_ready" in report["claims_not_made"]
    assert "throughput_proven" in report["claims_not_made"]
    assert "full_model_quality_proven" in report["claims_not_made"]
    assert "pallas_default_ready" in report["claims_not_made"]


def test_p88_require_tpu_is_recorded_for_unavailable(monkeypatch) -> None:
    smoke = _module()
    monkeypatch.setattr(
        smoke,
        "inspect_jax_environment",
        lambda: {
            "jax_import_ok": True,
            "jax_version": "test-jax",
            "jaxlib_version": "test-jaxlib",
            "platform": "cpu",
            "default_backend": "cpu",
            "devices": [],
            "tpu_devices_detected": False,
            "error_type": None,
            "error_message": None,
        },
    )

    report = smoke.run_smoke(require_tpu=True)

    assert report["status"] == "unavailable"
    assert report["require_tpu"] is True


def test_p88_report_json_is_written(tmp_path: Path, monkeypatch) -> None:
    smoke = _module()
    output = tmp_path / "pallas_tpu_compile_smoke.json"
    monkeypatch.setattr(
        smoke,
        "inspect_jax_environment",
        lambda: {
            "jax_import_ok": True,
            "jax_version": "test-jax",
            "jaxlib_version": "test-jaxlib",
            "platform": "cpu",
            "default_backend": "cpu",
            "devices": [],
            "tpu_devices_detected": False,
            "error_type": None,
            "error_message": None,
        },
    )

    report = smoke.run_smoke()
    smoke.write_reports(report, output)

    persisted = json.loads(output.read_text())
    assert persisted["phase"] == "P88"
    assert persisted["status"] == "unavailable"
    assert (tmp_path / "P88_TPU_COMPILE_PERFORMANCE_SMOKE_REPORT.md").is_file()
