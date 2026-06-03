from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qrwkv_xla.xla import (
    enable_transparent_hugepages,
    inspect_jax_environment,
    read_transparent_hugepage_status,
    run_runtime_environment_preflight,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_runtime_environment_preflight.py"


def test_transparent_hugepage_parser_detects_always_enabled(
    tmp_path: Path,
) -> None:
    path = tmp_path / "enabled"
    path.write_text("[always] madvise never\n", encoding="utf-8")

    info = read_transparent_hugepage_status(path)

    assert info.status == "enabled"
    assert info.enabled is True
    assert info.raw_status == "[always] madvise never"


def test_transparent_hugepage_parser_detects_madvise_disabled(
    tmp_path: Path,
) -> None:
    path = tmp_path / "enabled"
    path.write_text("always [madvise] never\n", encoding="utf-8")

    info = read_transparent_hugepage_status(path)

    assert info.status == "disabled"
    assert info.enabled is False


def test_transparent_hugepage_parser_detects_never_disabled(
    tmp_path: Path,
) -> None:
    path = tmp_path / "enabled"
    path.write_text("always madvise [never]\n", encoding="utf-8")

    info = read_transparent_hugepage_status(path)

    assert info.status == "disabled"
    assert info.enabled is False


def test_missing_hugepage_file_reports_unavailable(tmp_path: Path) -> None:
    info = read_transparent_hugepage_status(tmp_path / "missing")

    assert info.status == "unavailable"
    assert info.enabled is None
    assert info.error_message


def test_malformed_hugepage_file_reports_unknown(tmp_path: Path) -> None:
    path = tmp_path / "enabled"
    path.write_text("always madvise never\n", encoding="utf-8")

    info = read_transparent_hugepage_status(path)

    assert info.status == "unknown"
    assert info.enabled is None


def test_recommended_enable_command_is_included(tmp_path: Path) -> None:
    path = tmp_path / "enabled"
    path.write_text("always [madvise] never\n", encoding="utf-8")

    info = read_transparent_hugepage_status(path)

    assert (
        info.recommended_enable_command
        == 'sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"'
    )


def test_default_preflight_is_read_only(tmp_path: Path) -> None:
    path = tmp_path / "enabled"
    path.write_text("always [madvise] never\n", encoding="utf-8")

    report = run_runtime_environment_preflight(hugepage_path=path)

    assert report.mutation_attempted is False
    assert report.mutation_ok is None
    assert path.read_text(encoding="utf-8") == "always [madvise] never\n"


def test_explicit_enable_helper_writes_always_to_fake_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "enabled"
    path.write_text("always [madvise] never\n", encoding="utf-8")

    mutation = enable_transparent_hugepages(path)

    assert mutation.attempted is True
    assert mutation.ok is True
    assert path.read_text(encoding="utf-8") == "always\n"


def test_explicit_enable_helper_handles_os_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "enabled"

    def raise_os_error(
        self: Path,
        data: str,
        encoding: str | None = None,
    ) -> int:
        del self, data, encoding
        raise OSError("read-only fake sysfs")

    monkeypatch.setattr(Path, "write_text", raise_os_error)

    mutation = enable_transparent_hugepages(path)

    assert mutation.attempted is True
    assert mutation.ok is False
    assert "read-only fake sysfs" in str(mutation.error_message)


def test_jax_unavailable_reports_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "jax":
            raise ImportError("jax unavailable for test")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    info = inspect_jax_environment()

    assert info.jax_available is False
    assert info.devices == ()
    assert "jax unavailable" in str(info.error_message)


def test_fake_jax_devices_report_tpu_detected() -> None:
    fake_jax = SimpleNamespace(
        __version__="0.test",
        default_backend=lambda: "tpu",
        devices=lambda: (
            SimpleNamespace(id=0, platform="tpu", device_kind="TPU v5 lite"),
        ),
    )

    info = inspect_jax_environment(fake_jax)

    assert info.jax_available is True
    assert info.default_backend == "tpu"
    assert info.tpu_devices_detected is True
    assert info.devices[0].platform == "tpu"
    assert info.devices[0].device_kind == "TPU v5 lite"


def test_require_tpu_reports_fail_when_no_tpu_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "enabled"
    path.write_text("[always] madvise never\n", encoding="utf-8")
    fake_info = SimpleNamespace(
        jax_available=True,
        jax_version="0.test",
        jaxlib_version="0.lib",
        default_backend="cpu",
        devices=(),
        tpu_devices_detected=False,
        error_message=None,
    )

    monkeypatch.setattr(
        "qrwkv_xla.xla.environment_preflight.inspect_jax_environment",
        lambda: fake_info,
    )

    report = run_runtime_environment_preflight(
        hugepage_path=path,
        require_tpu=True,
    )

    assert report.status == "fail"
    assert report.tpu_devices_detected is False


def test_json_report_is_serializable(tmp_path: Path) -> None:
    path = tmp_path / "enabled"
    path.write_text("[always] madvise never\n", encoding="utf-8")

    report = run_runtime_environment_preflight(hugepage_path=path)
    payload = report.to_report()

    assert json.loads(json.dumps(payload))["phase"] == "P109"


def test_runtime_environment_preflight_cli_writes_json_report(
    tmp_path: Path,
) -> None:
    hugepage_path = tmp_path / "enabled"
    output_path = tmp_path / "report.json"
    hugepage_path.write_text("[always] madvise never\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--hugepage-path",
            str(hugepage_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["phase"] == "P109"
    assert report["transparent_hugepages"]["status"] == "enabled"


def test_baseline_preflight_makes_no_tpu_gpu_or_internet_claims(
    tmp_path: Path,
) -> None:
    path = tmp_path / "enabled"
    path.write_text("[always] madvise never\n", encoding="utf-8")

    report = run_runtime_environment_preflight(hugepage_path=path)

    assert "training_ready" in report.claims_not_made
    assert "performance_benchmark_complete" in report.claims_not_made
    assert "pallas_default_ready" in report.claims_not_made
