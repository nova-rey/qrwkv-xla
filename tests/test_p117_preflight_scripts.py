from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "scripts" / "bootstrap_tpu_vm.sh"
PREFLIGHT = ROOT / "scripts" / "run_p117_preflight.sh"
RUNBOOK = ROOT / "docs" / "P117_TPU_RUNBOOK.md"


def test_bootstrap_script_exists_and_is_strict() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")

    assert text.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in text
    assert "python -m pip install -e" in text
    assert "numpy<2" in text


def test_preflight_script_exists_and_is_strict() -> None:
    text = PREFLIGHT.read_text(encoding="utf-8")

    assert text.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in text


def test_preflight_script_uses_expected_textbook_release_and_hash() -> None:
    text = PREFLIGHT.read_text(encoding="utf-8")

    assert "p117-textbook-smoke-v0" in text
    assert "p117_teacher_textbook_tiny_gpt2_smoke.tar.gz" in text
    assert "cbe355a415606012eae4fa856aee180b1b6dc83ee06b4fb70188d57f253d7f23" in text


def test_preflight_script_runs_validation_readiness_and_dry_run() -> None:
    text = PREFLIGHT.read_text(encoding="utf-8")

    assert "validate_teacher_textbook.py" in text
    assert "run_big_burn_readiness_report.py" in text
    assert "run_first_serious_burn.py" in text
    assert "--mode dry_run" in text


def test_preflight_script_does_not_execute_real_mode_by_default() -> None:
    text = PREFLIGHT.read_text(encoding="utf-8")

    assert "--mode real" in text
    assert "--confirm-serious-burn" in text
    assert "not executed" in text
    assert "P117 must resolve that handoff" in text


def test_runbook_documents_tpu_lifecycle_and_troubleshooting() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    assert "gcloud compute tpus queued-resources create" in text
    assert "gcloud compute tpus tpu-vm ssh" in text
    assert "gcloud compute tpus tpu-vm stop" in text
    assert "gcloud compute tpus tpu-vm start" in text
    assert "gcloud compute tpus tpu-vm delete" in text
    assert "Reservation not found" in text
    assert "Insufficient capacity" in text
    assert "NumPy 2" in text
