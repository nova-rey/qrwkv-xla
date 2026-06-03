from __future__ import annotations

import json
from pathlib import Path

from qrwkv_xla.readiness import (
    BIG_BURN_READINESS_CLAIMS_NOT_MADE,
    REQUIRED_BIG_BURN_READINESS_CATEGORIES,
    BigBurnReadinessReport,
    ReadinessCheck,
    ReadinessStatus,
    aggregate_readiness_status,
    build_big_burn_readiness_report,
    recommended_next_action_for_status,
)
from scripts.run_big_burn_readiness_report import main as readiness_cli_main


def test_readiness_report_generates_without_tpu_gpu_hf_or_internet(
    tmp_path: Path,
) -> None:
    report = build_big_burn_readiness_report(work_dir=tmp_path)

    assert report.phase == "P111"
    assert report.scope == "big_burn_readiness_report"
    assert report.status is ReadinessStatus.PASS
    assert all("tpu_required=True" not in item for item in _all_evidence(report))
    assert "qwen_specific_support" in report.claims_not_made


def test_report_json_is_serializable(tmp_path: Path) -> None:
    report = build_big_burn_readiness_report(work_dir=tmp_path)

    payload = json.loads(json.dumps(report.to_report()))

    assert payload["phase"] == "P111"
    assert payload["status"] == "pass"
    assert isinstance(payload["checks"], list)


def test_report_includes_required_readiness_categories(tmp_path: Path) -> None:
    report = build_big_burn_readiness_report(work_dir=tmp_path)

    names = {check.name for check in report.checks}

    assert names == set(REQUIRED_BIG_BURN_READINESS_CATEGORIES)


def test_status_aggregation_fails_if_any_check_fails() -> None:
    checks = (
        _check("a", ReadinessStatus.PASS),
        _check("b", ReadinessStatus.FAIL),
        _check("c", ReadinessStatus.WARN),
    )

    assert aggregate_readiness_status(checks) is ReadinessStatus.FAIL


def test_status_aggregation_warns_if_no_failures_and_any_warning() -> None:
    checks = (
        _check("a", ReadinessStatus.PASS),
        _check("b", ReadinessStatus.WARN),
    )

    assert aggregate_readiness_status(checks) is ReadinessStatus.WARN


def test_status_aggregation_passes_when_all_checks_pass() -> None:
    checks = (
        _check("a", ReadinessStatus.PASS),
        _check("b", ReadinessStatus.PASS),
    )

    assert aggregate_readiness_status(checks) is ReadinessStatus.PASS


def test_recommended_next_action_changes_for_pass_warn_fail() -> None:
    assert "Proceed to P112" in recommended_next_action_for_status(ReadinessStatus.PASS)
    assert "Review warnings" in recommended_next_action_for_status(ReadinessStatus.WARN)
    assert "Do not proceed" in recommended_next_action_for_status(ReadinessStatus.FAIL)


def test_report_includes_claims_not_made(tmp_path: Path) -> None:
    report = build_big_burn_readiness_report(work_dir=tmp_path)

    assert report.claims_not_made == BIG_BURN_READINESS_CLAIMS_NOT_MADE
    assert "training_success_guaranteed" in report.claims_not_made
    assert "tokenizer_remapping_supported" in report.claims_not_made


def test_cli_writes_json_report(tmp_path: Path) -> None:
    output = tmp_path / "readiness.json"

    code = readiness_cli_main(["--output", str(output)])

    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["phase"] == "P111"
    assert payload["scope"] == "big_burn_readiness_report"


def test_cli_strict_exits_nonzero_for_warning(tmp_path: Path) -> None:
    output = tmp_path / "warn.json"

    code = readiness_cli_main(
        ["--output", str(output), "--strict"],
        report_builder=_warn_report,
    )

    assert code == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "warn"


def test_no_training_or_optimizer_occurs() -> None:
    source = Path("src/qrwkv_xla/readiness/big_burn.py").read_text(encoding="utf-8")

    assert "train_step" not in source
    assert "run_tiny_overfit_rehearsal" not in source
    assert "optimizer" not in source


def test_no_p112_burn_is_started() -> None:
    source = Path("src/qrwkv_xla/readiness/big_burn.py").read_text(encoding="utf-8")

    assert "P112 First Serious Compute Burn" in source
    assert "p112_started" in source
    assert "run_p112" not in source
    assert "first_serious_compute_burn" not in source


def _check(name: str, status: ReadinessStatus) -> ReadinessCheck:
    return ReadinessCheck(
        name=name,
        status=status,
        summary=f"{name} summary",
    )


def _warn_report(**_: object) -> BigBurnReadinessReport:
    check = _check("synthetic_warning", ReadinessStatus.WARN)
    return BigBurnReadinessReport(
        phase="P111",
        status=ReadinessStatus.WARN,
        scope="big_burn_readiness_report",
        checks=(check,),
        blockers=(),
        warnings=("synthetic warning",),
        recommended_next_action=recommended_next_action_for_status(
            ReadinessStatus.WARN
        ),
        claims_not_made=BIG_BURN_READINESS_CLAIMS_NOT_MADE,
    )


def _all_evidence(report: BigBurnReadinessReport) -> tuple[str, ...]:
    return tuple(item for check in report.checks for item in check.evidence)
