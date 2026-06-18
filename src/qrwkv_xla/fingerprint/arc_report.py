from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from qrwkv_xla.artifacts._json import write_json

REQUIRED_ARC2_FLAGS: tuple[tuple[str, str], ...] = (
    ("p140_real_student_forward_smoke", "P140 real student forward smoke"),
    ("p141_main_runner_fingerprint_mode", "P141 main runner fingerprint mode"),
    ("p142_input_conditioned_rehearsal", "P142 input-conditioned rehearsal"),
    ("p143_teacher_side_capture_skeleton", "P143 teacher-side capture skeleton"),
    ("p144_known_stat_parity", "P144 capture parity"),
    ("p145_tiny_real_teacher_capture_path", "P145 tiny real teacher capture"),
    (
        "p146_real_teacher_artifact_training_rehearsal",
        "P146 real-teacher artifact training rehearsal",
    ),
    ("p147_baseline_comparison_harness", "P147 baseline comparison harness"),
    ("p148_quality_per_byte_experiment", "P148 quality-per-byte smoke"),
)


@dataclass(frozen=True)
class FingerprintArc2ReportConfig:
    output_dir: Path
    snapshot_path: Path = Path("docs/QRWKV_SNAPSHOT.yaml")
    overwrite: bool = False


@dataclass(frozen=True)
class FingerprintArc2ReportResult:
    status: str
    output_dir: Path
    report_path: Path
    summary_path: Path
    recommendation: str


def run_fingerprint_arc2_report(
    config: FingerprintArc2ReportConfig,
) -> FingerprintArc2ReportResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.output_dir / "p149_arc2_report.json"
    summary_path = config.output_dir / "p149_arc2_summary.md"
    if not config.overwrite:
        for path in (report_path, summary_path):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite existing file: {path}")

    snapshot = _load_snapshot(config.snapshot_path)
    report = build_fingerprint_arc2_report(snapshot, snapshot_path=config.snapshot_path)
    write_json(report_path, report)
    summary_path.write_text(_render_summary(report), encoding="utf-8")
    return FingerprintArc2ReportResult(
        status=str(report["status"]),
        output_dir=config.output_dir,
        report_path=report_path,
        summary_path=summary_path,
        recommendation=str(report["go_no_go"]["recommendation"]),
    )


def build_fingerprint_arc2_report(
    snapshot: dict[str, Any],
    *,
    snapshot_path: Path | None = None,
) -> dict[str, Any]:
    main_contains = _mapping(snapshot.get("main_contains"))
    evidence = [
        {
            "flag": flag,
            "description": description,
            "present": bool(main_contains.get(flag, False)),
        }
        for flag, description in REQUIRED_ARC2_FLAGS
    ]
    missing = [item for item in evidence if not item["present"]]
    status = "pass" if not missing else "fail"
    recommendation = "go_with_constraints" if status == "pass" else "no_go"
    return {
        "phase": "P149",
        "run_kind": "arc2_report_go_no_go",
        "status": status,
        "source_snapshot": None if snapshot_path is None else str(snapshot_path),
        "arc": {
            "id": "arc2_real_student_integration_and_teacher_pure_capture",
            "current_phase": snapshot.get("current_phase"),
            "checkpoint": snapshot.get("checkpoint"),
            "covered_phases": [f"P{number}" for number in range(140, 149)],
        },
        "evidence": evidence,
        "go_no_go": {
            "recommendation": recommendation,
            "go": status == "pass",
            "constraints": _constraints(),
            "next_phase": "larger_controlled_fingerprint_experiments",
            "no_go_blockers": [f"{item['flag']} missing" for item in missing],
        },
        "claims": {
            "general_quality_claim_made": False,
            "trained_baseline_win_claim_made": False,
            "radlads_parity_claim_made": False,
            "scale_readiness_claim_made": False,
            "production_readiness_claim_made": False,
            "pallas_default_claim_made": False,
        },
        "findings": _findings(),
        "open_gaps": _open_gaps(),
        "recommended_next_steps": _recommended_next_steps(),
    }


def _load_snapshot(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"snapshot must be a mapping: {path}")
    return payload


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _constraints() -> list[str]:
    return [
        "Proceed only to larger controlled fingerprint experiments, "
        "not production scale.",
        "Add a trained non-fingerprint baseline before making method-vs-method claims.",
        "Keep teacher/corpus/student budgets fixed and reported.",
        "Keep quality-per-byte claims scoped to measured tiny smoke settings.",
        "Do not claim RADLADS parity, scale readiness, or production readiness.",
    ]


def _findings() -> list[dict[str, Any]]:
    return [
        {
            "id": "real_student_path",
            "status": "complete",
            "summary": (
                "P140-P142 establish real registered student forward/training "
                "through fingerprint_corridor."
            ),
        },
        {
            "id": "teacher_capture_path",
            "status": "complete_tiny",
            "summary": (
                "P143-P145 establish calibrated synthetic capture and tiny "
                "local-files-only real teacher capture."
            ),
        },
        {
            "id": "producer_consumer_rehearsal",
            "status": "complete_tiny",
            "summary": (
                "P146 proves a tiny real-teacher fingerprint artifact can train "
                "the real student path without teacher access during training."
            ),
        },
        {
            "id": "scoreboard_and_measurement",
            "status": "complete_reference_only",
            "summary": (
                "P147-P148 add baseline/fingerprint scoreboard and tiny "
                "corridor-adherence-per-byte metrics against an init-only reference."
            ),
        },
    ]


def _open_gaps() -> list[dict[str, str]]:
    return [
        {
            "id": "trained_baseline",
            "severity": "high",
            "summary": "No competitive trained non-fingerprint baseline exists yet.",
        },
        {
            "id": "generalization_eval",
            "severity": "high",
            "summary": "P148 uses train-artifact reuse, not held-out evaluation.",
        },
        {
            "id": "scale",
            "severity": "medium",
            "summary": "Artifacts and runs remain tiny JSONL CPU-safe smokes.",
        },
        {
            "id": "mixed_objective",
            "severity": "medium",
            "summary": "Main runner still lacks exemplar/mixed-objective training.",
        },
    ]


def _recommended_next_steps() -> list[str]:
    return [
        "Add a cheap trained baseline arm under the P147/P148 harness.",
        "Add a held-out tiny eval artifact before stronger quality-per-byte language.",
        "Run a small fixed-budget fingerprint experiment with recorded artifact bytes.",
        "Use P149 constraints as gates before scaling capture or training.",
    ]


def _render_summary(report: dict[str, Any]) -> str:
    go_no_go = report["go_no_go"]
    evidence = report["evidence"]
    return "\n".join(
        (
            "# P149 Arc 2 Report / Go-No-Go",
            "",
            f"Status: {report['status']}",
            f"Recommendation: {go_no_go['recommendation']}",
            f"Next phase: {go_no_go['next_phase']}",
            "",
            "Arc 2 has enough tiny evidence to proceed to larger controlled "
            "fingerprint experiments, but not enough evidence for production, "
            "scale, RADLADS parity, or general quality claims.",
            "",
            "## Evidence",
            *(f"- {item['description']}: {item['present']}" for item in evidence),
            "",
            "## Constraints",
            *(f"- {item}" for item in go_no_go["constraints"]),
            "",
            "## Open Gaps",
            *(
                f"- {item['id']} ({item['severity']}): {item['summary']}"
                for item in report["open_gaps"]
            ),
            "",
            "## Claims Not Made",
            "- General quality claim: false",
            "- Trained baseline win claim: false",
            "- RADLADS parity claim: false",
            "- Scale readiness claim: false",
            "- Production readiness claim: false",
            "",
        )
    )
