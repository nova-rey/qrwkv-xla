from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def validate_fingerprint_smoke_report(report: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    report_type = str(report.get("report_type", ""))
    if report_type not in {
        "corridor_only_smoke_report",
        "mixed_fingerprint_smoke_report",
    }:
        blockers.append("report_type must identify a fingerprint smoke report")

    for key in (
        "report_schema_phase",
        "status",
        "artifact",
        "training_path_kind",
        "smoke_student_kind",
        "smoke_student_uses_input_ids",
        "main_runner_integrated",
        "real_student_backend_integrated",
        "teacher_required",
        "hf_required",
        "accelerator_required",
        "requested_steps",
        "optimizer_steps_completed",
        "seed",
        "learning_rate",
        "corridor_targets",
        "limitations",
    ):
        if key not in report:
            blockers.append(f"missing required report field: {key}")

    if report_type == "corridor_only_smoke_report":
        for key in ("loss", "corridor_metrics"):
            if key not in report:
                blockers.append(f"missing corridor report section: {key}")
    if report_type == "mixed_fingerprint_smoke_report":
        for key in (
            "exemplar_reservoir",
            "loss_weights",
            "mixed_loss",
            "corridor_metrics",
            "exemplar_metrics",
        ):
            if key not in report:
                blockers.append(f"missing mixed report section: {key}")
    return blockers


def render_fingerprint_smoke_summary(report: Mapping[str, Any]) -> str:
    report_type = report.get("report_type")
    if report_type == "mixed_fingerprint_smoke_report":
        return render_mixed_fingerprint_smoke_summary(report)
    return render_corridor_fingerprint_smoke_summary(report)


def render_corridor_fingerprint_smoke_summary(report: Mapping[str, Any]) -> str:
    artifact = _mapping(report.get("artifact"))
    corridor = _mapping(report.get("corridor_targets"))
    loss = _mapping(report.get("loss"))
    metrics = _mapping(report.get("corridor_metrics"))
    return "\n".join(
        (
            "# Fingerprint Corridor Smoke Summary",
            "",
            f"Status: {report.get('status')}",
            f"Training path: {report.get('training_path_kind')}",
            f"Student: {report.get('smoke_student_kind')}",
            f"Uses input IDs: {_bool_text(report.get('smoke_student_uses_input_ids'))}",
            "Main runner integrated: "
            f"{_bool_text(report.get('main_runner_integrated'))}",
            f"Teacher required: {_bool_text(report.get('teacher_required'))}",
            "",
            "## Artifact",
            "- Type: "
            f"{artifact.get('artifact_type')} v{artifact.get('artifact_version')}",
            f"- Corridor records: {corridor.get('num_records')}",
            f"- Vocab size: {artifact.get('vocab_size')}",
            f"- Max sequence length: {artifact.get('max_seq_len')}",
            "",
            "## Run",
            f"- Requested steps: {report.get('requested_steps')}",
            f"- Optimizer steps completed: {report.get('optimizer_steps_completed')}",
            f"- Corridor batches consumed: {corridor.get('batches_consumed')}",
            "",
            "## Loss",
            f"- Initial: {loss.get('initial_total')}",
            f"- Final: {loss.get('final_total')}",
            f"- Delta: {loss.get('delta_total')}",
            f"- Non-increasing: {loss.get('non_increasing')} (diagnostic only)",
            "",
            "## Corridor Metrics",
            f"- Inside all rate: {metrics.get('inside_all_rate')}",
            f"- Entropy inside rate: {metrics.get('inside_entropy_rate')}",
            f"- Top-1 margin inside rate: {metrics.get('inside_top1_margin_rate')}",
            "",
            "## Limitations",
            *_limitation_lines(report),
            "",
        )
    )


def render_mixed_fingerprint_smoke_summary(report: Mapping[str, Any]) -> str:
    artifact = _mapping(report.get("artifact"))
    corridor = _mapping(report.get("corridor_targets"))
    exemplar = _mapping(report.get("exemplar_reservoir"))
    weights = _mapping(report.get("loss_weights"))
    loss = _mapping(report.get("mixed_loss"))
    corridor_metrics = _mapping(report.get("corridor_metrics"))
    exemplar_metrics = _mapping(report.get("exemplar_metrics"))
    return "\n".join(
        (
            "# Mixed Fingerprint Smoke Summary",
            "",
            f"Status: {report.get('status')}",
            f"Training path: {report.get('training_path_kind')}",
            f"Student: {report.get('smoke_student_kind')}",
            f"Uses input IDs: {_bool_text(report.get('smoke_student_uses_input_ids'))}",
            "Main runner integrated: "
            f"{_bool_text(report.get('main_runner_integrated'))}",
            "Real student backend integrated: "
            f"{_bool_text(report.get('real_student_backend_integrated'))}",
            f"Teacher required: {_bool_text(report.get('teacher_required'))}",
            "Exemplar reservoir enabled: "
            f"{_bool_text(report.get('exemplar_reservoir_enabled'))}",
            "",
            "## Artifact",
            "- Type: "
            f"{artifact.get('artifact_type')} v{artifact.get('artifact_version')}",
            f"- Corridor records: {corridor.get('num_records')}",
            f"- Exemplar records: {exemplar.get('num_records')}",
            f"- Exemplar payload: {exemplar.get('payload_type')}",
            "",
            "## Objective",
            f"- Corridor loss weight: {weights.get('corridor')}",
            f"- Exemplar loss weight: {weights.get('exemplar')}",
            "",
            "## Run",
            f"- Requested steps: {report.get('requested_steps')}",
            f"- Optimizer steps completed: {report.get('optimizer_steps_completed')}",
            f"- Corridor batches consumed: {corridor.get('batches_consumed')}",
            f"- Exemplar batches consumed: {exemplar.get('batches_consumed')}",
            "",
            "## Loss",
            f"- Initial mixed loss: {loss.get('initial_total')}",
            f"- Final mixed loss: {loss.get('final_total')}",
            f"- Delta: {loss.get('delta_total')}",
            f"- Non-increasing: {loss.get('non_increasing')} (diagnostic only)",
            "",
            "## Corridor Metrics",
            f"- Inside all rate: {corridor_metrics.get('inside_all_rate')}",
            f"- Entropy inside rate: {corridor_metrics.get('inside_entropy_rate')}",
            "",
            "## Exemplar Metrics",
            f"- KL: {exemplar_metrics.get('kl_loss')}",
            f"- Cross entropy: {exemplar_metrics.get('cross_entropy')}",
            f"- Teacher entropy: {exemplar_metrics.get('teacher_entropy')}",
            "",
            "## Limitations",
            *_limitation_lines(report),
            "",
        )
    )


def write_fingerprint_smoke_summary(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    summary_path = Path(path)
    summary_path.write_text(
        render_fingerprint_smoke_summary(report),
        encoding="utf-8",
    )
    return summary_path


def _limitation_lines(report: Mapping[str, Any]) -> tuple[str, ...]:
    limitations = report.get("limitations")
    if not isinstance(limitations, (list, tuple)) or not limitations:
        return ("- Standalone smoke only.",)
    return tuple(f"- {str(item)}" for item in limitations)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bool_text(value: Any) -> str:
    return str(bool(value)).lower()
