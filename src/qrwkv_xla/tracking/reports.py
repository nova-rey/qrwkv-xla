from __future__ import annotations

from pathlib import Path
from typing import Any

from qrwkv_xla.tracking.json_io import write_json

P47_DEFAULT_ARTIFACT_DIR = Path("artifacts/p47_experiment_tracking_smoke")


def write_tracking_smoke_reports(
    report: dict[str, Any],
    *,
    out_dir: str | Path = P47_DEFAULT_ARTIFACT_DIR,
    overwrite: bool = False,
) -> dict[str, Path]:
    output = Path(out_dir)
    if output.exists() and not overwrite:
        json_path = output / "tracking_smoke_report.json"
        markdown_path = output / "P47_RESULTS.md"
        if json_path.exists() or markdown_path.exists():
            raise FileExistsError(f"P47 report output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    json_path = write_json(output / "tracking_smoke_report.json", report)
    markdown_path = output / "P47_RESULTS.md"
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    metadata = report.get("metadata", {})
    paths = report.get("paths", {})
    lines = [
        "# P47 Experiment Tracking Smoke Results",
        "",
        f"- Overall status: {report.get('overall_status')}",
        f"- Tracking mode: {report.get('tracking_mode')}",
        f"- Artifact path: {report.get('artifact_path')}",
        f"- Local run id: {report.get('local_run_id')}",
        f"- WandB status: {report.get('wandb_status')}",
        f"- Created at UTC: {metadata.get('created_at_utc')}",
        f"- Repo commit: {report.get('commit')}",
        f"- Git dirty classification: {report.get('git_dirty')}",
        f"- Backend: {report.get('backend')}",
        f"- Device count: {report.get('device_count')}",
        f"- Steps: {report.get('steps')}",
        f"- Final loss: {report.get('final_loss')}",
        f"- Final loss finite: {report.get('loss_is_finite')}",
        f"- Metrics logged: {report.get('metrics_logged_count')}",
        f"- Artifacts logged: {report.get('artifacts_logged_count')}",
        f"- Summary written: {report.get('summary_written')}",
        f"- Tokens seen: {summary.get('tokens_seen')}",
        f"- Examples seen: {summary.get('examples_seen')}",
        "",
        "## Files",
        "",
    ]
    for label in (
        "report_json",
        "run_metadata",
        "config",
        "metrics",
        "summary",
        "artifacts_manifest",
    ):
        if label in paths:
            lines.append(f"- `{label}`: `{paths[label]}`")
    limitations = report.get("limitations", [])
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in limitations],
            "",
            "## Scope",
            "",
            "P47 proves optional experiment tracking plumbing for a tiny local "
            "smoke only. Local tracking is the source of truth. WandB is optional "
            "and is not required for normal development or CI.",
            "",
        ]
    )
    return "\n".join(lines)
