from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from qrwkv_xla.sharding.smoke import PjitShardingSmokeResult

P46_DEFAULT_ARTIFACT_DIR = Path("artifacts/p46_pjit_sharding_smoke")
P46_JSON_REPORT = "pjit_sharding_smoke_report.json"
P46_MARKDOWN_REPORT = "P46_RESULTS.md"


def write_p46_reports(
    result: PjitShardingSmokeResult,
    *,
    out_dir: str | Path = P46_DEFAULT_ARTIFACT_DIR,
    overwrite: bool = False,
) -> dict[str, Path]:
    output_dir = Path(out_dir)
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    json_path = output_dir / P46_JSON_REPORT
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path = output_dir / P46_MARKDOWN_REPORT
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _markdown(payload: dict[str, Any]) -> str:
    fallback = payload.get("fallback_reason") or "none"
    lines = [
        "# P46 pjit / Sharding Compile Smoke",
        "",
        "## Summary",
        "",
        f"- phase: {payload['phase']}",
        f"- created_at_utc: {payload['created_at_utc']}",
        f"- overall_status: {payload['overall_status']}",
        f"- step_status: {payload['step_status']}",
        f"- compile_api: {payload['compile_api']}",
        f"- requested_compile_api: {payload['requested_compile_api']}",
        f"- policy: {payload['policy']}",
        f"- batch_size: {payload['batch_size']}",
        f"- sequence_length: {payload['sequence_length']}",
        f"- loss: {payload['loss']:.8f}",
        f"- loss_is_finite: {payload['loss_is_finite']}",
        f"- update_ran: {payload['update_ran']}",
        "",
        "## Backend / Mesh",
        "",
        f"- backend: {payload['backend']}",
        f"- platform: {payload['platform']}",
        f"- device_count: {payload['device_count']}",
        f"- local_device_count: {payload['local_device_count']}",
        f"- device_kinds: {', '.join(payload['device_kinds'])}",
        f"- mesh_shape: {payload['mesh_shape']}",
        f"- mesh_axis_names: {payload['mesh_axis_names']}",
        f"- multi_device: {payload['multi_device']}",
        f"- multi_device_execution: {payload['multi_device_execution']}",
        f"- fallback_reason: {fallback}",
        "",
        "## Policy Details",
        "",
        f"- param_partition: {payload['policy_details']['param_partition']}",
        f"- batch_partition: {payload['policy_details']['batch_partition']}",
        f"- output_partition: {payload['policy_details']['output_partition']}",
        "",
        "## Known Caveats",
        "",
    ]
    lines.extend(f"- {caveat}" for caveat in payload["limitations"])
    return "\n".join(lines).rstrip() + "\n"
