from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from qrwkv_xla.targets import validate_target_bundle
from qrwkv_xla.teacher_export import (
    ExportRequest,
    HFTeacherExporter,
    load_teacher_export_config,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    os.environ.get("QRWKV_RUN_HF_INTEGRATION") != "1",
    reason="set QRWKV_RUN_HF_INTEGRATION=1 to run optional HF integration",
)
def test_hf_tiny_integration_opt_in(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    config = load_teacher_export_config(
        ROOT / "configs" / "teacher_export_hf_tiny.yaml"
    )
    config = replace(
        config,
        runtime=replace(config.runtime, output_dir=tmp_path / "hf_tiny"),
    )

    result = HFTeacherExporter().export(
        ExportRequest(config=config, output_dir=config.runtime.output_dir)
    )

    validate_target_bundle(result.output_dir)
