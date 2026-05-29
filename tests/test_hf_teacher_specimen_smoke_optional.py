from __future__ import annotations

import os
from pathlib import Path

import pytest

from qrwkv_xla.teachers import run_hf_teacher_specimen_smoke


@pytest.mark.skipif(
    os.environ.get("QRWKV_RUN_OPTIONAL_HF_SPECIMEN") != "1",
    reason="optional HF specimen smoke is disabled by default",
)
def test_optional_hf_teacher_specimen_smoke_local_cache(tmp_path: Path) -> None:
    result = run_hf_teacher_specimen_smoke(target_store=tmp_path / "targets")

    if result.status == "unavailable":
        pytest.skip(result.reason or result.error_message or "HF specimen unavailable")

    assert result.status == "pass"
    assert result.target_store_validated is True
    assert result.vocab_contract_extracted is True
    assert result.target_type == "full_logits"
