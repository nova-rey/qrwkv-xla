from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from qrwkv_xla.artifacts import (
    load_fingerprint_exemplars,
    load_fingerprint_targets,
    validate_fingerprint_artifact,
)
from qrwkv_xla.fingerprint import (
    FingerprintCaptureConfig,
    FingerprintExemplarReservoirCaptureConfig,
    FingerprintModeDiscoveryConfig,
    build_synthetic_capture_examples,
    capture_fingerprint_artifact,
)
from qrwkv_xla.training import (
    RealStudentFingerprintForwardConfig,
    run_real_student_fingerprint_forward_smoke,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_fingerprint_artifact.py"


def test_synthetic_capture_emits_valid_loadable_artifact(tmp_path: Path) -> None:
    result = _capture(tmp_path, max_exemplars=6)

    assert result.validation_ok is True
    assert result.manifest_path.is_file()
    assert result.modes_path.is_file()
    assert result.targets_path.is_file()
    assert result.exemplars_path is not None
    assert result.exemplars_path.is_file()
    assert result.capture_summary_path.is_file()

    validation = validate_fingerprint_artifact(result.output_dir)
    targets = load_fingerprint_targets(result.output_dir, batch_size=3)
    exemplars = load_fingerprint_exemplars(result.output_dir, batch_size=2)

    assert validation.ok is True
    assert validation.metadata["records"] == 32
    assert targets.num_records == 32
    assert targets.max_seq_len == 8
    assert targets.vocab_size == 16
    assert exemplars.num_records == 6
    assert exemplars.vocab_size == 16


def test_exemplar_budget_is_configurable(tmp_path: Path) -> None:
    small = _capture(tmp_path / "small", max_exemplars=2)
    larger = _capture(tmp_path / "larger", max_exemplars=5)

    small_manifest = _json(small.manifest_path)
    larger_manifest = _json(larger.manifest_path)

    assert small.summary["exemplars_retained"] == 2
    assert larger.summary["exemplars_retained"] == 5
    assert small_manifest["exemplar_reservoir"]["max_exemplars"] == 2
    assert larger_manifest["exemplar_reservoir"]["max_exemplars"] == 5
    assert small_manifest["exemplar_reservoir"]["num_records"] == 2
    assert larger_manifest["exemplar_reservoir"]["num_records"] == 5


def test_mode_count_is_data_driven(tmp_path: Path) -> None:
    one_position = _capture(
        tmp_path / "one_position",
        max_seq_len=1,
        num_examples=1,
        max_exemplars=1,
    )
    many_positions = _capture(
        tmp_path / "many_positions",
        max_seq_len=8,
        num_examples=4,
        max_exemplars=1,
    )

    assert one_position.summary["modes_discovered"] == 1
    assert many_positions.summary["modes_discovered"] > 1
    assert (
        many_positions.summary["modes_discovered"]
        != one_position.summary["modes_discovered"]
    )
    assert many_positions.summary["modes_discovered"] == len(
        _observed_mode_keys(many_positions.modes_path)
    )


def test_max_modes_guard_rejects_mode_explosion(tmp_path: Path) -> None:
    examples = build_synthetic_capture_examples(
        num_examples=4,
        max_seq_len=8,
        vocab_size=16,
    )
    config = _config(tmp_path, max_exemplars=1)
    config = replace(
        config,
        mode_discovery=replace(FingerprintModeDiscoveryConfig(), max_modes=1),
    )

    with pytest.raises(ValueError, match="more modes than max_modes=1"):
        capture_fingerprint_artifact(config, examples)


def test_bounds_are_valid_and_match_mode_bounds(tmp_path: Path) -> None:
    result = _capture(tmp_path, max_exemplars=4)
    modes = {
        mode["mode_id"]: mode["bounds"] for mode in _json(result.modes_path)["modes"]
    }
    rows = _jsonl(result.targets_path)

    assert rows
    for row in rows:
        assert row["mode_id"] in modes
        mode_bounds = modes[row["mode_id"]]
        for stat, bounds in row["bounds"].items():
            assert bounds["min"] <= bounds["max"]
            assert bounds == mode_bounds[stat]
            if stat == "entropy":
                assert bounds["min"] >= 0.0
            else:
                assert 0.0 <= bounds["min"] <= bounds["max"] <= 1.0


def test_dense_exemplar_probs_are_valid(tmp_path: Path) -> None:
    result = _capture(tmp_path, max_exemplars=8)
    rows = _jsonl(result.exemplars_path)

    assert rows
    for row in rows:
        assert len(row["teacher_probs"]) == 16
        assert sum(row["teacher_probs"]) == pytest.approx(1.0, abs=1e-5)
        assert all(probability >= 0.0 for probability in row["teacher_probs"])
        assert 0 <= row["position"] < 8
        assert isinstance(row["mode_id"], int)
        assert math.isfinite(row["interestingness_score"])
        assert row["reason_codes"]


def test_capture_summary_records_required_fields(tmp_path: Path) -> None:
    result = _capture(tmp_path, max_exemplars=3)
    summary = _json(result.capture_summary_path)

    assert summary["phase"] == "P143"
    assert summary["capture_method"] == "teacher_side_capture_skeleton_v0"
    assert summary["mode_discovery_method"] == "stat_bands_v0"
    assert summary["corridor_bounds_method"] == "minmax"
    assert summary["examples_processed"] == 4
    assert summary["target_positions_processed"] == 32
    assert summary["modes_discovered"] > 1
    assert summary["records_per_mode"]
    assert summary["max_exemplars"] == 3
    assert summary["exemplars_retained"] == 3
    assert summary["exemplar_reason_code_distribution"]
    assert summary["artifact_validated"] is True
    assert summary["teacher_required"] is False
    assert summary["hf_required"] is False


def test_cli_synthetic_capture_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli_artifact"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--synthetic-fixture",
            "tiny",
            "--output-dir",
            str(output_dir),
            "--vocab-size",
            "16",
            "--max-seq-len",
            "8",
            "--num-examples",
            "4",
            "--max-exemplars",
            "6",
            "--overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "status=pass" in completed.stdout
    assert validate_fingerprint_artifact(output_dir).ok is True
    assert (output_dir / "capture_summary.json").is_file()


def test_capture_output_is_consumable_by_p140_real_student_path(tmp_path: Path) -> None:
    result = _capture(tmp_path / "artifact", max_exemplars=4)
    forward = run_real_student_fingerprint_forward_smoke(
        RealStudentFingerprintForwardConfig(
            artifact_dir=result.output_dir,
            output_dir=tmp_path / "p140",
            batch_size=2,
        )
    )

    assert forward.status == "pass"
    assert forward.num_corridor_records == 32
    assert forward.teacher_required is False
    assert forward.optimizer_steps_completed == 0


def _capture(
    tmp_path: Path,
    *,
    max_exemplars: int,
    max_seq_len: int = 8,
    num_examples: int = 4,
) -> object:
    examples = build_synthetic_capture_examples(
        num_examples=num_examples,
        max_seq_len=max_seq_len,
        vocab_size=16,
    )
    return capture_fingerprint_artifact(
        _config(tmp_path, max_exemplars=max_exemplars),
        examples,
    )


def _config(tmp_path: Path, *, max_exemplars: int) -> FingerprintCaptureConfig:
    return FingerprintCaptureConfig(
        output_dir=tmp_path / "artifact",
        overwrite=True,
        exemplar_reservoir=FingerprintExemplarReservoirCaptureConfig(
            enabled=True,
            max_exemplars=max_exemplars,
        ),
    )


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path | None) -> list[dict]:
    assert path is not None
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _observed_mode_keys(path: Path) -> set[tuple[int, int, int]]:
    return {
        (
            mode["mode_key"]["entropy_bin"],
            mode["mode_key"]["top1_margin_bin"],
            mode["mode_key"]["top32_mass_bin"],
        )
        for mode in _json(path)["modes"]
    }
