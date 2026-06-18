from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.artifacts import (
    load_fingerprint_exemplars,
    load_fingerprint_targets,
    summarize_fingerprint_artifact,
    validate_fingerprint_artifact,
)
from qrwkv_xla.distill import (
    DISTILL_MODE_FINGERPRINT_CORRIDOR,
    DistillFingerprintConfig,
    DistillStageConfig,
    run_distill_stage,
)
from qrwkv_xla.fingerprint import (
    FingerprintCaptureConfig,
    FingerprintCaptureExample,
    FingerprintCorridorBoundsConfig,
    FingerprintExemplarReservoirCaptureConfig,
    FingerprintModeDiscoveryConfig,
    build_synthetic_capture_examples,
    capture_fingerprint_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_fingerprint_artifact.py"
STAT_BINS = FingerprintModeDiscoveryConfig(
    entropy_bins=(0.0, 0.5, 1.0, 1.3, 2.0, math.inf),
    top1_margin_bins=(0.0, 0.1, 0.5, 0.9, 1.0, math.inf),
    top32_mass_bins=(0.0, 0.99, math.inf),
)
MINMAX_BOUNDS = FingerprintCorridorBoundsConfig(
    method="minmax",
    min_width=1e-12,
)


def test_known_probability_stats_match_captured_mode_means(tmp_path: Path) -> None:
    result = _capture_known(tmp_path)
    modes_by_key = _modes_by_key(result.modes_path)
    expected_by_key: dict[tuple[int, int, int], list[dict[str, float]]] = {}

    for probs in _known_probs():
        expected = _expected_stats(probs)
        expected_by_key.setdefault(_expected_mode_key(expected), []).append(expected)

    for mode_key, expected_rows in expected_by_key.items():
        mode = modes_by_key[mode_key]
        for stat in expected_rows[0]:
            expected_mean = float(np.mean([row[stat] for row in expected_rows]))
            assert mode["bounds"][stat]["mean"] == pytest.approx(
                expected_mean,
                abs=1e-6,
            )


def test_stat_band_mode_assignment_and_counts_match_expected(tmp_path: Path) -> None:
    result = _capture_known(tmp_path)
    modes = _json(result.modes_path)["modes"]
    expected_keys = [
        _expected_mode_key(_expected_stats(probs)) for probs in _known_probs()
    ]

    assert {tuple(mode["mode_key"].values()) for mode in modes} == set(expected_keys)
    assert len(modes) == len(set(expected_keys))
    assert result.summary["records_per_mode"] == {
        str(mode["mode_id"]): expected_keys.count(tuple(mode["mode_key"].values()))
        for mode in modes
    }


def test_minmax_bounds_match_expected_values_within_mode(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        mode_discovery=FingerprintModeDiscoveryConfig(
            entropy_bins=(0.0, 10.0, math.inf),
            top1_margin_bins=(0.0, 1.0, math.inf),
            top32_mass_bins=(0.0, 0.99, math.inf),
        ),
        bounds=FingerprintCorridorBoundsConfig(method="minmax", min_width=1e-12),
    )
    result = capture_fingerprint_artifact(config, _known_examples())
    mode = _json(result.modes_path)["modes"][0]
    expected = [_expected_stats(probs) for probs in _known_probs()]

    for stat in ("entropy", "top1_margin", "top8_mass", "top32_mass", "tail_mass"):
        values = [row[stat] for row in expected]
        assert mode["bounds"][stat]["min"] == pytest.approx(min(values), abs=1e-6)
        assert mode["bounds"][stat]["max"] == pytest.approx(max(values), abs=1e-6)

    rows = _jsonl(result.targets_path)
    assert all(row["bounds"] == mode["bounds"] for row in rows)


def test_min_width_widens_identical_bounds_and_clamps_probabilities(
    tmp_path: Path,
) -> None:
    examples = (_example_from_probs("uniform", [_uniform_probs()] * 2),)
    result = capture_fingerprint_artifact(
        _config(
            tmp_path,
            bounds=FingerprintCorridorBoundsConfig(method="minmax", min_width=0.2),
        ),
        examples,
    )
    mode = _json(result.modes_path)["modes"][0]
    entropy = _expected_stats(_uniform_probs())["entropy"]

    assert mode["bounds"]["entropy"]["min"] == pytest.approx(entropy - 0.1, abs=1e-6)
    assert mode["bounds"]["entropy"]["max"] == pytest.approx(entropy + 0.1, abs=1e-6)
    assert mode["bounds"]["top1_margin"]["min"] == 0.0
    assert mode["bounds"]["top1_margin"]["max"] == pytest.approx(0.1, abs=1e-6)
    assert mode["bounds"]["top32_mass"]["min"] == pytest.approx(0.9, abs=1e-6)
    assert mode["bounds"]["top32_mass"]["max"] == 1.0


def test_quantile_bounds_match_numpy_quantile(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        mode_discovery=FingerprintModeDiscoveryConfig(
            entropy_bins=(0.0, 10.0, math.inf),
            top1_margin_bins=(0.0, 1.0, math.inf),
            top32_mass_bins=(0.0, 0.99, math.inf),
        ),
        bounds=FingerprintCorridorBoundsConfig(
            method="quantile",
            lower_quantile=0.25,
            upper_quantile=0.75,
            min_width=1e-12,
        ),
    )
    result = capture_fingerprint_artifact(config, _known_examples())
    mode = _json(result.modes_path)["modes"][0]
    expected = [_expected_stats(probs) for probs in _known_probs()]

    for stat in ("entropy", "top1_margin"):
        values = np.asarray([row[stat] for row in expected])
        assert mode["bounds"][stat]["min"] == pytest.approx(
            np.quantile(values, 0.25), abs=1e-6
        )
        assert mode["bounds"][stat]["max"] == pytest.approx(
            np.quantile(values, 0.75), abs=1e-6
        )
    assert validate_fingerprint_artifact(result.output_dir).ok is True


def test_exemplar_budget_zero_and_top_interestingness_order(tmp_path: Path) -> None:
    none = capture_fingerprint_artifact(
        _config(tmp_path / "none", max_exemplars=0),
        _known_examples(),
    )
    top_two = capture_fingerprint_artifact(
        _config(tmp_path / "top_two", max_exemplars=2),
        _known_examples(),
    )

    assert none.summary["exemplars_retained"] == 0
    assert _json(none.manifest_path)["exemplar_reservoir"]["num_records"] == 0
    retained_ids = [row["example_id"] for row in _jsonl(top_two.exemplars_path)]

    assert retained_ids == ["known-000000:pos-0", "known-000000:pos-2"]
    assert top_two.summary["exemplars_retained"] == 2


def test_stratified_exemplar_policy_covers_observed_modes_when_budget_allows(
    tmp_path: Path,
) -> None:
    result = capture_fingerprint_artifact(
        _config(
            tmp_path,
            max_exemplars=3,
            exemplars=FingerprintExemplarReservoirCaptureConfig(
                enabled=True,
                max_exemplars=3,
                selection_policy="stratified_interestingness_v0",
                per_mode_min=1,
            ),
        ),
        _known_examples(),
    )
    modes = _json(result.modes_path)["modes"]
    retained_modes = {row["mode_id"] for row in _jsonl(result.exemplars_path)}

    assert len(modes) == 3
    assert retained_modes == {mode["mode_id"] for mode in modes}
    assert (
        result.summary["exemplar_selection_policy"] == "stratified_interestingness_v0"
    )
    assert result.summary["exemplars_retained"] == 3


def test_dynamic_mode_count_differs_by_fixture_data(tmp_path: Path) -> None:
    one_mode = capture_fingerprint_artifact(
        _config(tmp_path / "one"),
        (_example_from_probs("uniform", [_uniform_probs()] * 4),),
    )
    multi_mode = _capture_known(tmp_path / "multi")

    assert one_mode.summary["modes_discovered"] == 1
    assert multi_mode.summary["modes_discovered"] > one_mode.summary["modes_discovered"]


def test_summary_and_artifact_summary_are_accurate(tmp_path: Path) -> None:
    result = _capture_known(tmp_path, max_exemplars=2)
    summary = _json(result.capture_summary_path)
    artifact = summarize_fingerprint_artifact(result.output_dir)
    reason_count = sum(summary["exemplar_reason_code_distribution"].values())

    assert summary["phase"] == "P143"
    assert summary["examples_processed"] == 1
    assert summary["target_positions_processed"] == 4
    assert summary["modes_discovered"] == 3
    assert summary["records_per_mode"]
    assert summary["max_exemplars"] == 2
    assert summary["exemplars_retained"] == 2
    assert reason_count >= summary["exemplars_retained"]
    assert summary["artifact_validated"] is True
    assert summary["capture_config"]
    assert artifact.num_corridor_records == 4
    assert artifact.has_exemplars is True
    assert artifact.num_exemplar_records == 2
    assert artifact.num_modes == 3


def test_artifact_validates_loads_and_runs_through_p141(tmp_path: Path) -> None:
    examples = build_synthetic_capture_examples(
        num_examples=2,
        max_seq_len=8,
        vocab_size=16,
    )
    result = capture_fingerprint_artifact(_config(tmp_path, max_exemplars=4), examples)

    assert validate_fingerprint_artifact(result.output_dir).ok is True
    assert load_fingerprint_targets(result.output_dir, batch_size=2).num_records == 16
    assert load_fingerprint_exemplars(result.output_dir, batch_size=2).num_records == 4

    run = run_distill_stage(
        DistillStageConfig(
            mode=DISTILL_MODE_FINGERPRINT_CORRIDOR,
            training=replace(DistillStageConfig().training, max_steps=1),
            optimizer=replace(DistillStageConfig().optimizer, learning_rate=0.01),
            fingerprint=DistillFingerprintConfig(
                artifact_dir=result.output_dir,
                batch_size=2,
                student_backend="current_qrwkv",
                output_dir=tmp_path / "runner_out",
            ),
        )
    )

    assert run.status == "pass"
    assert run.real_student_backend_integrated is True
    assert run.teacher_required is False
    assert math.isfinite(run.final_loss)


def test_cli_parity_smoke_supports_quantile_and_stratified_policy(
    tmp_path: Path,
) -> None:
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
            "--bounds-method",
            "quantile",
            "--lower-quantile",
            "0.25",
            "--upper-quantile",
            "0.75",
            "--exemplar-selection-policy",
            "stratified_interestingness_v0",
            "--per-mode-min",
            "1",
            "--overwrite",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "status=pass" in completed.stdout
    summary = _json(output_dir / "capture_summary.json")
    assert summary["corridor_bounds_method"] == "quantile"
    assert summary["exemplar_selection_policy"] == "stratified_interestingness_v0"
    assert validate_fingerprint_artifact(output_dir).ok is True


def _capture_known(
    tmp_path: Path,
    *,
    max_exemplars: int = 4,
) -> object:
    return capture_fingerprint_artifact(
        _config(tmp_path, max_exemplars=max_exemplars),
        _known_examples(),
    )


def _config(
    tmp_path: Path,
    *,
    max_exemplars: int = 4,
    mode_discovery: FingerprintModeDiscoveryConfig = STAT_BINS,
    bounds: FingerprintCorridorBoundsConfig = MINMAX_BOUNDS,
    exemplars: FingerprintExemplarReservoirCaptureConfig | None = None,
) -> FingerprintCaptureConfig:
    return FingerprintCaptureConfig(
        output_dir=tmp_path / "artifact",
        overwrite=True,
        mode_discovery=mode_discovery,
        corridor_bounds=bounds,
        exemplar_reservoir=exemplars
        or FingerprintExemplarReservoirCaptureConfig(
            enabled=True,
            max_exemplars=max_exemplars,
        ),
    )


def _known_examples() -> tuple[FingerprintCaptureExample, ...]:
    return (_example_from_probs("known-000000", _known_probs()),)


def _known_probs() -> list[list[float]]:
    return [
        _uniform_probs(),
        [0.97, 0.01, 0.01, 0.01],
        [0.50, 0.25, 0.15, 0.10],
        [0.94, 0.02, 0.02, 0.02],
    ]


def _uniform_probs() -> list[float]:
    return [0.25, 0.25, 0.25, 0.25]


def _example_from_probs(
    example_id: str,
    probs_by_position: list[list[float]],
) -> FingerprintCaptureExample:
    logits = np.log(np.asarray(probs_by_position, dtype=np.float32))
    return FingerprintCaptureExample(
        example_id=example_id,
        input_ids=tuple(range(len(probs_by_position))),
        logits=logits,
    )


def _expected_stats(probs: list[float]) -> dict[str, float]:
    values = np.asarray(probs, dtype=np.float64)
    sorted_values = np.flip(np.sort(values))
    return {
        "entropy": float(-np.sum(values * np.log(values))),
        "top1_margin": float(sorted_values[0] - sorted_values[1]),
        "top8_mass": 1.0,
        "top32_mass": 1.0,
        "tail_mass": 0.0,
    }


def _expected_mode_key(stats: dict[str, float]) -> tuple[int, int, int]:
    return (
        _bin_index(stats["entropy"], STAT_BINS.entropy_bins),
        _bin_index(stats["top1_margin"], STAT_BINS.top1_margin_bins),
        _bin_index(stats["top32_mass"], STAT_BINS.top32_mass_bins),
    )


def _bin_index(value: float, bins: tuple[float, ...]) -> int:
    index = int(np.searchsorted(np.asarray(bins), value, side="right") - 1)
    return max(0, min(index, len(bins) - 2))


def _modes_by_key(path: Path) -> dict[tuple[int, int, int], dict]:
    modes = {}
    for mode in _json(path)["modes"]:
        key = mode["mode_key"]
        modes[
            (
                key["entropy_bin"],
                key["top1_margin_bin"],
                key["top32_mass_bin"],
            )
        ] = mode
    return modes


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path | None) -> list[dict]:
    assert path is not None
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
