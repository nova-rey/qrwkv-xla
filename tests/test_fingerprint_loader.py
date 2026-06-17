from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.artifacts import (
    FingerprintLoaderConfig,
    FingerprintTargetDataset,
    load_fingerprint_targets,
)

FIXTURE = (
    Path(__file__).parent / "fixtures" / "behavioral_fingerprint" / "v0_1_valid_tiny"
)


def test_valid_fixture_loads_records() -> None:
    dataset = load_fingerprint_targets(FIXTURE, batch_size=2)

    assert dataset.num_records == 8
    assert dataset.max_seq_len == 16
    assert dataset.vocab_size == 128
    assert dataset.tracked_stats == (
        "entropy",
        "top1_margin",
        "top8_mass",
        "top32_mass",
        "tail_mass",
    )
    records = list(dataset.iter_records())
    assert records[0].example_id == "ex000000"
    assert records[0].input_ids == (1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert records[0].weight == 1.0


def test_batch_shapes_are_fixed() -> None:
    batch = next(load_fingerprint_targets(FIXTURE, batch_size=2).iter_batches())

    assert batch.input_ids.shape == (2, 16)
    assert batch.input_ids.dtype == np.int32
    assert batch.position.shape == (2,)
    assert batch.mode_id.shape == (2,)
    assert batch.entropy_min.shape == (2,)
    assert batch.entropy_max.shape == (2,)
    assert batch.top1_margin_min.shape == (2,)
    assert batch.top1_margin_max.shape == (2,)
    assert batch.top8_mass_min.shape == (2,)
    assert batch.top8_mass_max.shape == (2,)
    assert batch.top32_mass_min.shape == (2,)
    assert batch.top32_mass_max.shape == (2,)
    assert batch.tail_mass_min.shape == (2,)
    assert batch.tail_mass_max.shape == (2,)
    assert batch.weight.shape == (2,)
    assert batch.weight.dtype == np.float32


def test_full_iteration_drop_remainder_false() -> None:
    batches = list(load_fingerprint_targets(FIXTURE, batch_size=3).iter_batches())

    assert [batch.input_ids.shape[0] for batch in batches] == [3, 3, 2]
    assert sum(batch.input_ids.shape[0] for batch in batches) == 8


def test_full_iteration_drop_remainder_true() -> None:
    batches = list(
        load_fingerprint_targets(
            FIXTURE,
            batch_size=3,
            drop_remainder=True,
        ).iter_batches()
    )

    assert [batch.input_ids.shape[0] for batch in batches] == [3, 3]
    assert sum(batch.input_ids.shape[0] for batch in batches) == 6


def test_max_records_limits_iteration() -> None:
    dataset = load_fingerprint_targets(FIXTURE, batch_size=2, max_records=5)
    batches = list(dataset.iter_batches())

    assert dataset.num_records == 5
    assert [batch.input_ids.shape[0] for batch in batches] == [2, 2, 1]
    assert sum(batch.input_ids.shape[0] for batch in batches) == 5


def test_deterministic_shuffle() -> None:
    ordered = _example_order(load_fingerprint_targets(FIXTURE, batch_size=2))
    same_seed_a = _example_order(
        load_fingerprint_targets(FIXTURE, batch_size=2, shuffle=True, seed=7)
    )
    same_seed_b = _example_order(
        load_fingerprint_targets(FIXTURE, batch_size=2, shuffle=True, seed=7)
    )
    different_seed = _example_order(
        load_fingerprint_targets(FIXTURE, batch_size=2, shuffle=True, seed=8)
    )

    assert ordered == tuple(f"ex00000{index}" for index in range(8))
    assert same_seed_a == same_seed_b
    assert same_seed_a != ordered
    assert different_seed != same_seed_a


def test_validation_hook_fails_before_loading_records(tmp_path: Path) -> None:
    artifact = _copy_fixture(tmp_path)
    _mutate_manifest(
        artifact,
        lambda manifest: manifest.update({"artifact_type": "not_behavioral"}),
    )

    with pytest.raises(ValueError, match="validation failed"):
        FingerprintTargetDataset(
            FingerprintLoaderConfig(artifact_dir=artifact, batch_size=2)
        )


def test_loader_rejects_variable_length_rows_without_pad_policy(tmp_path: Path) -> None:
    artifact = _copy_fixture(tmp_path)
    shard = artifact / "targets" / "targets-00000.jsonl"
    rows = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines()]
    rows[0]["input_ids"] = [1, 2, 3]
    shard.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fixed-length input_ids"):
        load_fingerprint_targets(artifact, batch_size=2)


def test_fixture_jsonl_has_one_object_per_physical_line() -> None:
    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    for shard in manifest["target_shards"]:
        shard_path = FIXTURE / shard["path"]
        lines = shard_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == shard["num_records"]
        for line in lines:
            payload = json.loads(line)
            assert isinstance(payload, dict)


def test_inspect_cli_reports_first_batch() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_fingerprint_targets.py",
            str(FIXTURE),
            "--batch-size",
            "2",
            "--max-batches",
            "1",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    assert "artifact_type=behavioral_fingerprint" in completed.stdout
    assert "num_records=8" in completed.stdout
    assert "batch_0_input_ids_shape=(2, 16)" in completed.stdout
    assert "batch_0_mode_ids=[0, 0]" in completed.stdout


def _example_order(dataset: FingerprintTargetDataset) -> tuple[str, ...]:
    return tuple(record.example_id for record in dataset.iter_records())


def _copy_fixture(tmp_path: Path) -> Path:
    artifact = tmp_path / "fingerprint_artifact"
    shutil.copytree(FIXTURE, artifact)
    return artifact


def _mutate_manifest(artifact: Path, mutate) -> None:
    path = artifact / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
