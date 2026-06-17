from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qrwkv_xla.artifacts import (
    FingerprintExemplarDataset,
    FingerprintExemplarLoaderConfig,
    load_fingerprint_exemplars,
    validate_fingerprint_artifact,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "behavioral_fingerprint"
FIXTURE = FIXTURE_ROOT / "v0_1_with_exemplars_tiny"
NO_EXEMPLARS_FIXTURE = FIXTURE_ROOT / "v0_1_valid_tiny"


def test_exemplar_fixture_validates() -> None:
    result = validate_fingerprint_artifact(FIXTURE)

    assert result.ok is True
    assert result.blockers == ()
    assert result.metadata["exemplar_reservoir_enabled"] is True
    assert result.metadata["exemplar_payload_type"] == "dense_probs"
    assert result.metadata["exemplar_records"] == 4
    assert result.metadata["exemplar_shards"] == 1


def test_artifact_without_exemplars_still_valid_but_require_loader_fails() -> None:
    result = validate_fingerprint_artifact(NO_EXEMPLARS_FIXTURE)

    assert result.ok is True
    with pytest.raises(ValueError, match="no exemplar_reservoir"):
        load_fingerprint_exemplars(NO_EXEMPLARS_FIXTURE, batch_size=2)

    dataset = load_fingerprint_exemplars(
        NO_EXEMPLARS_FIXTURE,
        batch_size=2,
        require_exemplars=False,
    )
    assert dataset.num_records == 0
    assert list(dataset.iter_batches()) == []


def test_exemplar_loader_shapes_and_metadata() -> None:
    dataset = load_fingerprint_exemplars(FIXTURE, batch_size=2)
    batch = next(dataset.iter_batches())

    assert dataset.num_records == 4
    assert dataset.vocab_size == 16
    assert dataset.max_seq_len == 8
    assert batch.input_ids.shape == (2, 8)
    assert batch.input_ids.dtype == np.int32
    assert batch.position.shape == (2,)
    assert batch.teacher_probs.shape == (2, 16)
    assert batch.teacher_probs.dtype == np.float32
    assert batch.weight.shape == (2,)
    assert batch.mode_id.tolist() == [0, 0]
    assert batch.reason_codes == (("top1_anchor",), ("smoothed_peak", "mode_boundary"))
    assert batch.example_id == ("p137e000", "p137e001")


def test_exemplar_loader_optional_mode_id_uses_sentinel() -> None:
    batch = next(
        load_fingerprint_exemplars(FIXTURE, batch_size=4, max_records=4).iter_batches()
    )

    assert batch.mode_id.tolist() == [0, 0, 1, -1]
    assert np.isnan(batch.interestingness_score[-1])


def test_exemplar_loader_deterministic_shuffle() -> None:
    ordered = _example_order(load_fingerprint_exemplars(FIXTURE, batch_size=2))
    same_seed_a = _example_order(
        load_fingerprint_exemplars(FIXTURE, batch_size=2, shuffle=True, seed=7)
    )
    same_seed_b = _example_order(
        load_fingerprint_exemplars(FIXTURE, batch_size=2, shuffle=True, seed=7)
    )
    different_seed = _example_order(
        load_fingerprint_exemplars(FIXTURE, batch_size=2, shuffle=True, seed=8)
    )

    assert ordered == tuple(f"p137e00{index}" for index in range(4))
    assert same_seed_a == same_seed_b
    assert same_seed_a != ordered
    assert different_seed != same_seed_a


def test_exemplar_max_records_and_drop_remainder() -> None:
    dataset = load_fingerprint_exemplars(
        FIXTURE,
        batch_size=2,
        max_records=3,
        drop_remainder=True,
    )
    batches = list(dataset.iter_batches())

    assert dataset.num_records == 3
    assert [batch.input_ids.shape[0] for batch in batches] == [2]


def test_exemplar_loader_validation_hook_fails_before_loading(tmp_path: Path) -> None:
    artifact = _copy_fixture(tmp_path)
    _mutate_manifest(
        artifact,
        lambda manifest: manifest["exemplar_reservoir"].update(
            {"payload_type": "dense_logits"}
        ),
    )

    with pytest.raises(ValueError, match="validation failed"):
        FingerprintExemplarDataset(
            FingerprintExemplarLoaderConfig(artifact_dir=artifact, batch_size=2)
        )


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda artifact: _mutate_first_exemplar_row(
                artifact,
                lambda row: row.update({"teacher_probs": [1.0, 0.0]}),
            ),
            "len(teacher_probs)",
        ),
        (
            lambda artifact: _mutate_first_exemplar_row(
                artifact,
                lambda row: row["teacher_probs"].__setitem__(1, 0.25),
            ),
            "sum to 1.0",
        ),
        (
            lambda artifact: _mutate_first_exemplar_row(
                artifact,
                lambda row: row["teacher_probs"].__setitem__(0, -1.0),
            ),
            "non-negative",
        ),
        (
            lambda artifact: _mutate_first_exemplar_row(
                artifact,
                lambda row: row.update({"mode_id": 99}),
            ),
            "unknown mode_id=99",
        ),
        (
            lambda artifact: (
                artifact / "exemplars" / "exemplars-00000.jsonl"
            ).unlink(),
            "exemplar shard does not exist",
        ),
    ],
)
def test_bad_exemplar_payloads_fail_validation(
    tmp_path: Path, mutate, expected
) -> None:
    artifact = _copy_fixture(tmp_path)

    mutate(artifact)

    result = validate_fingerprint_artifact(artifact)
    assert result.ok is False
    assert any(expected in blocker for blocker in result.blockers), result.blockers


def test_inspect_exemplars_cli_reports_first_batch() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_fingerprint_exemplars.py",
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
    assert "payload_type=dense_probs" in completed.stdout
    assert "num_exemplars=4" in completed.stdout
    assert "batch_0_teacher_probs_shape=(2, 16)" in completed.stdout
    assert "top1_anchor" in completed.stdout


def _example_order(dataset: FingerprintExemplarDataset) -> tuple[str, ...]:
    return tuple(record.example_id for record in dataset.iter_records())


def _copy_fixture(tmp_path: Path) -> Path:
    artifact = tmp_path / "fingerprint_artifact"
    shutil.copytree(FIXTURE, artifact)
    return artifact


def _mutate_manifest(artifact: Path, mutate) -> None:
    _mutate_json(artifact / "manifest.json", mutate)


def _mutate_first_exemplar_row(artifact: Path, mutate) -> None:
    shard = artifact / "exemplars" / "exemplars-00000.jsonl"
    rows = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines()]
    mutate(rows[0])
    shard.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _mutate_json(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
