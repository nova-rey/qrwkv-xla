from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from qrwkv_xla.artifacts import (
    BEHAVIORAL_FINGERPRINT_VERSION,
    validate_fingerprint_artifact,
)

FIXTURE = (
    Path(__file__).parent / "fixtures" / "behavioral_fingerprint" / "v0_1_valid_tiny"
)


def test_valid_tiny_behavioral_fingerprint_artifact_passes() -> None:
    result = validate_fingerprint_artifact(FIXTURE)

    assert result.ok is True
    assert result.blockers == ()
    assert result.metadata["artifact_version"] == BEHAVIORAL_FINGERPRINT_VERSION
    assert result.metadata["shards"] == 1
    assert result.metadata["records"] == 8
    assert result.metadata["modes"] == 2


def test_cli_passes_valid_tiny_fixture() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_fingerprint_artifact.py",
            str(FIXTURE),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    assert "status=pass" in completed.stdout
    assert "artifact_type=behavioral_fingerprint" in completed.stdout
    assert "records=8" in completed.stdout


def test_missing_manifest_fails(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()

    _assert_fails(artifact, "missing manifest.json")


def test_wrong_artifact_type_fails(tmp_path: Path) -> None:
    artifact = _copy_fixture(tmp_path)
    _mutate_manifest(artifact, lambda manifest: manifest.update({"artifact_type": "x"}))

    _assert_fails(artifact, "artifact_type")


def test_missing_modes_file_fails(tmp_path: Path) -> None:
    artifact = _copy_fixture(tmp_path)
    (artifact / "modes.json").unlink()

    _assert_fails(artifact, "modes_file does not exist")


def test_missing_target_shard_fails(tmp_path: Path) -> None:
    artifact = _copy_fixture(tmp_path)
    (artifact / "targets" / "targets-00000.jsonl").unlink()

    _assert_fails(artifact, "target shard does not exist")


def test_invalid_mode_bounds_fail(tmp_path: Path) -> None:
    artifact = _copy_fixture(tmp_path)

    def mutate(modes: dict) -> None:
        modes["modes"][0]["bounds"]["top8_mass"]["min"] = 0.91
        modes["modes"][0]["bounds"]["top8_mass"]["max"] = 0.70

    _mutate_json(artifact / "modes.json", mutate)

    _assert_fails(artifact, "top8_mass")


def test_duplicate_mode_id_fails(tmp_path: Path) -> None:
    artifact = _copy_fixture(tmp_path)

    def mutate(modes: dict) -> None:
        modes["modes"][1]["mode_id"] = 0

    _mutate_json(artifact / "modes.json", mutate)

    _assert_fails(artifact, "duplicate mode_id=0")


def test_unknown_mode_id_in_target_row_fails(tmp_path: Path) -> None:
    artifact = _copy_fixture(tmp_path)
    _mutate_first_row(artifact, lambda row: row.update({"mode_id": 99}))

    _assert_fails(artifact, "unknown mode_id=99")


def test_malformed_jsonl_row_fails(tmp_path: Path) -> None:
    artifact = _copy_fixture(tmp_path)
    shard = artifact / "targets" / "targets-00000.jsonl"
    shard.write_text(
        shard.read_text(encoding="utf-8") + "{not-json}\n", encoding="utf-8"
    )
    _mutate_manifest(
        artifact,
        lambda manifest: manifest["target_shards"][0].update({"num_records": 9}),
    )
    _mutate_manifest(
        artifact,
        lambda manifest: manifest["sequence"].update({"target_positions": 9}),
    )

    _assert_fails(artifact, "malformed JSONL row")


def test_token_id_outside_vocab_fails(tmp_path: Path) -> None:
    artifact = _copy_fixture(tmp_path)
    _mutate_first_row(artifact, lambda row: row.update({"input_ids": [1, 128]}))

    _assert_fails(artifact, "outside vocabulary range")


def test_row_bound_outside_mode_bound_fails(tmp_path: Path) -> None:
    artifact = _copy_fixture(tmp_path)

    def mutate(row: dict) -> None:
        row["bounds"]["top8_mass"]["min"] = 0.80
        row["bounds"]["top8_mass"]["max"] = 0.99

    _mutate_first_row(artifact, mutate)

    _assert_fails(artifact, "outside mode bounds")


def test_shard_num_records_mismatch_fails(tmp_path: Path) -> None:
    artifact = _copy_fixture(tmp_path)
    _mutate_manifest(
        artifact,
        lambda manifest: manifest["target_shards"][0].update({"num_records": 7}),
    )

    _assert_fails(artifact, "target shard record count mismatch")


def _copy_fixture(tmp_path: Path) -> Path:
    artifact = tmp_path / "fingerprint_artifact"
    shutil.copytree(FIXTURE, artifact)
    return artifact


def _mutate_manifest(artifact: Path, mutate) -> None:
    _mutate_json(artifact / "manifest.json", mutate)


def _mutate_json(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _mutate_first_row(artifact: Path, mutate) -> None:
    shard = artifact / "targets" / "targets-00000.jsonl"
    rows = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines()]
    mutate(rows[0])
    shard.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _assert_fails(artifact: Path, expected: str) -> None:
    result = validate_fingerprint_artifact(artifact)
    assert result.ok is False
    assert any(expected in blocker for blocker in result.blockers), result.blockers
