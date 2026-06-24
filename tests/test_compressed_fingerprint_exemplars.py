from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from qrwkv_xla.artifacts import (
    load_fingerprint_exemplars,
    validate_fingerprint_artifact,
)
from qrwkv_xla.distill.losses import cascaded_soft_labels_loss
from qrwkv_xla.fingerprint.capture import (
    FingerprintCaptureConfig,
    FingerprintCaptureExample,
    FingerprintExemplarReservoirCaptureConfig,
    capture_fingerprint_artifact,
)
from qrwkv_xla.training import compute_fingerprint_exemplar_loss

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_real_teacher_fingerprint_artifact.py"


def test_compressed_capture_load_loss_gradient_and_optimizer_step(
    tmp_path: Path,
) -> None:
    artifact = _build_compressed_artifact(tmp_path)
    validation = validate_fingerprint_artifact(artifact)
    dataset = load_fingerprint_exemplars(artifact, batch_size=2)
    batch = next(dataset.iter_batches())
    student_logits = jnp.linspace(
        -0.5,
        0.5,
        batch.input_ids.shape[0] * dataset.vocab_size,
        dtype=jnp.float32,
    ).reshape((batch.input_ids.shape[0], dataset.vocab_size))

    output = compute_fingerprint_exemplar_loss(student_logits, batch)
    reference = cascaded_soft_labels_loss(
        student_logits[:, None, :],
        jnp.asarray(batch.top_token_ids)[:, None, :],
        jnp.asarray(batch.top_log_probs)[:, None, :],
        jnp.ones((student_logits.shape[0], 1), dtype=jnp.float32),
        tail_mass=jnp.asarray(batch.tail_mass)[:, None],
        bucket_mass=jnp.asarray(batch.bucket_mass)[:, None, :],
        bucket_edges=jnp.asarray(batch.bucket_edges),
        top_mass=jnp.asarray(batch.top_mass)[:, None],
        teacher_entropy=jnp.asarray(batch.teacher_entropy)[:, None],
    )
    gradient = jax.grad(
        lambda logits: compute_fingerprint_exemplar_loss(logits, batch).loss
    )(student_logits)
    updated = student_logits - 0.01 * gradient

    assert validation.ok
    assert batch.target_type == "cascaded_soft_labels_v1"
    assert batch.teacher_probs is None
    assert np.isfinite(float(output.loss))
    assert np.isclose(float(output.loss), float(reference.total_loss), atol=1e-6)
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.any(np.asarray(updated) != np.asarray(student_logits))


def test_compressed_rows_and_manifest_bind_the_encoding_contract(
    tmp_path: Path,
) -> None:
    artifact = _build_compressed_artifact(tmp_path)
    manifest = _read_json(artifact / "manifest.json")
    contract = manifest["exemplar_reservoir"]["encoding_contract"]
    rows = _read_rows(artifact)

    assert contract["kind"] == "cascaded_soft_labels_v1"
    assert contract["top_k"] == 4
    assert rows
    assert all("teacher_probs" not in row for row in rows)
    assert all(len(row["top_token_ids"]) <= 4 for row in rows)
    assert all(
        np.isclose(row["top_mass"] + sum(row["bucket_mass"]), 1.0, atol=2e-3)
        for row in rows
    )


def test_compressed_validator_rejects_dense_payload_and_contract_mismatch(
    tmp_path: Path,
) -> None:
    artifact = _build_compressed_artifact(tmp_path)
    shard = artifact / "exemplars" / "exemplars-00000.jsonl"
    rows = [json.loads(line) for line in shard.read_text().splitlines()]
    rows[0]["teacher_probs"] = [1.0] + [0.0] * 11
    _write_rows(shard, rows)

    validation = validate_fingerprint_artifact(artifact)
    assert not validation.ok
    assert any("teacher_probs" in blocker for blocker in validation.blockers)

    artifact = _build_compressed_artifact(tmp_path, name="contract-mismatch")
    manifest = _read_json(artifact / "manifest.json")
    manifest["exemplar_reservoir"]["encoding_contract"]["top_k"] = 3
    (artifact / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    validation = validate_fingerprint_artifact(artifact)
    assert not validation.ok
    assert any("configured K" in blocker for blocker in validation.blockers)


def test_real_teacher_cli_exposes_compressed_payload_controls() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--exemplar-target-type" in completed.stdout
    assert "--exemplar-top-k" in completed.stdout
    assert "--exemplar-bucket-edges" in completed.stdout
    assert "--exemplar-top-log-probs-dtype" in completed.stdout


def _build_compressed_artifact(tmp_path: Path, *, name: str = "compressed") -> Path:
    output = tmp_path / name
    logits = np.asarray(
        [
            [3.0, 2.0, 1.0, 0.5, 0.1, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2],
            [-0.4, 0.2, 1.4, 0.3, 0.8, -0.2, 0.0, 0.7, -0.6, 0.1, -0.1, 0.4],
        ],
        dtype=np.float32,
    )
    examples = (
        FingerprintCaptureExample(
            example_id="compressed-000",
            input_ids=(1, 2),
            logits=logits,
        ),
    )
    capture_fingerprint_artifact(
        FingerprintCaptureConfig(
            output_dir=output,
            exemplar_reservoir=FingerprintExemplarReservoirCaptureConfig(
                max_exemplars=2,
                payload_type="cascaded_soft_labels_v1",
                top_k=4,
                shard_size=1,
            ),
        ),
        examples,
    )
    return output


def _read_rows(artifact: Path) -> list[dict]:
    manifest = _read_json(artifact / "manifest.json")
    rows = []
    for shard in manifest["exemplar_reservoir"]["shards"]:
        path = artifact / shard["path"]
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    return rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
