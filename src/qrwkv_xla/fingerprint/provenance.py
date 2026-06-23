from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

from qrwkv_xla.artifacts import load_fingerprint_targets
from qrwkv_xla.artifacts._json import read_json_object


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def hash_checkpoint_bundle(checkpoint_dir: Path) -> dict[str, str]:
    metadata_path = checkpoint_dir / "checkpoint.json"
    params_path = checkpoint_dir / "params.npz"
    metadata_sha256 = file_sha256(metadata_path)
    params_sha256 = file_sha256(params_path)
    return {
        "checkpoint_metadata_path": str(metadata_path),
        "checkpoint_metadata_sha256": metadata_sha256,
        "params_path": str(params_path),
        "params_sha256": params_sha256,
        "checkpoint_bundle_sha256": stable_hash(
            {
                "checkpoint_metadata_sha256": metadata_sha256,
                "params_sha256": params_sha256,
            }
        ),
    }


def parameter_fingerprint(params: Any) -> str:
    entries = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        array = np.ascontiguousarray(np.asarray(leaf))
        entries.append(
            (
                jax.tree_util.keystr(path),
                str(array.shape),
                str(array.dtype),
                array,
            )
        )
    digest = hashlib.sha256()
    for path, shape, dtype, array in sorted(entries, key=lambda item: item[0]):
        for value in (path, shape, dtype):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        digest.update(array.tobytes())
    return f"sha256:{digest.hexdigest()}"


def ordered_artifact_examples(
    artifact_dir: Path,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    dataset = load_fingerprint_targets(artifact_dir, batch_size=1)
    output: dict[str, tuple[int, ...]] = {}
    for record in dataset.iter_records():
        prior = output.setdefault(record.example_id, record.input_ids)
        if prior != record.input_ids:
            raise ValueError(f"inconsistent tokenized inputs for {record.example_id}")
    if not output:
        raise ValueError("fingerprint artifact contains zero source examples")
    return tuple(output.items())


def join_sources_by_example_id(
    source_file: Path,
    artifact_example_ids: tuple[str, ...],
    *,
    allow_legacy_positional_source_join: bool = False,
) -> dict[str, Any]:
    rows = []
    for line_number, line in enumerate(
        source_file.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise ValueError(
                f"source line {line_number} must contain a string text field"
            )
        rows.append(payload)
    if not rows:
        raise ValueError("source file contains zero text records")

    explicit_ids = [row.get("example_id") for row in rows]
    if all(isinstance(value, str) and value.strip() for value in explicit_ids):
        source_by_id: dict[str, str] = {}
        for row in rows:
            example_id = str(row["example_id"])
            if example_id in source_by_id:
                raise ValueError(f"duplicate source example_id: {example_id}")
            source_by_id[example_id] = str(row["text"]).strip()
        missing = [
            example_id
            for example_id in artifact_example_ids
            if example_id not in source_by_id
        ]
        if missing:
            raise ValueError(
                "source file missing artifact example IDs: " + ", ".join(missing)
            )
        ordered_texts = [
            source_by_id[example_id] for example_id in artifact_example_ids
        ]
        return {
            "source_join_kind": "example_id",
            "source_join_complete": True,
            "lineage_confidence": "full",
            "publication_grade_lineage": True,
            "ordered_source_texts": ordered_texts,
            "warnings": [],
        }

    if not allow_legacy_positional_source_join:
        raise ValueError(
            "source rows require explicit example_id; pass "
            "allow_legacy_positional_source_join only for legacy fixtures"
        )
    if any(value is not None for value in explicit_ids):
        raise ValueError(
            "source rows must either all include example_id or all omit it"
        )
    if len(rows) < len(artifact_example_ids):
        raise ValueError("legacy source file has fewer rows than artifact examples")
    return {
        "source_join_kind": "legacy_positional",
        "source_join_complete": True,
        "lineage_confidence": "reduced",
        "publication_grade_lineage": False,
        "ordered_source_texts": [
            str(row["text"]).strip() for row in rows[: len(artifact_example_ids)]
        ],
        "warnings": [
            "legacy positional source join enabled; lineage confidence reduced"
        ],
    }


def build_artifact_source_lineage(
    artifact_dir: Path,
    source_file: Path,
    *,
    allow_legacy_positional_source_join: bool = False,
) -> dict[str, Any]:
    examples = ordered_artifact_examples(artifact_dir)
    example_ids = tuple(item[0] for item in examples)
    token_sequences = tuple(item[1] for item in examples)
    source_join = join_sources_by_example_id(
        source_file,
        example_ids,
        allow_legacy_positional_source_join=allow_legacy_positional_source_join,
    )
    manifest = read_json_object(artifact_dir / "manifest.json")
    capture_summary = _optional_json(artifact_dir / "capture_summary.json")
    subset_source = _optional_json(artifact_dir / "budget_subset_source.json")
    source_text_hashes = [
        stable_hash(text) for text in source_join["ordered_source_texts"]
    ]
    token_sequence_hashes = [stable_hash(sequence) for sequence in token_sequences]
    return {
        "source_file": str(source_file),
        "source_file_sha256": file_sha256(source_file),
        "ordered_example_ids": list(example_ids),
        "ordered_example_ids_sha256": stable_hash(example_ids),
        "source_example_set_sha256": stable_hash(sorted(example_ids)),
        "source_text_hashes": source_text_hashes,
        "ordered_source_text_sha256": stable_hash(source_text_hashes),
        "token_sequence_hashes": token_sequence_hashes,
        "tokenized_inputs_sha256": stable_hash(token_sequence_hashes),
        "capture_config_sha256": subset_source.get("source_capture_config_sha256")
        or stable_hash(
            {
                "sequence": manifest.get("sequence"),
                "stats": manifest.get("stats"),
                "bounds_method": capture_summary.get("corridor_bounds_method"),
                "exemplar_selection_policy": capture_summary.get(
                    "exemplar_selection_policy"
                ),
                "max_exemplars": capture_summary.get("max_exemplars"),
            }
        ),
        "teacher_identity_sha256": subset_source.get("source_teacher_identity_sha256")
        or stable_hash(manifest.get("teacher", {})),
        "artifact_manifest_sha256": file_sha256(artifact_dir / "manifest.json"),
        "source_join_kind": source_join["source_join_kind"],
        "source_join_complete": source_join["source_join_complete"],
        "lineage_confidence": source_join["lineage_confidence"],
        "publication_grade_lineage": source_join["publication_grade_lineage"],
        "warnings": source_join["warnings"],
    }


def _optional_json(path: Path) -> dict[str, Any]:
    return read_json_object(path) if path.is_file() else {}
