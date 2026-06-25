from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from qrwkv_xla.artifacts._json import read_json_object

BEHAVIORAL_FINGERPRINT_ARTIFACT_TYPE = "behavioral_fingerprint"
BEHAVIORAL_FINGERPRINT_VERSION = "0.1"
SUPPORTED_BEHAVIORAL_FINGERPRINT_VERSIONS = (BEHAVIORAL_FINGERPRINT_VERSION,)
PROBABILITY_LIKE_STATS = frozenset(
    {"top1_margin", "top8_mass", "top32_mass", "tail_mass"}
)
TARGET_PAYLOAD_LEGACY_JSONL = "legacy_jsonl"
TARGET_PAYLOAD_PACKED_CORRIDOR_V1 = "packed_corridor_v1"
PACKED_TARGET_ARRAYS = {
    "examples_input_ids": 2,
    "position_example_index": 1,
    "position": 1,
    "mode_id": 1,
    "weight": 1,
}


@dataclass(frozen=True)
class FingerprintManifest:
    artifact_type: str
    artifact_version: str
    created_by: str
    teacher: dict[str, Any]
    sequence: dict[str, Any]
    stats: dict[str, Any]
    modes_file: str
    target_shards: tuple[dict[str, Any], ...]
    target_payload: dict[str, Any] | None = None
    exemplar_reservoir: dict[str, Any] | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> FingerprintManifest:
        exemplar_reservoir = payload.get("exemplar_reservoir")
        return cls(
            artifact_type=str(payload.get("artifact_type", "")),
            artifact_version=str(payload.get("artifact_version", "")),
            created_by=str(payload.get("created_by", "")),
            teacher=_mapping_or_empty(payload.get("teacher")),
            sequence=_mapping_or_empty(payload.get("sequence")),
            stats=_mapping_or_empty(payload.get("stats")),
            modes_file=str(payload.get("modes_file", "")),
            target_shards=tuple(
                _mapping_or_empty(item) for item in payload.get("target_shards", ())
            ),
            target_payload=(
                _mapping_or_empty(payload.get("target_payload"))
                if payload.get("target_payload") is not None
                else None
            ),
            exemplar_reservoir=(
                _mapping_or_empty(exemplar_reservoir)
                if exemplar_reservoir is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "pass" if self.ok else "fail"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status
        return payload


def validate_fingerprint_artifact(path: str | Path) -> ValidationResult:
    root = Path(path)
    blockers: list[str] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {
        "artifact_type": BEHAVIORAL_FINGERPRINT_ARTIFACT_TYPE,
        "artifact_version": None,
        "shards": 0,
        "records": 0,
        "modes": 0,
        "exemplar_reservoir_enabled": False,
        "exemplar_payload_type": None,
        "exemplar_records": 0,
        "exemplar_shards": 0,
    }

    if not root.is_dir():
        return _result(blockers=[f"artifact path is not a directory: {root}"])

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return _result(blockers=[f"missing manifest.json: {manifest_path}"])

    try:
        manifest_payload = read_json_object(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _result(blockers=[f"manifest.json invalid: {exc}"])

    manifest = FingerprintManifest.from_payload(manifest_payload)
    metadata["artifact_version"] = manifest.artifact_version
    target_payload_kind = _target_payload_kind(manifest)
    blockers.extend(_validate_manifest(manifest, target_payload_kind))
    metadata["target_payload_type"] = target_payload_kind

    tracked_stats = _tracked_stats(manifest.stats)
    vocab_size = _positive_int(manifest.teacher.get("vocab_size"))
    max_seq_len = _positive_int(manifest.sequence.get("max_seq_len"))

    modes_by_id: dict[int, dict[str, Any]] = {}
    modes_file = root / manifest.modes_file if manifest.modes_file else root / ""
    if manifest.modes_file and modes_file.is_file():
        try:
            modes_payload = read_json_object(modes_file)
            mode_blockers, modes_by_id = _validate_modes(modes_payload, tracked_stats)
            blockers.extend(mode_blockers)
            metadata["modes"] = len(modes_by_id)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(f"{manifest.modes_file} invalid: {exc}")
    elif manifest.modes_file:
        blockers.append(f"modes_file does not exist: {manifest.modes_file}")

    if target_payload_kind == TARGET_PAYLOAD_PACKED_CORRIDOR_V1:
        metadata["shards"] = 1
        total_records, packed_blockers = _validate_packed_target_payload(
            root,
            manifest.target_payload,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            modes_by_id=modes_by_id,
        )
        blockers.extend(packed_blockers)
    else:
        metadata["shards"] = len(manifest.target_shards)
        total_records = 0
        for shard_index, shard in enumerate(manifest.target_shards):
            shard_path_value = shard.get("path")
            if not isinstance(shard_path_value, str) or not shard_path_value.strip():
                blockers.append(f"target_shards[{shard_index}] missing non-empty path")
                continue
            shard_path = root / shard_path_value
            expected_records = _non_negative_int(shard.get("num_records"))
            if expected_records is None:
                blockers.append(
                    f"target_shards[{shard_index}] num_records must be a "
                    "non-negative integer"
                )
            if not shard_path.is_file():
                blockers.append(f"target shard does not exist: {shard_path_value}")
                continue
            actual_records, shard_blockers = _validate_target_shard(
                shard_path,
                root=root,
                tracked_stats=tracked_stats,
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
                modes_by_id=modes_by_id,
            )
            total_records += actual_records
            blockers.extend(shard_blockers)
            if expected_records is not None and actual_records != expected_records:
                blockers.append(
                    f"target shard record count mismatch for {shard_path_value}: "
                    f"expected {expected_records}, got {actual_records}"
                )

    metadata["records"] = total_records
    target_positions = manifest.sequence.get("target_positions")
    if target_positions is not None:
        expected_total = _non_negative_int(target_positions)
        if expected_total is None:
            blockers.append("sequence.target_positions must be a non-negative integer")
        elif total_records != expected_total:
            blockers.append(
                "sequence.target_positions mismatch: "
                f"expected {expected_total}, got {total_records}"
            )

    exemplar_metadata, exemplar_blockers = _validate_exemplar_reservoir(
        root,
        manifest.exemplar_reservoir,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        modes_by_id=modes_by_id,
    )
    metadata.update(exemplar_metadata)
    blockers.extend(exemplar_blockers)

    return _result(blockers=blockers, warnings=warnings, metadata=metadata)


def _validate_manifest(
    manifest: FingerprintManifest, target_payload_kind: str
) -> list[str]:
    blockers: list[str] = []
    if manifest.artifact_type != BEHAVIORAL_FINGERPRINT_ARTIFACT_TYPE:
        blockers.append(
            "manifest artifact_type must be "
            f"{BEHAVIORAL_FINGERPRINT_ARTIFACT_TYPE!r}, got {manifest.artifact_type!r}"
        )
    if not manifest.artifact_version:
        blockers.append("manifest artifact_version is required")
    elif manifest.artifact_version not in SUPPORTED_BEHAVIORAL_FINGERPRINT_VERSIONS:
        blockers.append(
            f"manifest artifact_version unsupported: {manifest.artifact_version!r}"
        )
    if not manifest.created_by.strip():
        blockers.append("manifest created_by must be non-empty")
    if not manifest.teacher:
        blockers.append("manifest teacher metadata is required")
    else:
        for key in ("model_name", "tokenizer_name", "dtype"):
            if not str(manifest.teacher.get(key, "")).strip():
                blockers.append(f"manifest teacher.{key} must be non-empty")
        if _positive_int(manifest.teacher.get("vocab_size")) is None:
            blockers.append("manifest teacher.vocab_size must be a positive integer")
    if not manifest.sequence:
        blockers.append("manifest sequence metadata is required")
    else:
        if _positive_int(manifest.sequence.get("max_seq_len")) is None:
            blockers.append("manifest sequence.max_seq_len must be a positive integer")
    tracked = _tracked_stats(manifest.stats)
    if not tracked:
        blockers.append("manifest stats.tracked must be a non-empty list")
    if not manifest.modes_file.strip():
        blockers.append("manifest modes_file must be non-empty")
    if (
        target_payload_kind == TARGET_PAYLOAD_LEGACY_JSONL
        and not manifest.target_shards
    ):
        blockers.append("manifest target_shards must be non-empty")
    if target_payload_kind not in {
        TARGET_PAYLOAD_LEGACY_JSONL,
        TARGET_PAYLOAD_PACKED_CORRIDOR_V1,
    }:
        blockers.append(
            f"manifest target_payload.kind unsupported: {target_payload_kind!r}"
        )
    if target_payload_kind == TARGET_PAYLOAD_PACKED_CORRIDOR_V1:
        if manifest.target_shards:
            blockers.append(
                "manifest target_shards must be empty for packed_corridor_v1"
            )
        if not isinstance(manifest.target_payload, dict) or not manifest.target_payload:
            blockers.append("manifest target_payload must describe packed targets")
        elif manifest.target_payload.get("kind") != TARGET_PAYLOAD_PACKED_CORRIDOR_V1:
            blockers.append("manifest target_payload.kind mismatch")
    return blockers


def _target_payload_kind(manifest: FingerprintManifest) -> str:
    if manifest.target_payload is None:
        return TARGET_PAYLOAD_LEGACY_JSONL
    kind = manifest.target_payload.get("kind")
    if kind in {TARGET_PAYLOAD_LEGACY_JSONL, TARGET_PAYLOAD_PACKED_CORRIDOR_V1}:
        return str(kind)
    return str(kind or "")


def _validate_modes(
    payload: dict[str, Any],
    tracked_stats: tuple[str, ...],
) -> tuple[list[str], dict[int, dict[str, Any]]]:
    blockers: list[str] = []
    modes = payload.get("modes")
    if not isinstance(modes, list) or not modes:
        return ["modes.json modes must be a non-empty list"], {}

    seen: set[int] = set()
    modes_by_id: dict[int, dict[str, Any]] = {}
    for index, mode in enumerate(modes):
        source = f"modes.json modes[{index}]"
        if not isinstance(mode, dict):
            blockers.append(f"{source} must be an object")
            continue
        mode_id = mode.get("mode_id")
        if not isinstance(mode_id, int):
            blockers.append(f"{source}.mode_id must be an integer")
            continue
        if mode_id in seen:
            blockers.append(f"duplicate mode_id={mode_id} in modes.json")
        seen.add(mode_id)
        modes_by_id[mode_id] = mode
        if not str(mode.get("name", "")).strip():
            blockers.append(f"{source}.name must be non-empty")
        bounds = mode.get("bounds")
        if not isinstance(bounds, dict):
            blockers.append(f"{source}.bounds must be an object")
            continue
        for stat in tracked_stats:
            bound = bounds.get(stat)
            blockers.extend(_validate_bound(bound, f"{source}.bounds.{stat}", stat))
    return blockers, modes_by_id


def _validate_target_shard(
    shard_path: Path,
    *,
    root: Path,
    tracked_stats: tuple[str, ...],
    vocab_size: int | None,
    max_seq_len: int | None,
    modes_by_id: dict[int, dict[str, Any]],
) -> tuple[int, list[str]]:
    blockers: list[str] = []
    record_count = 0
    relative_path = _relative_path(shard_path, root)
    for line_number, line in enumerate(
        shard_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        record_count += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            blockers.append(
                f"malformed JSONL row in {relative_path} line {line_number}: {exc.msg}"
            )
            continue
        if not isinstance(row, dict):
            blockers.append(
                f"row in {relative_path} line {line_number} must be an object"
            )
            continue
        blockers.extend(
            _validate_target_row(
                row,
                source=f"{relative_path} line {line_number}",
                tracked_stats=tracked_stats,
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
                modes_by_id=modes_by_id,
            )
        )
    return record_count, blockers


def _validate_target_row(
    row: dict[str, Any],
    *,
    source: str,
    tracked_stats: tuple[str, ...],
    vocab_size: int | None,
    max_seq_len: int | None,
    modes_by_id: dict[int, dict[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    if not isinstance(row.get("example_id"), str) or not row["example_id"].strip():
        blockers.append(f"{source}: example_id must be a non-empty string")
    if not isinstance(row.get("position"), int) or row["position"] < 0:
        blockers.append(f"{source}: position must be a non-negative integer")
    input_ids = row.get("input_ids")
    if not isinstance(input_ids, list) or not input_ids:
        blockers.append(f"{source}: input_ids must be a non-empty list")
    else:
        for offset, token_id in enumerate(input_ids):
            if not isinstance(token_id, int):
                blockers.append(f"{source}: input_ids[{offset}] must be an integer")
            elif vocab_size is not None and not 0 <= token_id < vocab_size:
                blockers.append(
                    f"{source}: token id {token_id} outside vocabulary "
                    f"range [0, {vocab_size})"
                )
        if max_seq_len is not None and len(input_ids) > max_seq_len:
            blockers.append(
                f"{source}: len(input_ids)={len(input_ids)} exceeds "
                f"sequence.max_seq_len={max_seq_len}"
            )
    mode_id = row.get("mode_id")
    mode: dict[str, Any] | None = None
    if not isinstance(mode_id, int):
        blockers.append(f"{source}: mode_id must be an integer")
    elif mode_id not in modes_by_id:
        blockers.append(f"unknown mode_id={mode_id} in {source}")
    else:
        mode = modes_by_id[mode_id]
    bounds = row.get("bounds")
    if not isinstance(bounds, dict):
        blockers.append(f"{source}: bounds must be an object")
        return blockers
    mode_bounds = mode.get("bounds", {}) if mode else {}
    for stat in tracked_stats:
        bound = bounds.get(stat)
        stat_source = f"{source}: row bounds for {stat}"
        blockers.extend(_validate_bound(bound, stat_source, stat))
        if isinstance(bound, dict) and isinstance(mode_bounds, dict):
            blockers.extend(
                _validate_row_bound_inside_mode(
                    bound, mode_bounds.get(stat), stat_source
                )
            )
    return blockers


def _validate_packed_target_payload(
    root: Path,
    payload: dict[str, Any] | None,
    *,
    vocab_size: int | None,
    max_seq_len: int | None,
    modes_by_id: dict[int, dict[str, Any]],
) -> tuple[int, list[str]]:
    blockers: list[str] = []
    if not isinstance(payload, dict):
        return 0, ["target_payload must be an object for packed_corridor_v1"]
    expected_records = _non_negative_int(payload.get("num_records"))
    if expected_records is None:
        blockers.append("target_payload.num_records must be a non-negative integer")
        expected_records = 0
    expected_examples = _non_negative_int(payload.get("num_examples"))
    if expected_examples is None:
        blockers.append("target_payload.num_examples must be a non-negative integer")
        expected_examples = 0
    payload_max_seq_len = _positive_int(payload.get("max_seq_len"))
    if payload_max_seq_len is None:
        blockers.append("target_payload.max_seq_len must be a positive integer")
    elif max_seq_len is not None and payload_max_seq_len != max_seq_len:
        blockers.append(
            "target_payload.max_seq_len mismatch: "
            f"expected {max_seq_len}, got {payload_max_seq_len}"
        )
    if payload.get("mode_table_path") != "modes.json":
        blockers.append("target_payload.mode_table_path must be 'modes.json'")
    if payload.get("ordering") != "capture_position_order_v1":
        blockers.append("target_payload.ordering is unsupported")
    if (
        payload.get("example_index_contract")
        != "zero_based_row_index_into_examples_input_ids"
    ):
        blockers.append("target_payload.example_index_contract is unsupported")

    arrays_payload = payload.get("arrays")
    if not isinstance(arrays_payload, dict):
        blockers.append("target_payload.arrays must be an object")
        arrays_payload = {}
    arrays: dict[str, np.ndarray] = {}
    for name, rank in PACKED_TARGET_ARRAYS.items():
        array_payload = arrays_payload.get(name)
        array, array_blockers = _load_packed_array(
            root,
            name=name,
            rank=rank,
            payload=array_payload,
        )
        blockers.extend(array_blockers)
        if array is not None:
            arrays[name] = array

    examples = arrays.get("examples_input_ids")
    if examples is not None:
        if examples.shape[0] != expected_examples:
            blockers.append(
                "examples_input_ids example count mismatch: "
                f"expected {expected_examples}, got {examples.shape[0]}"
            )
        if max_seq_len is not None and examples.shape[1] != max_seq_len:
            blockers.append(
                "examples_input_ids max_seq_len mismatch: "
                f"expected {max_seq_len}, got {examples.shape[1]}"
            )
        if vocab_size is not None and examples.size:
            if bool(np.any((examples < 0) | (examples >= vocab_size))):
                blockers.append("examples_input_ids contains token ids outside vocab")

    for name in ("position_example_index", "position", "mode_id", "weight"):
        array = arrays.get(name)
        if array is not None and array.shape[0] != expected_records:
            blockers.append(
                f"{name} record count mismatch: expected {expected_records}, "
                f"got {array.shape[0]}"
            )

    example_index = arrays.get("position_example_index")
    if example_index is not None and example_index.size:
        if bool(np.any((example_index < 0) | (example_index >= expected_examples))):
            blockers.append("position_example_index contains invalid example indexes")
    position = arrays.get("position")
    if position is not None and position.size and max_seq_len is not None:
        if bool(np.any((position < 0) | (position >= max_seq_len))):
            blockers.append("position contains values outside sequence range")
    mode_id = arrays.get("mode_id")
    if mode_id is not None and mode_id.size:
        valid_modes = np.asarray(sorted(modes_by_id), dtype=mode_id.dtype)
        if valid_modes.size == 0 or bool(~np.all(np.isin(mode_id, valid_modes))):
            blockers.append("mode_id contains unknown mode ids")
    weight = arrays.get("weight")
    if weight is not None and weight.size:
        if bool(np.any(~np.isfinite(weight))):
            blockers.append("weight contains non-finite values")
        if bool(np.any(weight < 0.0)):
            blockers.append("weight contains negative values")

    metadata = payload.get("examples_metadata")
    blockers.extend(
        _validate_examples_metadata(
            root,
            metadata,
            expected_examples=expected_examples,
        )
    )
    return int(expected_records), blockers


def _load_packed_array(
    root: Path,
    *,
    name: str,
    rank: int,
    payload: Any,
) -> tuple[np.ndarray | None, list[str]]:
    blockers: list[str] = []
    if not isinstance(payload, dict):
        return None, [f"target_payload.arrays.{name} must be an object"]
    path_value = payload.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        return None, [f"target_payload.arrays.{name}.path must be non-empty"]
    array_path = root / path_value
    if not array_path.is_file():
        return None, [f"packed target array does not exist: {path_value}"]
    try:
        array = np.load(array_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        return None, [f"packed target array invalid: {path_value}: {exc}"]
    dtype = payload.get("dtype")
    if dtype != str(array.dtype):
        blockers.append(
            f"target_payload.arrays.{name}.dtype mismatch: "
            f"expected {dtype!r}, got {str(array.dtype)!r}"
        )
    shape = payload.get("shape")
    if shape != list(array.shape):
        blockers.append(
            f"target_payload.arrays.{name}.shape mismatch: "
            f"expected {shape!r}, got {list(array.shape)!r}"
        )
    if array.ndim != rank:
        blockers.append(f"target_payload.arrays.{name} must have rank {rank}")
    return array, blockers


def _validate_examples_metadata(
    root: Path,
    metadata: Any,
    *,
    expected_examples: int,
) -> list[str]:
    blockers: list[str] = []
    if not isinstance(metadata, dict):
        return ["target_payload.examples_metadata must be an object"]
    path_value = metadata.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        return ["target_payload.examples_metadata.path must be non-empty"]
    expected_records = _non_negative_int(metadata.get("num_records"))
    if expected_records is None:
        blockers.append("target_payload.examples_metadata.num_records invalid")
        expected_records = expected_examples
    if expected_records != expected_examples:
        blockers.append(
            "target_payload.examples_metadata.num_records mismatch: "
            f"expected {expected_examples}, got {expected_records}"
        )
    path = root / path_value
    if not path.is_file():
        return blockers + [f"examples metadata does not exist: {path_value}"]
    actual = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            blockers.append(
                f"malformed examples metadata row line {line_number + 1}: {exc.msg}"
            )
            continue
        if not isinstance(row, dict):
            blockers.append(f"examples metadata line {line_number + 1} must be object")
            continue
        if row.get("example_index") != actual:
            blockers.append(
                f"examples metadata line {line_number + 1} example_index mismatch"
            )
        if not isinstance(row.get("example_id"), str) or not row["example_id"].strip():
            blockers.append(
                f"examples metadata line {line_number + 1} example_id invalid"
            )
        actual += 1
    if actual != expected_examples:
        blockers.append(
            "examples metadata record count mismatch: "
            f"expected {expected_examples}, got {actual}"
        )
    return blockers


def _validate_exemplar_reservoir(
    root: Path,
    reservoir: dict[str, Any] | None,
    *,
    vocab_size: int | None,
    max_seq_len: int | None,
    modes_by_id: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    metadata: dict[str, Any] = {
        "exemplar_reservoir_enabled": False,
        "exemplar_payload_type": None,
        "exemplar_records": 0,
        "exemplar_shards": 0,
    }
    blockers: list[str] = []
    if reservoir is None:
        return metadata, blockers
    if not isinstance(reservoir, dict) or not reservoir:
        return metadata, ["exemplar_reservoir must be an object when present"]

    enabled = reservoir.get("enabled")
    if not isinstance(enabled, bool):
        blockers.append("exemplar_reservoir.enabled must be a boolean")
    metadata["exemplar_reservoir_enabled"] = bool(enabled)

    payload_type = reservoir.get("payload_type")
    if payload_type not in {"dense_probs", "cascaded_soft_labels_v1"}:
        blockers.append("exemplar_reservoir.payload_type is unsupported")
    metadata["exemplar_payload_type"] = (
        payload_type if isinstance(payload_type, str) else None
    )

    loss = reservoir.get("loss")
    if loss != "kl":
        blockers.append("exemplar_reservoir.loss must be 'kl' for P137")
    encoding_contract = reservoir.get("encoding_contract")
    if payload_type == "cascaded_soft_labels_v1":
        blockers.extend(
            _validate_cascaded_contract(
                encoding_contract, vocab_size=vocab_size
            )
        )

    expected_total = _non_negative_int(reservoir.get("num_records"))
    if expected_total is None:
        blockers.append("exemplar_reservoir.num_records must be a non-negative integer")

    shards = reservoir.get("shards")
    if not isinstance(shards, list) or not shards:
        blockers.append("exemplar_reservoir.shards must be a non-empty list")
        return metadata, blockers
    metadata["exemplar_shards"] = len(shards)

    total_records = 0
    for shard_index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            blockers.append(
                f"exemplar_reservoir.shards[{shard_index}] must be an object"
            )
            continue
        shard_path_value = shard.get("path")
        if not isinstance(shard_path_value, str) or not shard_path_value.strip():
            blockers.append(
                f"exemplar_reservoir.shards[{shard_index}] missing non-empty path"
            )
            continue
        expected_records = _non_negative_int(shard.get("num_records"))
        if expected_records is None:
            blockers.append(
                f"exemplar_reservoir.shards[{shard_index}].num_records must be a "
                "non-negative integer"
            )
        shard_path = root / shard_path_value
        if not shard_path.is_file():
            blockers.append(f"exemplar shard does not exist: {shard_path_value}")
            continue
        actual_records, shard_blockers = _validate_exemplar_shard(
            shard_path,
            root=root,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            modes_by_id=modes_by_id,
            payload_type=payload_type,
            encoding_contract=encoding_contract,
        )
        total_records += actual_records
        blockers.extend(shard_blockers)
        if expected_records is not None and actual_records != expected_records:
            blockers.append(
                f"exemplar shard record count mismatch for {shard_path_value}: "
                f"expected {expected_records}, got {actual_records}"
            )

    metadata["exemplar_records"] = total_records
    if expected_total is not None and total_records != expected_total:
        blockers.append(
            "exemplar_reservoir.num_records mismatch: "
            f"expected {expected_total}, got {total_records}"
        )
    return metadata, blockers


def _validate_exemplar_shard(
    shard_path: Path,
    *,
    root: Path,
    vocab_size: int | None,
    max_seq_len: int | None,
    modes_by_id: dict[int, dict[str, Any]],
    payload_type: str | None,
    encoding_contract: Any,
) -> tuple[int, list[str]]:
    blockers: list[str] = []
    record_count = 0
    relative_path = _relative_path(shard_path, root)
    for line_number, line in enumerate(
        shard_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        record_count += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            blockers.append(
                f"malformed JSONL row in {relative_path} line {line_number}: {exc.msg}"
            )
            continue
        if not isinstance(row, dict):
            blockers.append(
                f"row in {relative_path} line {line_number} must be an object"
            )
            continue
        blockers.extend(
            _validate_exemplar_row(
                row,
                source=f"{relative_path} line {line_number}",
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
                modes_by_id=modes_by_id,
                payload_type=payload_type,
                encoding_contract=encoding_contract,
            )
        )
    return record_count, blockers


def _validate_exemplar_row(
    row: dict[str, Any],
    *,
    source: str,
    vocab_size: int | None,
    max_seq_len: int | None,
    modes_by_id: dict[int, dict[str, Any]],
    payload_type: str | None,
    encoding_contract: Any,
) -> list[str]:
    blockers: list[str] = []
    if not isinstance(row.get("example_id"), str) or not row["example_id"].strip():
        blockers.append(f"{source}: example_id must be a non-empty string")
    position = row.get("position")
    if not isinstance(position, int):
        blockers.append(f"{source}: position must be an integer")
    elif max_seq_len is not None and not 0 <= position < max_seq_len:
        blockers.append(
            f"{source}: position={position} outside sequence range [0, {max_seq_len})"
        )

    input_ids = row.get("input_ids")
    if not isinstance(input_ids, list) or not input_ids:
        blockers.append(f"{source}: input_ids must be a non-empty list")
    else:
        if max_seq_len is not None and len(input_ids) != max_seq_len:
            blockers.append(
                f"{source}: len(input_ids)={len(input_ids)} must equal "
                f"sequence.max_seq_len={max_seq_len}"
            )
        for offset, token_id in enumerate(input_ids):
            if not isinstance(token_id, int):
                blockers.append(f"{source}: input_ids[{offset}] must be an integer")
            elif vocab_size is not None and not 0 <= token_id < vocab_size:
                blockers.append(
                    f"{source}: token id {token_id} outside vocabulary "
                    f"range [0, {vocab_size})"
                )

    teacher_probs = row.get("teacher_probs")
    if payload_type == "dense_probs":
        if not isinstance(teacher_probs, list):
            blockers.append(f"{source}: teacher_probs must be a list")
            teacher_probs = []
        if vocab_size is not None and len(teacher_probs) != vocab_size:
            blockers.append(
                f"{source}: len(teacher_probs)={len(teacher_probs)} must equal "
                f"teacher.vocab_size={vocab_size}"
            )
        prob_sum = 0.0
        prob_values_ok = True
        for offset, value in enumerate(teacher_probs):
            probability = _finite_number(value)
            if probability is None:
                blockers.append(f"{source}: teacher_probs[{offset}] must be finite")
                prob_values_ok = False
                continue
            if probability < 0.0:
                blockers.append(
                    f"{source}: teacher_probs[{offset}] must be non-negative"
                )
                prob_values_ok = False
            prob_sum += probability
        if prob_values_ok and abs(prob_sum - 1.0) > 1e-5:
            blockers.append(
                f"{source}: teacher_probs must sum to 1.0 within 1e-5, got {prob_sum}"
            )
    elif payload_type == "cascaded_soft_labels_v1":
        if "teacher_probs" in row:
            blockers.append(
                f"{source}: teacher_probs is forbidden for cascaded_soft_labels_v1"
            )
        blockers.extend(
            _validate_cascaded_exemplar_row(
                row,
                source=source,
                vocab_size=vocab_size,
                contract=encoding_contract,
            )
        )

    weight = _finite_number(row.get("weight"))
    if weight is None:
        blockers.append(f"{source}: weight must be finite")
    elif weight < 0.0:
        blockers.append(f"{source}: weight must be non-negative")

    mode_id = row.get("mode_id")
    if mode_id is not None:
        if not isinstance(mode_id, int):
            blockers.append(f"{source}: mode_id must be an integer when present")
        elif mode_id not in modes_by_id:
            blockers.append(f"unknown mode_id={mode_id} in {source}")

    score = row.get("interestingness_score")
    if score is not None and _finite_number(score) is None:
        blockers.append(f"{source}: interestingness_score must be finite when present")

    reason_codes = row.get("reason_codes")
    if reason_codes is not None:
        if not isinstance(reason_codes, list):
            blockers.append(f"{source}: reason_codes must be a list when present")
        else:
            for offset, reason in enumerate(reason_codes):
                if not isinstance(reason, str):
                    blockers.append(
                        f"{source}: reason_codes[{offset}] must be a string"
                    )
    return blockers


def _validate_cascaded_contract(
    contract: Any, *, vocab_size: int | None
) -> list[str]:
    if not isinstance(contract, dict):
        return ["cascaded exemplar encoding_contract must be an object"]
    blockers = []
    if contract.get("kind") != "cascaded_soft_labels_v1":
        blockers.append("cascaded encoding_contract.kind mismatch")
    if contract.get("version") != 1:
        blockers.append("unsupported cascaded encoding version")
    top_k = contract.get("top_k")
    if not isinstance(top_k, int) or top_k <= 0:
        blockers.append("cascaded encoding_contract.top_k must be positive")
    elif vocab_size is not None and top_k > vocab_size and vocab_size >= 256:
        blockers.append("cascaded encoding_contract.top_k exceeds vocabulary")
    edges = contract.get("bucket_edges")
    if not isinstance(edges, list) or len(edges) < 2:
        blockers.append("cascaded encoding_contract.bucket_edges is invalid")
    else:
        numbers = [_finite_number(value) for value in edges]
        if any(value is None for value in numbers):
            blockers.append("cascaded bucket_edges must be finite")
        elif (
            numbers[0] != 1.0
            or numbers[-1] != 0.0
            or any(
                left <= right
                for left, right in zip(numbers[:-1], numbers[1:], strict=True)
            )
        ):
            blockers.append("cascaded bucket_edges must be strictly descending 1 to 0")
    return blockers


def _validate_cascaded_exemplar_row(
    row: dict[str, Any],
    *,
    source: str,
    vocab_size: int | None,
    contract: Any,
) -> list[str]:
    blockers = []
    if not isinstance(contract, dict):
        return [f"{source}: missing cascaded encoding contract"]
    if row.get("encoding_kind") != contract.get("kind"):
        blockers.append(f"{source}: encoding kind does not match manifest")
    if row.get("encoding_version") != contract.get("version"):
        blockers.append(f"{source}: encoding version does not match manifest")
    if row.get("bucket_edges") != contract.get("bucket_edges"):
        blockers.append(f"{source}: bucket edges do not match manifest")
    token_ids = row.get("top_token_ids")
    log_probs = row.get("top_log_probs")
    configured_k = contract.get("top_k")
    if not isinstance(token_ids, list) or not isinstance(log_probs, list):
        blockers.append(f"{source}: malformed top-K arrays")
        return blockers
    if len(token_ids) != len(log_probs) or not token_ids:
        blockers.append(f"{source}: malformed top-K arrays")
    if isinstance(configured_k, int) and len(token_ids) > configured_k:
        blockers.append(f"{source}: stored K exceeds configured K")
    if len(set(token_ids)) != len(token_ids):
        blockers.append(f"{source}: top token ids must be unique")
    for token_id in token_ids:
        if not isinstance(token_id, int) or (
            vocab_size is not None and not 0 <= token_id < vocab_size
        ):
            blockers.append(f"{source}: top token id outside vocabulary")
            break
    if any(_finite_number(value) is None for value in log_probs):
        blockers.append(f"{source}: top log probabilities must be finite")
    bucket_mass = row.get("bucket_mass")
    bucket_count = row.get("bucket_count")
    bucket_mean = row.get("bucket_mean_logp")
    edges = contract.get("bucket_edges", [])
    expected_buckets = max(0, len(edges) - 1)
    for name, values in (
        ("bucket_mass", bucket_mass),
        ("bucket_count", bucket_count),
        ("bucket_mean_logp", bucket_mean),
    ):
        if not isinstance(values, list) or len(values) != expected_buckets:
            blockers.append(f"{source}: {name} shape mismatch")
    if isinstance(bucket_mass, list):
        masses = [_finite_number(value) for value in bucket_mass]
        if any(value is None or value < 0 for value in masses):
            blockers.append(f"{source}: bucket masses must be finite and nonnegative")
        top_mass = _finite_number(row.get("top_mass"))
        tail_mass = _finite_number(row.get("tail_mass"))
        if top_mass is None or tail_mass is None:
            blockers.append(f"{source}: top_mass and tail_mass must be finite")
        elif (
            abs(
                top_mass
                + sum(value for value in masses if value is not None)
                - 1.0
            )
            > 2e-3
        ):
            blockers.append(f"{source}: compressed probability mass is inconsistent")
        elif (
            abs(sum(value for value in masses if value is not None) - tail_mass)
            > 2e-3
        ):
            blockers.append(f"{source}: bucket mass does not match tail mass")
    return blockers


def _validate_bound(value: Any, source: str, stat: str) -> list[str]:
    blockers: list[str] = []
    if not isinstance(value, dict):
        return [f"{source} must be an object with min and max"]
    lower = _number(value.get("min"))
    upper = _number(value.get("max"))
    mean = _number(value.get("mean")) if "mean" in value else None
    if lower is None:
        blockers.append(f"{source}.min must be numeric")
    if upper is None:
        blockers.append(f"{source}.max must be numeric")
    if lower is not None and upper is not None and lower > upper:
        blockers.append(f"{source} invalid: min={lower} max={upper}")
    if (
        lower is not None
        and upper is not None
        and mean is not None
        and not lower <= mean <= upper
    ):
        blockers.append(
            f"{source}.mean outside bounds: min={lower} mean={mean} max={upper}"
        )
    if "mean" in value and mean is None:
        blockers.append(f"{source}.mean must be numeric when present")
    if stat in PROBABILITY_LIKE_STATS:
        for key, number in (("min", lower), ("max", upper), ("mean", mean)):
            if number is not None and not 0.0 <= number <= 1.0:
                blockers.append(f"{source}.{key} must be within [0.0, 1.0]")
    if stat == "entropy":
        for key, number in (("min", lower), ("max", upper), ("mean", mean)):
            if number is not None and number < 0.0:
                blockers.append(f"{source}.{key} must be non-negative")
    return blockers


def _validate_row_bound_inside_mode(
    row_bound: dict[str, Any],
    mode_bound: Any,
    source: str,
) -> list[str]:
    if not isinstance(mode_bound, dict):
        return []
    mode_min = _number(mode_bound.get("min"))
    mode_max = _number(mode_bound.get("max"))
    row_min = _number(row_bound.get("min"))
    row_max = _number(row_bound.get("max"))
    if None in {mode_min, mode_max, row_min, row_max}:
        return []
    if not mode_min <= row_min <= row_max <= mode_max:
        return [
            f"{source} outside mode bounds: mode_min={mode_min} "
            f"row_min={row_min} row_max={row_max} mode_max={mode_max}"
        ]
    return []


def _tracked_stats(stats: dict[str, Any]) -> tuple[str, ...]:
    tracked = stats.get("tracked")
    if not isinstance(tracked, list):
        return ()
    return tuple(str(item) for item in tracked if str(item).strip())


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _positive_int(value: Any) -> int | None:
    if not isinstance(value, int) or value <= 0:
        return None
    return value


def _non_negative_int(value: Any) -> int | None:
    if not isinstance(value, int) or value < 0:
        return None
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _finite_number(value: Any) -> float | None:
    number = _number(value)
    if number is None or not math.isfinite(number):
        return None
    return number


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _result(
    *,
    blockers: list[str],
    warnings: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ValidationResult:
    return ValidationResult(
        ok=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings or ()),
        metadata=metadata or {},
    )
