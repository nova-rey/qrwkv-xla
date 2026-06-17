from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from qrwkv_xla.artifacts._json import read_json_object

BEHAVIORAL_FINGERPRINT_ARTIFACT_TYPE = "behavioral_fingerprint"
BEHAVIORAL_FINGERPRINT_VERSION = "0.1"
SUPPORTED_BEHAVIORAL_FINGERPRINT_VERSIONS = (BEHAVIORAL_FINGERPRINT_VERSION,)
PROBABILITY_LIKE_STATS = frozenset(
    {"top1_margin", "top8_mass", "top32_mass", "tail_mass"}
)


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

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> FingerprintManifest:
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
    blockers.extend(_validate_manifest(manifest))

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

    return _result(blockers=blockers, warnings=warnings, metadata=metadata)


def _validate_manifest(manifest: FingerprintManifest) -> list[str]:
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
    if not manifest.target_shards:
        blockers.append("manifest target_shards must be non-empty")
    return blockers


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
