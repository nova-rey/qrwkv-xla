from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qrwkv_xla.artifacts._json import read_json_object
from qrwkv_xla.artifacts.fingerprint import (
    FingerprintManifest,
    validate_fingerprint_artifact,
)

REQUIRED_FINGERPRINT_STATS = (
    "entropy",
    "top1_margin",
    "top8_mass",
    "top32_mass",
    "tail_mass",
)
TARGET_PAYLOAD_LEGACY_JSONL = "legacy_jsonl"
TARGET_PAYLOAD_PACKED_CORRIDOR_V1 = "packed_corridor_v1"


@dataclass(frozen=True)
class FingerprintTargetRecord:
    example_id: str
    position: int
    input_ids: tuple[int, ...]
    mode_id: int
    entropy_min: float
    entropy_max: float
    top1_margin_min: float
    top1_margin_max: float
    top8_mass_min: float
    top8_mass_max: float
    top32_mass_min: float
    top32_mass_max: float
    tail_mass_min: float
    tail_mass_max: float
    weight: float = 1.0


@dataclass(frozen=True)
class FingerprintBatch:
    input_ids: np.ndarray
    position: np.ndarray
    mode_id: np.ndarray
    entropy_min: np.ndarray
    entropy_max: np.ndarray
    top1_margin_min: np.ndarray
    top1_margin_max: np.ndarray
    top8_mass_min: np.ndarray
    top8_mass_max: np.ndarray
    top32_mass_min: np.ndarray
    top32_mass_max: np.ndarray
    tail_mass_min: np.ndarray
    tail_mass_max: np.ndarray
    weight: np.ndarray


@dataclass(frozen=True)
class FingerprintLoaderConfig:
    artifact_dir: Path
    batch_size: int
    shuffle: bool = False
    seed: int = 0
    drop_remainder: bool = False
    max_records: int | None = None
    validate: bool = True


class FingerprintTargetDataset:
    def __init__(self, config: FingerprintLoaderConfig):
        self.config = config
        if config.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {config.batch_size}")
        if config.max_records is not None and config.max_records < 0:
            raise ValueError(f"max_records must be >= 0, got {config.max_records}")

        self.artifact_dir = Path(config.artifact_dir)
        if config.validate:
            result = validate_fingerprint_artifact(self.artifact_dir)
            if not result.ok:
                joined = "; ".join(result.blockers)
                raise ValueError(f"fingerprint artifact validation failed: {joined}")

        manifest_payload = read_json_object(self.artifact_dir / "manifest.json")
        self.manifest = FingerprintManifest.from_payload(manifest_payload)
        self._vocab_size = _positive_int(
            self.manifest.teacher.get("vocab_size"),
            "teacher.vocab_size",
        )
        self._max_seq_len = _positive_int(
            self.manifest.sequence.get("max_seq_len"),
            "sequence.max_seq_len",
        )
        self._tracked_stats = _tracked_stats(self.manifest)
        missing_stats = tuple(
            stat
            for stat in REQUIRED_FINGERPRINT_STATS
            if stat not in self._tracked_stats
        )
        if missing_stats:
            raise ValueError(
                "fingerprint loader requires tracked stats: "
                + ", ".join(REQUIRED_FINGERPRINT_STATS)
            )

        self._packed: _PackedTargets | None = None
        self._records: tuple[FingerprintTargetRecord, ...] | None = None
        if _target_payload_kind(self.manifest) == TARGET_PAYLOAD_PACKED_CORRIDOR_V1:
            self._packed = _PackedTargets.load(self.artifact_dir, self.manifest)
            record_count = self._packed.num_records
        else:
            records = list(self._load_jsonl_records())
            self._records = tuple(records)
            record_count = len(records)

        order = np.arange(record_count, dtype=np.int64)
        if config.max_records is not None:
            order = order[: config.max_records]
        if config.shuffle:
            rng = np.random.default_rng(config.seed)
            order = rng.permutation(order)
        self._order = order

    @property
    def num_records(self) -> int:
        return int(self._order.size)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    @property
    def tracked_stats(self) -> tuple[str, ...]:
        return self._tracked_stats

    def iter_records(self) -> Iterator[FingerprintTargetRecord]:
        if self._packed is not None:
            for index in self._order:
                yield self._packed.record_at(int(index))
            return
        assert self._records is not None
        for index in self._order:
            yield self._records[int(index)]

    def iter_batches(self) -> Iterator[FingerprintBatch]:
        batch_size = self.config.batch_size
        for start in range(0, self.num_records, batch_size):
            indexes = self._order[start : start + batch_size]
            if self.config.drop_remainder and indexes.size < batch_size:
                continue
            if self._packed is not None:
                yield self._packed.batch_at(indexes)
            else:
                assert self._records is not None
                records = tuple(self._records[int(index)] for index in indexes)
                yield _records_to_batch(records, max_seq_len=self.max_seq_len)

    def _load_jsonl_records(self) -> Iterator[FingerprintTargetRecord]:
        for shard in self.manifest.target_shards:
            shard_path = self.artifact_dir / str(shard["path"])
            for line in shard_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                yield _row_to_record(
                    json.loads(line),
                    max_seq_len=self.max_seq_len,
                )


class _PackedTargets:
    def __init__(
        self,
        *,
        example_ids: tuple[str, ...],
        examples_input_ids: np.ndarray,
        position_example_index: np.ndarray,
        position: np.ndarray,
        mode_id: np.ndarray,
        weight: np.ndarray,
        mode_bounds: dict[int, dict[str, dict[str, float]]],
    ) -> None:
        self.example_ids = example_ids
        self.examples_input_ids = examples_input_ids
        self.position_example_index = position_example_index
        self.position = position
        self.mode_id = mode_id
        self.weight = weight
        self.mode_bounds = mode_bounds

    @classmethod
    def load(cls, artifact_dir: Path, manifest: FingerprintManifest) -> _PackedTargets:
        payload = manifest.target_payload
        if not isinstance(payload, dict):
            raise ValueError("packed fingerprint target payload missing from manifest")
        arrays = payload.get("arrays")
        if not isinstance(arrays, dict):
            raise ValueError("packed fingerprint target arrays missing from manifest")
        return cls(
            example_ids=_load_example_ids(artifact_dir, payload),
            examples_input_ids=_load_manifest_array(
                artifact_dir, arrays, "examples_input_ids"
            ),
            position_example_index=_load_manifest_array(
                artifact_dir, arrays, "position_example_index"
            ),
            position=_load_manifest_array(artifact_dir, arrays, "position"),
            mode_id=_load_manifest_array(artifact_dir, arrays, "mode_id"),
            weight=_load_manifest_array(artifact_dir, arrays, "weight"),
            mode_bounds=_load_mode_bounds(artifact_dir / manifest.modes_file),
        )

    @property
    def num_records(self) -> int:
        return int(self.position.shape[0])

    def record_at(self, index: int) -> FingerprintTargetRecord:
        example_index = int(self.position_example_index[index])
        mode_id = int(self.mode_id[index])
        bounds = self.mode_bounds[mode_id]
        return FingerprintTargetRecord(
            example_id=self.example_ids[example_index],
            position=int(self.position[index]),
            input_ids=tuple(
                int(token) for token in self.examples_input_ids[example_index]
            ),
            mode_id=mode_id,
            entropy_min=float(bounds["entropy"]["min"]),
            entropy_max=float(bounds["entropy"]["max"]),
            top1_margin_min=float(bounds["top1_margin"]["min"]),
            top1_margin_max=float(bounds["top1_margin"]["max"]),
            top8_mass_min=float(bounds["top8_mass"]["min"]),
            top8_mass_max=float(bounds["top8_mass"]["max"]),
            top32_mass_min=float(bounds["top32_mass"]["min"]),
            top32_mass_max=float(bounds["top32_mass"]["max"]),
            tail_mass_min=float(bounds["tail_mass"]["min"]),
            tail_mass_max=float(bounds["tail_mass"]["max"]),
            weight=float(self.weight[index]),
        )

    def batch_at(self, indexes: np.ndarray) -> FingerprintBatch:
        example_indexes = self.position_example_index[indexes]
        bounds = [self.mode_bounds[int(mode_id)] for mode_id in self.mode_id[indexes]]
        return FingerprintBatch(
            input_ids=np.asarray(
                self.examples_input_ids[example_indexes],
                dtype=np.int32,
            ),
            position=np.asarray(self.position[indexes], dtype=np.int32),
            mode_id=np.asarray(self.mode_id[indexes], dtype=np.int32),
            entropy_min=_float_array(bound["entropy"]["min"] for bound in bounds),
            entropy_max=_float_array(bound["entropy"]["max"] for bound in bounds),
            top1_margin_min=_float_array(
                bound["top1_margin"]["min"] for bound in bounds
            ),
            top1_margin_max=_float_array(
                bound["top1_margin"]["max"] for bound in bounds
            ),
            top8_mass_min=_float_array(bound["top8_mass"]["min"] for bound in bounds),
            top8_mass_max=_float_array(bound["top8_mass"]["max"] for bound in bounds),
            top32_mass_min=_float_array(
                bound["top32_mass"]["min"] for bound in bounds
            ),
            top32_mass_max=_float_array(
                bound["top32_mass"]["max"] for bound in bounds
            ),
            tail_mass_min=_float_array(bound["tail_mass"]["min"] for bound in bounds),
            tail_mass_max=_float_array(bound["tail_mass"]["max"] for bound in bounds),
            weight=np.asarray(self.weight[indexes], dtype=np.float32),
        )


def load_fingerprint_targets(
    path: str | Path,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    seed: int = 0,
    drop_remainder: bool = False,
    max_records: int | None = None,
    validate: bool = True,
) -> FingerprintTargetDataset:
    return FingerprintTargetDataset(
        FingerprintLoaderConfig(
            artifact_dir=Path(path),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            drop_remainder=drop_remainder,
            max_records=max_records,
            validate=validate,
        )
    )


def _row_to_record(
    row: dict,
    *,
    max_seq_len: int,
) -> FingerprintTargetRecord:
    input_ids = tuple(int(token_id) for token_id in row["input_ids"])
    if len(input_ids) != max_seq_len:
        raise ValueError(
            "P133 requires fixed-length input_ids matching sequence.max_seq_len: "
            f"len(input_ids)={len(input_ids)} max_seq_len={max_seq_len} "
            f"example_id={row.get('example_id')!r}"
        )
    bounds = row["bounds"]
    return FingerprintTargetRecord(
        example_id=str(row["example_id"]),
        position=int(row["position"]),
        input_ids=input_ids,
        mode_id=int(row["mode_id"]),
        entropy_min=float(bounds["entropy"]["min"]),
        entropy_max=float(bounds["entropy"]["max"]),
        top1_margin_min=float(bounds["top1_margin"]["min"]),
        top1_margin_max=float(bounds["top1_margin"]["max"]),
        top8_mass_min=float(bounds["top8_mass"]["min"]),
        top8_mass_max=float(bounds["top8_mass"]["max"]),
        top32_mass_min=float(bounds["top32_mass"]["min"]),
        top32_mass_max=float(bounds["top32_mass"]["max"]),
        tail_mass_min=float(bounds["tail_mass"]["min"]),
        tail_mass_max=float(bounds["tail_mass"]["max"]),
        weight=float(row.get("weight", 1.0)),
    )


def _records_to_batch(
    records: tuple[FingerprintTargetRecord, ...],
    *,
    max_seq_len: int,
) -> FingerprintBatch:
    return FingerprintBatch(
        input_ids=np.asarray(
            [record.input_ids for record in records],
            dtype=np.int32,
        ).reshape((len(records), max_seq_len)),
        position=np.asarray([record.position for record in records], dtype=np.int32),
        mode_id=np.asarray([record.mode_id for record in records], dtype=np.int32),
        entropy_min=_float_array(record.entropy_min for record in records),
        entropy_max=_float_array(record.entropy_max for record in records),
        top1_margin_min=_float_array(record.top1_margin_min for record in records),
        top1_margin_max=_float_array(record.top1_margin_max for record in records),
        top8_mass_min=_float_array(record.top8_mass_min for record in records),
        top8_mass_max=_float_array(record.top8_mass_max for record in records),
        top32_mass_min=_float_array(record.top32_mass_min for record in records),
        top32_mass_max=_float_array(record.top32_mass_max for record in records),
        tail_mass_min=_float_array(record.tail_mass_min for record in records),
        tail_mass_max=_float_array(record.tail_mass_max for record in records),
        weight=_float_array(record.weight for record in records),
    )


def _float_array(values) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float32)


def _target_payload_kind(manifest: FingerprintManifest) -> str:
    if manifest.target_payload is None:
        return TARGET_PAYLOAD_LEGACY_JSONL
    kind = manifest.target_payload.get("kind")
    if kind == TARGET_PAYLOAD_PACKED_CORRIDOR_V1:
        return TARGET_PAYLOAD_PACKED_CORRIDOR_V1
    return TARGET_PAYLOAD_LEGACY_JSONL


def _load_manifest_array(
    artifact_dir: Path,
    arrays: dict,
    name: str,
) -> np.ndarray:
    payload = arrays.get(name)
    if not isinstance(payload, dict) or not isinstance(payload.get("path"), str):
        raise ValueError(f"packed target array {name!r} missing path")
    return np.load(artifact_dir / payload["path"], mmap_mode="r", allow_pickle=False)


def _load_example_ids(artifact_dir: Path, payload: dict) -> tuple[str, ...]:
    metadata = payload.get("examples_metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("path"), str):
        raise ValueError("packed target examples_metadata missing path")
    rows: list[str] = []
    for expected_index, line in enumerate(
        (artifact_dir / metadata["path"]).read_text(encoding="utf-8").splitlines()
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("example_index") != expected_index:
            raise ValueError("packed target examples_metadata is not ordered")
        rows.append(str(row["example_id"]))
    return tuple(rows)


def _load_mode_bounds(path: Path) -> dict[int, dict[str, dict[str, float]]]:
    payload = read_json_object(path)
    modes = payload.get("modes", ())
    return {int(mode["mode_id"]): mode["bounds"] for mode in modes}


def _tracked_stats(manifest: FingerprintManifest) -> tuple[str, ...]:
    tracked = manifest.stats.get("tracked", ())
    return tuple(str(stat) for stat in tracked)


def _positive_int(value, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
