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

        records = list(self._load_records())
        if config.max_records is not None:
            records = records[: config.max_records]
        if config.shuffle:
            rng = np.random.default_rng(config.seed)
            order = rng.permutation(len(records))
            records = [records[index] for index in order]
        self._records = tuple(records)

    @property
    def num_records(self) -> int:
        return len(self._records)

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
        yield from self._records

    def iter_batches(self) -> Iterator[FingerprintBatch]:
        batch_size = self.config.batch_size
        for start in range(0, len(self._records), batch_size):
            records = self._records[start : start + batch_size]
            if self.config.drop_remainder and len(records) < batch_size:
                continue
            yield _records_to_batch(records, max_seq_len=self.max_seq_len)

    def _load_records(self) -> Iterator[FingerprintTargetRecord]:
        for shard in self.manifest.target_shards:
            shard_path = self.artifact_dir / str(shard["path"])
            for line in shard_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                yield _row_to_record(
                    json.loads(line),
                    max_seq_len=self.max_seq_len,
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


def _tracked_stats(manifest: FingerprintManifest) -> tuple[str, ...]:
    tracked = manifest.stats.get("tracked", ())
    return tuple(str(stat) for stat in tracked)


def _positive_int(value, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
