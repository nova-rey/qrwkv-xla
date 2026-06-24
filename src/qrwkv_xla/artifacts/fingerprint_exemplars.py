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


@dataclass(frozen=True)
class FingerprintExemplarRecord:
    example_id: str
    input_ids: tuple[int, ...]
    position: int
    teacher_probs: tuple[float, ...] | None
    weight: float
    target_type: str = "dense_probs"
    top_token_ids: tuple[int, ...] | None = None
    top_log_probs: tuple[float, ...] | None = None
    top_mass: float | None = None
    tail_mass: float | None = None
    teacher_entropy: float | None = None
    bucket_edges: tuple[float, ...] | None = None
    bucket_mass: tuple[float, ...] | None = None
    bucket_count: tuple[int, ...] | None = None
    bucket_mean_logp: tuple[float, ...] | None = None
    mode_id: int | None = None
    interestingness_score: float | None = None
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class FingerprintExemplarBatch:
    input_ids: np.ndarray
    position: np.ndarray
    teacher_probs: np.ndarray | None
    weight: np.ndarray
    mode_id: np.ndarray
    interestingness_score: np.ndarray
    reason_codes: tuple[tuple[str, ...], ...]
    example_id: tuple[str, ...]
    target_type: str = "dense_probs"
    top_token_ids: np.ndarray | None = None
    top_log_probs: np.ndarray | None = None
    top_mass: np.ndarray | None = None
    tail_mass: np.ndarray | None = None
    teacher_entropy: np.ndarray | None = None
    bucket_edges: np.ndarray | None = None
    bucket_mass: np.ndarray | None = None
    bucket_count: np.ndarray | None = None
    bucket_mean_logp: np.ndarray | None = None


@dataclass(frozen=True)
class FingerprintExemplarLoaderConfig:
    artifact_dir: Path
    batch_size: int
    shuffle: bool = False
    seed: int = 0
    drop_remainder: bool = False
    max_records: int | None = None
    validate: bool = True
    require_exemplars: bool = True


class FingerprintExemplarDataset:
    def __init__(self, config: FingerprintExemplarLoaderConfig):
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

        reservoir = self.manifest.exemplar_reservoir
        if not reservoir:
            if config.require_exemplars:
                raise ValueError("fingerprint artifact has no exemplar_reservoir")
            self._records: tuple[FingerprintExemplarRecord, ...] = ()
            return
        self._target_type = str(reservoir.get("payload_type"))
        if self._target_type not in {"dense_probs", "cascaded_soft_labels_v1"}:
            raise ValueError("unsupported fingerprint exemplar payload type")
        self._encoding_contract = reservoir.get("encoding_contract", {})

        records = list(self._load_records(reservoir))
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

    def iter_records(self) -> Iterator[FingerprintExemplarRecord]:
        yield from self._records

    def iter_batches(self) -> Iterator[FingerprintExemplarBatch]:
        batch_size = self.config.batch_size
        for start in range(0, len(self._records), batch_size):
            records = self._records[start : start + batch_size]
            if self.config.drop_remainder and len(records) < batch_size:
                continue
            yield _records_to_batch(
                records,
                max_seq_len=self.max_seq_len,
                vocab_size=self.vocab_size,
            )

    def _load_records(
        self,
        reservoir: dict,
    ) -> Iterator[FingerprintExemplarRecord]:
        for shard in reservoir["shards"]:
            shard_path = self.artifact_dir / str(shard["path"])
            for line in shard_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                yield _row_to_record(
                    json.loads(line),
                    max_seq_len=self.max_seq_len,
                    vocab_size=self.vocab_size,
                    target_type=self._target_type,
                )


def load_fingerprint_exemplars(
    path: str | Path,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    seed: int = 0,
    drop_remainder: bool = False,
    max_records: int | None = None,
    validate: bool = True,
    require_exemplars: bool = True,
) -> FingerprintExemplarDataset:
    return FingerprintExemplarDataset(
        FingerprintExemplarLoaderConfig(
            artifact_dir=Path(path),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            drop_remainder=drop_remainder,
            max_records=max_records,
            validate=validate,
            require_exemplars=require_exemplars,
        )
    )


def _row_to_record(
    row: dict,
    *,
    max_seq_len: int,
    vocab_size: int,
    target_type: str,
) -> FingerprintExemplarRecord:
    input_ids = tuple(int(token_id) for token_id in row["input_ids"])
    if len(input_ids) != max_seq_len:
        raise ValueError(
            "P137 requires fixed-length exemplar input_ids matching "
            "sequence.max_seq_len: "
            f"len(input_ids)={len(input_ids)} max_seq_len={max_seq_len} "
            f"example_id={row.get('example_id')!r}"
        )
    teacher_probs = None
    compressed: dict[str, object] = {}
    if target_type == "dense_probs":
        teacher_probs = tuple(
            float(probability) for probability in row["teacher_probs"]
        )
        if len(teacher_probs) != vocab_size:
            raise ValueError(
                "dense_probs length must match teacher.vocab_size: "
                f"len(teacher_probs)={len(teacher_probs)} vocab_size={vocab_size}"
            )
    else:
        compressed = {
            "top_token_ids": tuple(int(value) for value in row["top_token_ids"]),
            "top_log_probs": tuple(float(value) for value in row["top_log_probs"]),
            "top_mass": float(row["top_mass"]),
            "tail_mass": float(row["tail_mass"]),
            "teacher_entropy": float(row["teacher_entropy"]),
            "bucket_edges": tuple(float(value) for value in row["bucket_edges"]),
            "bucket_mass": tuple(float(value) for value in row["bucket_mass"]),
            "bucket_count": tuple(int(value) for value in row["bucket_count"]),
            "bucket_mean_logp": tuple(
                float(value) for value in row["bucket_mean_logp"]
            ),
        }
    return FingerprintExemplarRecord(
        example_id=str(row["example_id"]),
        input_ids=input_ids,
        position=int(row["position"]),
        teacher_probs=teacher_probs,
        weight=float(row["weight"]),
        target_type=target_type,
        **compressed,
        mode_id=int(row["mode_id"]) if row.get("mode_id") is not None else None,
        interestingness_score=(
            float(row["interestingness_score"])
            if row.get("interestingness_score") is not None
            else None
        ),
        reason_codes=tuple(str(reason) for reason in row.get("reason_codes", ())),
    )


def _records_to_batch(
    records: tuple[FingerprintExemplarRecord, ...],
    *,
    max_seq_len: int,
    vocab_size: int,
) -> FingerprintExemplarBatch:
    target_type = records[0].target_type
    if any(record.target_type != target_type for record in records):
        raise ValueError("fingerprint exemplar batch mixes target types")
    return FingerprintExemplarBatch(
        input_ids=np.asarray(
            [record.input_ids for record in records],
            dtype=np.int32,
        ).reshape((len(records), max_seq_len)),
        position=np.asarray([record.position for record in records], dtype=np.int32),
        teacher_probs=(
            np.asarray(
                [record.teacher_probs for record in records],
                dtype=np.float32,
            ).reshape((len(records), vocab_size))
            if target_type == "dense_probs"
            else None
        ),
        weight=np.asarray([record.weight for record in records], dtype=np.float32),
        mode_id=np.asarray(
            [-1 if record.mode_id is None else record.mode_id for record in records],
            dtype=np.int32,
        ),
        interestingness_score=np.asarray(
            [
                np.nan
                if record.interestingness_score is None
                else record.interestingness_score
                for record in records
            ],
            dtype=np.float32,
        ),
        reason_codes=tuple(record.reason_codes for record in records),
        example_id=tuple(record.example_id for record in records),
        target_type=target_type,
        top_token_ids=_optional_array(records, "top_token_ids", np.int32),
        top_log_probs=_optional_array(records, "top_log_probs", np.float32),
        top_mass=_optional_array(records, "top_mass", np.float32),
        tail_mass=_optional_array(records, "tail_mass", np.float32),
        teacher_entropy=_optional_array(records, "teacher_entropy", np.float32),
        bucket_edges=(
            np.asarray(records[0].bucket_edges, dtype=np.float32)
            if records[0].bucket_edges is not None
            else None
        ),
        bucket_mass=_optional_array(records, "bucket_mass", np.float32),
        bucket_count=_optional_array(records, "bucket_count", np.int32),
        bucket_mean_logp=_optional_array(records, "bucket_mean_logp", np.float32),
    )


def _optional_array(records, name: str, dtype) -> np.ndarray | None:
    values = [getattr(record, name) for record in records]
    if values[0] is None:
        return None
    return np.asarray(values, dtype=dtype)


def _positive_int(value, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
