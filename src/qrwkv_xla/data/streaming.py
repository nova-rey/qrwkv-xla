from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from qrwkv_xla.generation import TokenizerMetadata
from qrwkv_xla.lm.data import LMBatch
from qrwkv_xla.lm.tokenized_corpus import LoadedTokenizedCorpus, load_tokenized_corpus

STREAMING_DATASET_FORMAT = "qrwkv_xla.streaming_dataset"
STREAMING_DATASET_SCHEMA_VERSION = "0.1"
STREAMING_DATASET_CREATED_BY = "qrwkv_xla.data.streaming"
PHASE = "P44"
BOUNDARY_POLICY = "prepacked_sequences_no_cross_shard_stitching"


@dataclass(frozen=True)
class StreamingSourceInfo:
    kind: str
    path: str | None
    sha256: str | None = None
    tokenized_corpus_format: str | None = None
    tokenized_corpus_created_by: str | None = None


@dataclass(frozen=True)
class StreamingCorpusInfo:
    num_documents: int
    num_sequences: int
    num_tokens: int
    sequence_length: int
    shard_tokens: int
    padded_tokens: int
    boundary_policy: str = BOUNDARY_POLICY


@dataclass(frozen=True)
class StreamingTokenShardManifest:
    path: str
    num_sequences: int
    num_tokens: int
    dtype: str
    sha256: str
    first_sequence_index: int
    provenance: str


@dataclass(frozen=True)
class StreamingDatasetManifest:
    format: str
    schema_version: str
    phase: str
    created_at_utc: str
    tokenizer: TokenizerMetadata
    corpus: StreamingCorpusInfo
    source: StreamingSourceInfo
    shards: tuple[StreamingTokenShardManifest, ...]
    created_by: str = STREAMING_DATASET_CREATED_BY
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StreamingCursor:
    position: int = 0
    shuffle: bool = False
    seed: int = 0

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "position": int(self.position),
            "shuffle": bool(self.shuffle),
            "seed": int(self.seed),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> StreamingCursor:
        if payload is None:
            return cls()
        return cls(
            position=int(payload.get("position", 0)),
            shuffle=bool(payload.get("shuffle", False)),
            seed=int(payload.get("seed", 0)),
        )


@dataclass(frozen=True)
class StreamingBatch:
    input_ids: np.ndarray
    labels: np.ndarray
    attention_mask: np.ndarray
    label_mask: np.ndarray
    cursor: StreamingCursor

    def as_lm_batch(self) -> LMBatch:
        return LMBatch(
            input_ids=jnp.asarray(self.input_ids, dtype=jnp.int32),
            labels=jnp.asarray(self.labels, dtype=jnp.int32),
            attention_mask=jnp.asarray(self.attention_mask, dtype=jnp.int32),
            label_mask=jnp.asarray(self.label_mask, dtype=jnp.int32),
        )

    def as_trainer_batch(self) -> dict[str, jnp.ndarray]:
        batch = self.as_lm_batch()
        return {
            "input_ids": batch.input_ids,
            "labels": batch.labels,
            "attention_mask": batch.attention_mask,
            "label_mask": batch.label_mask,
        }


class StreamingDataset:
    def __init__(
        self,
        root: str | Path,
        *,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        if seed < 0:
            raise ValueError("seed must be >= 0")
        self.root = Path(root)
        self.manifest = read_streaming_dataset_manifest(self.root / "manifest.json")
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._order = list(range(self.manifest.corpus.num_sequences))
        if self.shuffle:
            random.Random(self.seed).shuffle(self._order)
        self._cached_shard_path: str | None = None
        self._cached_shard_arrays: dict[str, np.ndarray] | None = None

    @property
    def tokens_available(self) -> int:
        return self.manifest.corpus.num_tokens

    @property
    def num_shards(self) -> int:
        return len(self.manifest.shards)

    def iter_batches(
        self,
        *,
        batch_size: int,
        cursor: StreamingCursor | None = None,
        drop_last: bool = False,
        max_batches: int | None = None,
    ) -> Iterator[StreamingBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if max_batches is not None and max_batches <= 0:
            raise ValueError("max_batches must be > 0 when provided")
        active_cursor = cursor or StreamingCursor(
            position=0,
            shuffle=self.shuffle,
            seed=self.seed,
        )
        if active_cursor.position < 0 or active_cursor.position > len(self._order):
            raise ValueError("cursor position is outside the dataset")
        if active_cursor.shuffle != self.shuffle or active_cursor.seed != self.seed:
            raise ValueError("cursor shuffle/seed does not match dataset configuration")

        position = active_cursor.position
        batches_emitted = 0
        while position < len(self._order):
            if max_batches is not None and batches_emitted >= max_batches:
                break
            indices = self._order[position : position + batch_size]
            if len(indices) < batch_size and drop_last:
                break
            arrays = self._read_indices(indices)
            if len(indices) < batch_size:
                arrays = _pad_batch(
                    arrays,
                    batch_size=batch_size,
                    sequence_length=self.manifest.corpus.sequence_length,
                    pad_token_id=_pad_token_id(self.manifest.tokenizer),
                )
            position += len(indices)
            batches_emitted += 1
            yield StreamingBatch(
                input_ids=arrays["input_ids"],
                labels=arrays["labels"],
                attention_mask=arrays["attention_mask"],
                label_mask=arrays["label_mask"],
                cursor=StreamingCursor(
                    position=position,
                    shuffle=self.shuffle,
                    seed=self.seed,
                ),
            )

    def validate_masks(self) -> None:
        pad_token_id = _pad_token_id(self.manifest.tokenizer)
        for shard in self.manifest.shards:
            arrays = self._read_shard(shard)
            expected_attention = arrays["input_ids"] != pad_token_id
            expected_label = arrays["labels"] != pad_token_id
            if not np.array_equal(arrays["attention_mask"], expected_attention):
                raise ValueError(f"attention_mask mismatch in {shard.path}")
            if not np.array_equal(arrays["label_mask"], expected_label):
                raise ValueError(f"label_mask mismatch in {shard.path}")
            if (
                not np.isfinite(arrays["input_ids"]).all()
                or not np.isfinite(arrays["labels"]).all()
            ):
                raise ValueError(f"non-finite token ids in {shard.path}")

    def _read_indices(self, indices: Sequence[int]) -> dict[str, np.ndarray]:
        rows: list[dict[str, np.ndarray]] = []
        for global_index in indices:
            shard = self._shard_for_global_index(global_index)
            arrays = self._read_shard(shard)
            local_index = global_index - shard.first_sequence_index
            rows.append({name: arrays[name][local_index] for name in arrays})
        if not rows:
            return _empty_batch(self.manifest.corpus.sequence_length)
        return {
            name: np.ascontiguousarray([row[name] for row in rows], dtype=np.int32)
            for name in ("input_ids", "labels", "attention_mask", "label_mask")
        }

    def _read_shard(
        self,
        shard: StreamingTokenShardManifest,
    ) -> dict[str, np.ndarray]:
        if (
            self._cached_shard_path == shard.path
            and self._cached_shard_arrays is not None
        ):
            return self._cached_shard_arrays
        shard_path = self.root / shard.path
        if not shard_path.is_file():
            raise ValueError(f"streaming shard is missing: {shard_path}")
        with np.load(shard_path) as loaded:
            try:
                arrays = {
                    "input_ids": np.asarray(loaded["input_ids"], dtype=np.int32),
                    "labels": np.asarray(loaded["labels"], dtype=np.int32),
                    "attention_mask": np.asarray(
                        loaded["attention_mask"], dtype=np.int32
                    ),
                    "label_mask": np.asarray(loaded["label_mask"], dtype=np.int32),
                }
            except KeyError as exc:
                raise ValueError(
                    "streaming shard missing required array "
                    f"{exc.args[0]!r}: {shard_path}"
                ) from exc
        _validate_arrays(
            arrays=arrays,
            sequence_length=self.manifest.corpus.sequence_length,
            shard_path=shard_path,
        )
        actual_sha256 = _hash_arrays(**arrays)
        if actual_sha256 != shard.sha256:
            raise ValueError(
                f"streaming shard sha256 mismatch for {shard_path}: "
                f"{actual_sha256} != {shard.sha256}"
            )
        self._cached_shard_path = shard.path
        self._cached_shard_arrays = arrays
        return arrays

    def _shard_for_global_index(self, global_index: int) -> StreamingTokenShardManifest:
        if global_index < 0 or global_index >= self.manifest.corpus.num_sequences:
            raise ValueError(f"global index is outside the dataset: {global_index}")
        for shard in reversed(self.manifest.shards):
            if global_index >= shard.first_sequence_index:
                return shard
        raise AssertionError("unreachable: no shard for global index")


def build_streaming_dataset_from_tokenized_corpus(
    tokenized_corpus: str | Path | LoadedTokenizedCorpus,
    output_dir: str | Path,
    *,
    num_documents: int,
    shard_tokens: int,
    overwrite: bool = False,
    created_at_utc: str | None = None,
    notes: tuple[str, ...] = (),
) -> StreamingDatasetManifest:
    loaded = (
        tokenized_corpus
        if isinstance(tokenized_corpus, LoadedTokenizedCorpus)
        else load_tokenized_corpus(tokenized_corpus)
    )
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise ValueError(f"streaming dataset output already exists: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    shards_dir = output_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in shards_dir.glob("*.npz"):
        stale_path.unlink()

    shard_manifests: list[StreamingTokenShardManifest] = []
    first_sequence_index = 0
    for shard_index, tokenized_shard in enumerate(loaded.manifest.shards):
        source_path = loaded.root / tokenized_shard.path
        with np.load(source_path) as source:
            arrays = {
                "input_ids": np.asarray(source["input_ids"], dtype=np.int32),
                "labels": np.asarray(source["labels"], dtype=np.int32),
                "attention_mask": np.asarray(source["attention_mask"], dtype=np.int32),
                "label_mask": np.asarray(source["loss_mask"], dtype=np.int32),
            }
        shard_name = f"shard_{shard_index:05d}.npz"
        shard_path = shards_dir / shard_name
        np.savez_compressed(shard_path, **arrays)
        shard_manifests.append(
            StreamingTokenShardManifest(
                path=f"shards/{shard_name}",
                num_sequences=int(arrays["input_ids"].shape[0]),
                num_tokens=int(np.count_nonzero(arrays["label_mask"])),
                dtype="int32",
                sha256=_hash_arrays(**arrays),
                first_sequence_index=first_sequence_index,
                provenance=tokenized_shard.path,
            )
        )
        first_sequence_index += int(arrays["input_ids"].shape[0])

    manifest = StreamingDatasetManifest(
        format=STREAMING_DATASET_FORMAT,
        schema_version=STREAMING_DATASET_SCHEMA_VERSION,
        phase=PHASE,
        created_at_utc=created_at_utc or _utc_now(),
        tokenizer=loaded.manifest.tokenizer,
        corpus=StreamingCorpusInfo(
            num_documents=num_documents,
            num_sequences=int(loaded.input_ids.shape[0]),
            num_tokens=int(np.count_nonzero(loaded.loss_mask)),
            sequence_length=loaded.manifest.packing.sequence_length,
            shard_tokens=int(shard_tokens),
            padded_tokens=int(
                loaded.loss_mask.size - np.count_nonzero(loaded.loss_mask)
            ),
        ),
        source=StreamingSourceInfo(
            kind="tokenized_corpus",
            path=str(loaded.root),
            sha256=_hash_file(loaded.root / "manifest.json"),
            tokenized_corpus_format=loaded.manifest.format,
            tokenized_corpus_created_by=loaded.manifest.created_by,
        ),
        shards=tuple(shard_manifests),
        notes=notes,
    )
    validate_streaming_dataset_manifest(manifest)
    (output_path / "manifest.json").write_text(
        json.dumps(_manifest_to_dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def read_streaming_dataset_manifest(path: str | Path) -> StreamingDatasetManifest:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ValueError(f"streaming dataset manifest is missing: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"malformed streaming dataset manifest {manifest_path}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("streaming dataset manifest must be a JSON object")

    tokenizer_payload = _mapping(payload.get("tokenizer"), "manifest.tokenizer")
    corpus_payload = _mapping(payload.get("corpus"), "manifest.corpus")
    source_payload = _mapping(payload.get("source"), "manifest.source")
    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("manifest.shards must be a list")

    manifest = StreamingDatasetManifest(
        format=str(payload.get("format", "")),
        schema_version=str(payload.get("schema_version", "")),
        phase=str(payload.get("phase", "")),
        created_at_utc=str(payload.get("created_at_utc", "")),
        tokenizer=TokenizerMetadata(
            backend=str(tokenizer_payload.get("backend", "")),
            tokenizer_id=_optional_str(tokenizer_payload.get("tokenizer_id")),
            vocab_size=int(tokenizer_payload.get("vocab_size", 0)),
            eos_token_id=_optional_int(tokenizer_payload.get("eos_token_id")),
            pad_token_id=_optional_int(tokenizer_payload.get("pad_token_id")),
            revision=_optional_str(tokenizer_payload.get("revision")),
            unk_token_id=_optional_int(tokenizer_payload.get("unk_token_id")),
        ),
        corpus=StreamingCorpusInfo(
            num_documents=int(corpus_payload.get("num_documents", 0)),
            num_sequences=int(corpus_payload.get("num_sequences", 0)),
            num_tokens=int(corpus_payload.get("num_tokens", 0)),
            sequence_length=int(corpus_payload.get("sequence_length", 0)),
            shard_tokens=int(corpus_payload.get("shard_tokens", 0)),
            padded_tokens=int(corpus_payload.get("padded_tokens", 0)),
            boundary_policy=str(corpus_payload.get("boundary_policy", BOUNDARY_POLICY)),
        ),
        source=StreamingSourceInfo(
            kind=str(source_payload.get("kind", "")),
            path=_optional_str(source_payload.get("path")),
            sha256=_optional_str(source_payload.get("sha256")),
            tokenized_corpus_format=_optional_str(
                source_payload.get("tokenized_corpus_format")
            ),
            tokenized_corpus_created_by=_optional_str(
                source_payload.get("tokenized_corpus_created_by")
            ),
        ),
        shards=tuple(
            StreamingTokenShardManifest(
                path=str(shard_payload.get("path", "")),
                num_sequences=int(shard_payload.get("num_sequences", 0)),
                num_tokens=int(shard_payload.get("num_tokens", 0)),
                dtype=str(shard_payload.get("dtype", "")),
                sha256=str(shard_payload.get("sha256", "")),
                first_sequence_index=int(shard_payload.get("first_sequence_index", 0)),
                provenance=str(shard_payload.get("provenance", "")),
            )
            for shard_payload in (
                _mapping(item, "manifest.shards[]") for item in raw_shards
            )
        ),
        created_by=str(payload.get("created_by", "")),
        notes=tuple(str(note) for note in payload.get("notes", [])),
    )
    validate_streaming_dataset_manifest(manifest)
    return manifest


def validate_streaming_dataset_manifest(manifest: StreamingDatasetManifest) -> None:
    if manifest.format != STREAMING_DATASET_FORMAT:
        raise ValueError(f"unsupported streaming dataset format: {manifest.format!r}")
    if manifest.schema_version != STREAMING_DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported streaming dataset schema_version: {manifest.schema_version!r}"
        )
    if manifest.phase != PHASE:
        raise ValueError(f"unsupported streaming dataset phase: {manifest.phase!r}")
    if manifest.created_by != STREAMING_DATASET_CREATED_BY:
        raise ValueError("streaming dataset created_by is not recognized")
    if not manifest.created_at_utc:
        raise ValueError("streaming dataset created_at_utc is required")
    if manifest.tokenizer.vocab_size <= 0:
        raise ValueError("streaming dataset tokenizer.vocab_size must be > 0")
    if manifest.tokenizer.pad_token_id is None:
        raise ValueError("streaming dataset tokenizer.pad_token_id is required")
    if manifest.corpus.num_documents <= 0:
        raise ValueError("streaming dataset corpus.num_documents must be > 0")
    if manifest.corpus.num_sequences <= 0:
        raise ValueError("streaming dataset corpus.num_sequences must be > 0")
    if manifest.corpus.num_tokens <= 0:
        raise ValueError("streaming dataset corpus.num_tokens must be > 0")
    if manifest.corpus.sequence_length <= 1:
        raise ValueError("streaming dataset corpus.sequence_length must be > 1")
    if manifest.corpus.shard_tokens <= 0:
        raise ValueError("streaming dataset corpus.shard_tokens must be > 0")
    if manifest.source.kind != "tokenized_corpus":
        raise ValueError("streaming dataset source.kind must be 'tokenized_corpus'")
    if not manifest.shards:
        raise ValueError("streaming dataset manifest must list at least one shard")

    expected_first_sequence_index = 0
    total_sequences = 0
    total_tokens = 0
    for shard in manifest.shards:
        if not shard.path.startswith("shards/") or not shard.path.endswith(".npz"):
            raise ValueError(
                "streaming shard path must be a relative NPZ under shards/"
            )
        if shard.num_sequences <= 0:
            raise ValueError("streaming shard num_sequences must be > 0")
        if shard.num_tokens <= 0:
            raise ValueError("streaming shard num_tokens must be > 0")
        if shard.dtype != "int32":
            raise ValueError("streaming shard dtype must be int32")
        if len(shard.sha256) != 64:
            raise ValueError("streaming shard sha256 must be a 64-char hex digest")
        if shard.first_sequence_index != expected_first_sequence_index:
            raise ValueError("streaming shard first_sequence_index is not contiguous")
        if not shard.provenance:
            raise ValueError("streaming shard provenance is required")
        expected_first_sequence_index += shard.num_sequences
        total_sequences += shard.num_sequences
        total_tokens += shard.num_tokens

    if total_sequences != manifest.corpus.num_sequences:
        raise ValueError("streaming corpus num_sequences does not match shards")
    if total_tokens != manifest.corpus.num_tokens:
        raise ValueError("streaming corpus num_tokens does not match shards")
    if manifest.corpus.padded_tokens < 0:
        raise ValueError("streaming corpus padded_tokens must be >= 0")


def _validate_arrays(
    *,
    arrays: dict[str, np.ndarray],
    sequence_length: int,
    shard_path: Path,
) -> None:
    expected_shape: tuple[int, int] | None = None
    for name, array in arrays.items():
        if array.ndim != 2 or int(array.shape[1]) != sequence_length:
            raise ValueError(f"invalid {name} shape in {shard_path}: {array.shape}")
        if expected_shape is None:
            expected_shape = (int(array.shape[0]), int(array.shape[1]))
        elif array.shape != expected_shape:
            raise ValueError(f"streaming shard arrays shape mismatch in {shard_path}")


def _pad_batch(
    arrays: dict[str, np.ndarray],
    *,
    batch_size: int,
    sequence_length: int,
    pad_token_id: int,
) -> dict[str, np.ndarray]:
    pad_count = batch_size - int(arrays["input_ids"].shape[0])
    if pad_count <= 0:
        return arrays
    pad_tokens = np.full((pad_count, sequence_length), pad_token_id, dtype=np.int32)
    zero_mask = np.zeros((pad_count, sequence_length), dtype=np.int32)
    return {
        "input_ids": np.concatenate([arrays["input_ids"], pad_tokens], axis=0),
        "labels": np.concatenate([arrays["labels"], pad_tokens], axis=0),
        "attention_mask": np.concatenate(
            [arrays["attention_mask"], zero_mask],
            axis=0,
        ),
        "label_mask": np.concatenate([arrays["label_mask"], zero_mask], axis=0),
    }


def _empty_batch(sequence_length: int) -> dict[str, np.ndarray]:
    return {
        name: np.empty((0, sequence_length), dtype=np.int32)
        for name in ("input_ids", "labels", "attention_mask", "label_mask")
    }


def _manifest_to_dict(manifest: StreamingDatasetManifest) -> dict[str, Any]:
    return asdict(manifest)


def _hash_arrays(**arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pad_token_id(tokenizer: TokenizerMetadata) -> int:
    if tokenizer.pad_token_id is None:
        raise ValueError("streaming dataset tokenizer.pad_token_id is required")
    return int(tokenizer.pad_token_id)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
