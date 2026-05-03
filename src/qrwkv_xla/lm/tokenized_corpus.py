from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from qrwkv_xla.generation import TokenizerConfig, TokenizerMetadata
from qrwkv_xla.generation.tokenizer import Tokenizer
from qrwkv_xla.prompting import (
    PromptCorpus,
    build_prompt_corpus_manifest,
    filter_prompt_corpus,
    read_prompt_corpus,
)

TOKENIZED_CORPUS_FORMAT = "qrwkv_xla.tokenized_corpus"
TOKENIZED_CORPUS_FORMAT_VERSION = 1
TOKENIZED_CORPUS_CREATED_BY = "qrwkv_xla.lm.tokenized_corpus"


@dataclass(frozen=True)
class TokenizedCorpusSource:
    kind: str
    path: str | None
    sha256: str
    record_count: int
    selected_count: int
    corpus_id: str | None = None
    prompt_split: str | None = None
    prompt_tags: tuple[str, ...] = ()
    prompt_limit: int | None = None


@dataclass(frozen=True)
class TokenizedCorpusPacking:
    sequence_length: int
    append_eos: bool = True
    drop_remainder: bool = True
    stride: int | None = None
    policy: str = "concat_pack_v1"


@dataclass(frozen=True)
class TokenizedShardInfo:
    path: str
    sha256: str
    num_sequences: int
    num_tokens: int


@dataclass(frozen=True)
class TokenizedCorpusTotals:
    num_shards: int
    num_sequences: int
    num_tokens: int


@dataclass(frozen=True)
class TokenizedCorpusManifest:
    format: str
    format_version: int
    created_at: str
    source: TokenizedCorpusSource
    tokenizer: TokenizerMetadata
    packing: TokenizedCorpusPacking
    shards: tuple[TokenizedShardInfo, ...]
    totals: TokenizedCorpusTotals
    created_by: str = TOKENIZED_CORPUS_CREATED_BY
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LoadedTokenizedCorpus:
    root: Path
    manifest: TokenizedCorpusManifest
    input_ids: np.ndarray
    labels: np.ndarray
    attention_mask: np.ndarray
    loss_mask: np.ndarray

    @property
    def token_sequences(self) -> tuple[tuple[int, ...], ...]:
        sequences: list[tuple[int, ...]] = []
        for input_row, label_row in zip(self.input_ids, self.labels, strict=True):
            sequences.append(
                tuple([*(int(token_id) for token_id in input_row), int(label_row[-1])])
            )
        return tuple(sequences)


def build_tokenized_sequences(
    corpus: PromptCorpus,
    tokenizer: Tokenizer,
    *,
    sequence_length: int,
    prompt_split: str | None = "train",
    prompt_tags: tuple[str, ...] = (),
    prompt_limit: int | None = None,
    append_eos: bool = True,
    drop_remainder: bool = True,
    stride: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    if sequence_length <= 1:
        raise ValueError("sequence_length must be > 1")
    eos_token_id = tokenizer.metadata.eos_token_id
    pad_token_id = tokenizer.metadata.pad_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must expose eos_token_id")
    if pad_token_id is None:
        raise ValueError("Tokenizer must expose pad_token_id")
    filtered = filter_prompt_corpus(
        corpus,
        split=prompt_split,
        tags=prompt_tags,
        limit=prompt_limit,
    )
    if not filtered.records:
        raise ValueError("Prompt corpus selection produced no records")

    packed: list[int] = []
    for record in filtered.records:
        encoded = [int(token_id) for token_id in tokenizer.encode(record.text)]
        packed.extend(encoded)
        if append_eos and (not encoded or encoded[-1] != eos_token_id):
            packed.append(eos_token_id)
    if not packed:
        raise ValueError("Tokenized prompt corpus produced no tokens")

    example_length = sequence_length + 1
    pack_stride = sequence_length if stride is None else stride
    if pack_stride <= 0:
        raise ValueError("stride must be > 0")

    sequences: list[tuple[int, ...]] = []
    for start in range(0, len(packed), pack_stride):
        chunk = packed[start : start + example_length]
        if len(chunk) < example_length:
            if drop_remainder:
                break
            chunk = [*chunk, *([pad_token_id] * (example_length - len(chunk)))]
        sequences.append(tuple(chunk))
    if not sequences:
        raise ValueError(
            "Tokenized prompt corpus produced no full sequences for the requested "
            f"sequence_length={sequence_length}"
        )
    return tuple(sequences)


def write_tokenized_corpus(
    sequences: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
    output_dir: str | Path,
    *,
    sequence_length: int,
    tokenizer: TokenizerMetadata,
    source: TokenizedCorpusSource,
    shard_size_tokens: int = 4096,
    overwrite: bool = False,
    packing: TokenizedCorpusPacking | None = None,
    created_at: str | None = None,
    notes: tuple[str, ...] = (),
) -> TokenizedCorpusManifest:
    if sequence_length <= 1:
        raise ValueError("sequence_length must be > 1")
    if shard_size_tokens <= 0:
        raise ValueError("shard_size_tokens must be > 0")
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer metadata must expose eos_token_id and pad_token_id")

    rows = [tuple(int(token_id) for token_id in row) for row in sequences]
    if not rows:
        raise ValueError("Tokenized corpus must contain at least one sequence")

    example_length = sequence_length + 1
    token_array = np.asarray(rows, dtype=np.int32)
    if token_array.ndim != 2 or token_array.shape[1] != example_length:
        raise ValueError(
            "Tokenized corpus sequences must have shape "
            f"[num_sequences, {example_length}]"
        )

    input_ids = np.ascontiguousarray(token_array[:, :-1], dtype=np.int32)
    labels = np.ascontiguousarray(token_array[:, 1:], dtype=np.int32)
    attention_mask = np.ascontiguousarray(
        input_ids != tokenizer.pad_token_id,
        dtype=np.int32,
    )
    loss_mask = np.ascontiguousarray(labels != tokenizer.pad_token_id, dtype=np.int32)

    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise ValueError(f"Tokenized corpus output already exists: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    shards_dir = output_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in shards_dir.glob("*.npz"):
        stale_path.unlink()

    sequences_per_shard = max(1, shard_size_tokens // sequence_length)
    shard_infos: list[TokenizedShardInfo] = []
    for shard_index, start in enumerate(
        range(0, input_ids.shape[0], sequences_per_shard)
    ):
        stop = min(start + sequences_per_shard, input_ids.shape[0])
        shard_input_ids = input_ids[start:stop]
        shard_labels = labels[start:stop]
        shard_attention_mask = attention_mask[start:stop]
        shard_loss_mask = loss_mask[start:stop]
        shard_name = f"shard-{shard_index:05d}.npz"
        shard_path = shards_dir / shard_name
        np.savez_compressed(
            shard_path,
            input_ids=shard_input_ids,
            labels=shard_labels,
            attention_mask=shard_attention_mask,
            loss_mask=shard_loss_mask,
        )
        shard_infos.append(
            TokenizedShardInfo(
                path=f"shards/{shard_name}",
                sha256=_hash_arrays(
                    input_ids=shard_input_ids,
                    labels=shard_labels,
                    attention_mask=shard_attention_mask,
                    loss_mask=shard_loss_mask,
                ),
                num_sequences=int(shard_input_ids.shape[0]),
                num_tokens=int(np.count_nonzero(shard_loss_mask)),
            )
        )

    manifest = TokenizedCorpusManifest(
        format=TOKENIZED_CORPUS_FORMAT,
        format_version=TOKENIZED_CORPUS_FORMAT_VERSION,
        created_at=created_at or _utc_now(),
        source=source,
        tokenizer=tokenizer,
        packing=packing
        or TokenizedCorpusPacking(
            sequence_length=sequence_length,
            append_eos=True,
            drop_remainder=True,
            stride=sequence_length,
        ),
        shards=tuple(shard_infos),
        totals=TokenizedCorpusTotals(
            num_shards=len(shard_infos),
            num_sequences=int(input_ids.shape[0]),
            num_tokens=int(np.count_nonzero(loss_mask)),
        ),
        notes=notes,
    )
    validate_tokenized_corpus_manifest(manifest)
    (output_path / "manifest.json").write_text(
        json.dumps(_manifest_to_dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def write_tokenized_corpus_from_prompt_jsonl(
    prompt_corpus: str | Path,
    output_dir: str | Path,
    *,
    tokenizer: Tokenizer,
    sequence_length: int,
    prompt_split: str | None = "train",
    prompt_tags: tuple[str, ...] = (),
    prompt_limit: int | None = None,
    shard_size_tokens: int = 4096,
    overwrite: bool = False,
    append_eos: bool = True,
    drop_remainder: bool = True,
    stride: int | None = None,
    created_at: str | None = None,
) -> TokenizedCorpusManifest:
    corpus = read_prompt_corpus(prompt_corpus)
    filtered = filter_prompt_corpus(
        corpus,
        split=prompt_split,
        tags=prompt_tags,
        limit=prompt_limit,
    )
    if not filtered.records:
        raise ValueError("Prompt corpus selection produced no records")

    sequences = build_tokenized_sequences(
        corpus,
        tokenizer,
        sequence_length=sequence_length,
        prompt_split=prompt_split,
        prompt_tags=prompt_tags,
        prompt_limit=prompt_limit,
        append_eos=append_eos,
        drop_remainder=drop_remainder,
        stride=stride,
    )
    filtered_manifest = build_prompt_corpus_manifest(
        filtered,
        description="Tokenized corpus source selection.",
        notes=["packed for Stage 3 next-token CE"],
    )
    source = TokenizedCorpusSource(
        kind="jsonl_prompts",
        path=str(corpus.source_path) if corpus.source_path is not None else None,
        sha256=filtered_manifest.sha256,
        record_count=len(corpus.records),
        selected_count=len(filtered.records),
        corpus_id=corpus.corpus_id,
        prompt_split=prompt_split,
        prompt_tags=tuple(prompt_tags),
        prompt_limit=prompt_limit,
    )
    packing = TokenizedCorpusPacking(
        sequence_length=sequence_length,
        append_eos=append_eos,
        drop_remainder=drop_remainder,
        stride=sequence_length if stride is None else stride,
    )
    return write_tokenized_corpus(
        sequences,
        output_dir,
        sequence_length=sequence_length,
        tokenizer=tokenizer.metadata,
        source=source,
        shard_size_tokens=shard_size_tokens,
        overwrite=overwrite,
        packing=packing,
        created_at=created_at,
        notes=("static Stage 3 next-token CE batches",),
    )


def load_tokenized_corpus(
    root: str | Path,
    *,
    expected_sequence_length: int | None = None,
    expected_tokenizer: TokenizerMetadata | TokenizerConfig | None = None,
) -> LoadedTokenizedCorpus:
    root_path = Path(root)
    manifest = read_tokenized_corpus_manifest(root_path / "manifest.json")
    validate_tokenized_corpus_manifest(
        manifest,
        expected_sequence_length=expected_sequence_length,
        expected_tokenizer=expected_tokenizer,
    )

    input_shards: list[np.ndarray] = []
    label_shards: list[np.ndarray] = []
    attention_shards: list[np.ndarray] = []
    loss_shards: list[np.ndarray] = []

    for shard_info in manifest.shards:
        shard_path = root_path / shard_info.path
        if not shard_path.is_file():
            raise ValueError(f"Tokenized corpus shard is missing: {shard_path}")
        with np.load(shard_path) as shard:
            try:
                shard_input_ids = np.asarray(shard["input_ids"], dtype=np.int32)
                shard_labels = np.asarray(shard["labels"], dtype=np.int32)
                shard_attention_mask = np.asarray(
                    shard["attention_mask"], dtype=np.int32
                )
                shard_loss_mask = np.asarray(shard["loss_mask"], dtype=np.int32)
            except KeyError as exc:
                raise ValueError(
                    "Tokenized corpus shard is missing required array "
                    f"{exc.args[0]!r}: {shard_path}"
                ) from exc
        _validate_shard_arrays(
            shard_path=shard_path,
            input_ids=shard_input_ids,
            labels=shard_labels,
            attention_mask=shard_attention_mask,
            loss_mask=shard_loss_mask,
            sequence_length=manifest.packing.sequence_length,
        )
        actual_hash = _hash_arrays(
            input_ids=shard_input_ids,
            labels=shard_labels,
            attention_mask=shard_attention_mask,
            loss_mask=shard_loss_mask,
        )
        if actual_hash != shard_info.sha256:
            raise ValueError(
                "Tokenized corpus shard sha256 mismatch for "
                f"{shard_path}: {actual_hash} != {shard_info.sha256}"
            )
        if int(shard_input_ids.shape[0]) != shard_info.num_sequences:
            raise ValueError(
                "Tokenized corpus shard num_sequences mismatch for "
                f"{shard_path}: {shard_input_ids.shape[0]} != "
                f"{shard_info.num_sequences}"
            )
        if int(np.count_nonzero(shard_loss_mask)) != shard_info.num_tokens:
            raise ValueError(
                "Tokenized corpus shard num_tokens mismatch for "
                f"{shard_path}: {int(np.count_nonzero(shard_loss_mask))} != "
                f"{shard_info.num_tokens}"
            )
        input_shards.append(shard_input_ids)
        label_shards.append(shard_labels)
        attention_shards.append(shard_attention_mask)
        loss_shards.append(shard_loss_mask)

    if not input_shards:
        raise ValueError("Tokenized corpus manifest lists no shards")

    input_ids = np.concatenate(input_shards, axis=0)
    labels = np.concatenate(label_shards, axis=0)
    attention_mask = np.concatenate(attention_shards, axis=0)
    loss_mask = np.concatenate(loss_shards, axis=0)

    if int(input_ids.shape[0]) != manifest.totals.num_sequences:
        raise ValueError(
            "Tokenized corpus totals.num_sequences mismatch: "
            f"{input_ids.shape[0]} != {manifest.totals.num_sequences}"
        )
    if int(np.count_nonzero(loss_mask)) != manifest.totals.num_tokens:
        raise ValueError(
            "Tokenized corpus totals.num_tokens mismatch: "
            f"{int(np.count_nonzero(loss_mask))} != {manifest.totals.num_tokens}"
        )

    return LoadedTokenizedCorpus(
        root=root_path,
        manifest=manifest,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
    )


def read_tokenized_corpus_manifest(path: str | Path) -> TokenizedCorpusManifest:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ValueError(f"Tokenized corpus manifest is missing: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Malformed tokenized corpus manifest {manifest_path}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Tokenized corpus manifest must be a JSON object")

    source_payload = _mapping(payload.get("source"), "manifest.source")
    tokenizer_payload = _mapping(payload.get("tokenizer"), "manifest.tokenizer")
    packing_payload = _mapping(payload.get("packing"), "manifest.packing")
    totals_payload = _mapping(payload.get("totals"), "manifest.totals")
    shards_payload = payload.get("shards")
    if not isinstance(shards_payload, list):
        raise ValueError("manifest.shards must be a list")

    manifest = TokenizedCorpusManifest(
        format=str(payload.get("format", "")),
        format_version=int(payload.get("format_version", 0)),
        created_at=str(payload.get("created_at", "")),
        source=TokenizedCorpusSource(
            kind=str(source_payload.get("kind", "")),
            path=_optional_str(source_payload.get("path")),
            sha256=str(source_payload.get("sha256", "")),
            record_count=int(source_payload.get("record_count", 0)),
            selected_count=int(source_payload.get("selected_count", 0)),
            corpus_id=_optional_str(source_payload.get("corpus_id")),
            prompt_split=_optional_str(source_payload.get("prompt_split")),
            prompt_tags=tuple(
                str(tag) for tag in source_payload.get("prompt_tags", [])
            ),
            prompt_limit=_optional_int(source_payload.get("prompt_limit")),
        ),
        tokenizer=TokenizerMetadata(
            backend=str(tokenizer_payload.get("backend", "")),
            tokenizer_id=_optional_str(tokenizer_payload.get("tokenizer_id")),
            vocab_size=int(tokenizer_payload.get("vocab_size", 0)),
            eos_token_id=_optional_int(tokenizer_payload.get("eos_token_id")),
            pad_token_id=_optional_int(tokenizer_payload.get("pad_token_id")),
            revision=_optional_str(tokenizer_payload.get("revision")),
            unk_token_id=_optional_int(tokenizer_payload.get("unk_token_id")),
        ),
        packing=TokenizedCorpusPacking(
            sequence_length=int(packing_payload.get("sequence_length", 0)),
            append_eos=bool(packing_payload.get("append_eos", True)),
            drop_remainder=bool(packing_payload.get("drop_remainder", True)),
            stride=_optional_int(packing_payload.get("stride")),
            policy=str(packing_payload.get("policy", "")),
        ),
        shards=tuple(
            TokenizedShardInfo(
                path=str(item.get("path", "")),
                sha256=str(item.get("sha256", "")),
                num_sequences=int(item.get("num_sequences", 0)),
                num_tokens=int(item.get("num_tokens", 0)),
            )
            for item in (
                _mapping(entry, "manifest.shards[]") for entry in shards_payload
            )
        ),
        totals=TokenizedCorpusTotals(
            num_shards=int(totals_payload.get("num_shards", 0)),
            num_sequences=int(totals_payload.get("num_sequences", 0)),
            num_tokens=int(totals_payload.get("num_tokens", 0)),
        ),
        created_by=str(payload.get("created_by", "")),
        notes=tuple(str(note) for note in payload.get("notes", [])),
    )
    validate_tokenized_corpus_manifest(manifest)
    return manifest


def validate_tokenized_corpus_manifest(
    manifest: TokenizedCorpusManifest,
    *,
    expected_sequence_length: int | None = None,
    expected_tokenizer: TokenizerMetadata | TokenizerConfig | None = None,
) -> None:
    if manifest.format != TOKENIZED_CORPUS_FORMAT:
        raise ValueError(f"Unsupported tokenized corpus format: {manifest.format!r}")
    if manifest.format_version != TOKENIZED_CORPUS_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported tokenized corpus format_version: {manifest.format_version!r}"
        )
    if not manifest.created_at:
        raise ValueError("Tokenized corpus manifest created_at is required")
    if manifest.source.kind != "jsonl_prompts":
        raise ValueError("Tokenized corpus source.kind must be 'jsonl_prompts'")
    if len(manifest.source.sha256) != 64:
        raise ValueError("Tokenized corpus source.sha256 must be a hex SHA-256 digest")
    if manifest.source.record_count <= 0:
        raise ValueError("Tokenized corpus source.record_count must be > 0")
    if manifest.source.selected_count <= 0:
        raise ValueError("Tokenized corpus source.selected_count must be > 0")
    if manifest.tokenizer.vocab_size <= 0:
        raise ValueError("Tokenized corpus tokenizer.vocab_size must be > 0")
    if manifest.tokenizer.eos_token_id is None:
        raise ValueError("Tokenized corpus tokenizer.eos_token_id is required")
    if manifest.tokenizer.pad_token_id is None:
        raise ValueError("Tokenized corpus tokenizer.pad_token_id is required")
    if manifest.packing.sequence_length <= 1:
        raise ValueError("Tokenized corpus packing.sequence_length must be > 1")
    if manifest.packing.stride is None or manifest.packing.stride <= 0:
        raise ValueError("Tokenized corpus packing.stride must be > 0")
    if not manifest.shards:
        raise ValueError("Tokenized corpus manifest must list at least one shard")
    for shard in manifest.shards:
        if not shard.path.startswith("shards/") or not shard.path.endswith(".npz"):
            raise ValueError("Tokenized corpus shard paths must be relative NPZ paths")
        if len(shard.sha256) != 64:
            raise ValueError(
                "Tokenized corpus shard.sha256 must be a hex SHA-256 digest"
            )
        if shard.num_sequences <= 0:
            raise ValueError("Tokenized corpus shard.num_sequences must be > 0")
        if shard.num_tokens <= 0:
            raise ValueError("Tokenized corpus shard.num_tokens must be > 0")
    if manifest.totals.num_shards != len(manifest.shards):
        raise ValueError(
            "Tokenized corpus totals.num_shards mismatch: "
            f"{manifest.totals.num_shards} != {len(manifest.shards)}"
        )
    if manifest.totals.num_sequences <= 0:
        raise ValueError("Tokenized corpus totals.num_sequences must be > 0")
    if manifest.totals.num_tokens <= 0:
        raise ValueError("Tokenized corpus totals.num_tokens must be > 0")
    if expected_sequence_length is not None and (
        manifest.packing.sequence_length != expected_sequence_length
    ):
        raise ValueError(
            "Tokenized corpus sequence_length mismatch: "
            f"{manifest.packing.sequence_length} != {expected_sequence_length}"
        )
    if expected_tokenizer is not None:
        _validate_tokenizer_match(manifest.tokenizer, expected_tokenizer)


def _validate_tokenizer_match(
    actual: TokenizerMetadata,
    expected: TokenizerMetadata | TokenizerConfig,
) -> None:
    if isinstance(expected, TokenizerConfig):
        if actual.backend != expected.backend:
            raise ValueError(
                "Tokenized corpus tokenizer backend mismatch: "
                f"{actual.backend!r} != {expected.backend!r}"
            )
        if (
            expected.tokenizer_id is not None
            and actual.tokenizer_id != expected.tokenizer_id
        ):
            raise ValueError(
                "Tokenized corpus tokenizer tokenizer_id mismatch: "
                f"{actual.tokenizer_id!r} != {expected.tokenizer_id!r}"
            )
        if expected.vocab_size is not None and actual.vocab_size != expected.vocab_size:
            raise ValueError(
                "Tokenized corpus tokenizer vocab_size mismatch: "
                f"{actual.vocab_size!r} != {expected.vocab_size!r}"
            )
        for name in ("eos_token_id", "pad_token_id", "revision"):
            expected_value = getattr(expected, name)
            if expected_value is not None and getattr(actual, name) != expected_value:
                raise ValueError(
                    f"Tokenized corpus tokenizer {name} mismatch: "
                    f"{getattr(actual, name)!r} != {expected_value!r}"
                )
        return

    for name in (
        "backend",
        "tokenizer_id",
        "vocab_size",
        "eos_token_id",
        "pad_token_id",
        "revision",
    ):
        actual_value = getattr(actual, name)
        expected_value = getattr(expected, name)
        if actual_value != expected_value:
            raise ValueError(
                f"Tokenized corpus tokenizer {name} mismatch: "
                f"{actual_value!r} != {expected_value!r}"
            )


def _validate_shard_arrays(
    *,
    shard_path: Path,
    input_ids: np.ndarray,
    labels: np.ndarray,
    attention_mask: np.ndarray,
    loss_mask: np.ndarray,
    sequence_length: int,
) -> None:
    for name, array in (
        ("input_ids", input_ids),
        ("labels", labels),
        ("attention_mask", attention_mask),
        ("loss_mask", loss_mask),
    ):
        if array.ndim != 2:
            raise ValueError(f"Tokenized corpus {name} must be rank 2: {shard_path}")
        if array.shape[1] != sequence_length:
            raise ValueError(
                f"Tokenized corpus {name} shape mismatch: {shard_path} has "
                f"{array.shape}, expected [*, {sequence_length}]"
            )
    if input_ids.shape != labels.shape:
        raise ValueError(f"Tokenized corpus labels shape mismatch: {shard_path}")
    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            f"Tokenized corpus attention_mask shape mismatch: {shard_path}"
        )
    if loss_mask.shape != input_ids.shape:
        raise ValueError(f"Tokenized corpus loss_mask shape mismatch: {shard_path}")


def _hash_arrays(**arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _manifest_to_dict(manifest: TokenizedCorpusManifest) -> dict[str, Any]:
    payload = asdict(manifest)
    payload["format_version"] = int(payload["format_version"])
    return payload


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
