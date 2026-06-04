from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from qrwkv_xla.artifacts._json import write_json
from qrwkv_xla.artifacts.teacher_textbook import (
    TeacherTextbookValidationReport,
    validate_teacher_textbook,
    write_teacher_textbook_validation_report,
)
from qrwkv_xla.targets.schema import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
    TargetStoreMetadata,
)
from qrwkv_xla.targets.store import TeacherTargetStore

DEFAULT_FAKE_TEACHER_MODEL_ID = "fake-deterministic-teacher"
DEFAULT_FAKE_TOKENIZER_ID = "fake-deterministic-tokenizer"
DEFAULT_FAKE_VOCAB_SIZE = 32
DEFAULT_TEXT_EXAMPLES = (
    "hello world",
    "tiny teacher textbook",
    "radjax matched vocab",
    "current qrwkv student",
)
CLAIMS_NOT_MADE = (
    "no_model_quality_claim",
    "no_real_hf_teacher_claim",
    "no_qwen_parity_claim",
    "no_training_claim",
    "no_remote_teacher_service_claim",
)


@dataclass(frozen=True)
class TinyTextExample:
    example_id: str
    text: str


@dataclass(frozen=True)
class TeacherTextbookBuildConfig:
    output_dir: Path
    dataset_path: Path | None = None
    teacher_mode: str = "fake"
    teacher_model_id: str = DEFAULT_FAKE_TEACHER_MODEL_ID
    sequence_length: int = 16
    batch_size: int = 2
    max_examples: int = 4
    logits_dtype: str = "float32"
    local_files_only: bool = True
    allow_downloads: bool = False
    seed: int = 0
    overwrite: bool = False
    vocab_size: int = DEFAULT_FAKE_VOCAB_SIZE
    include_hidden_states: bool = False


def build_teacher_textbook(
    config: TeacherTextbookBuildConfig,
) -> TeacherTextbookValidationReport:
    if config.teacher_mode == "fake":
        return build_fake_teacher_textbook(config)
    if config.teacher_mode == "hf":
        raise RuntimeError(
            "teacher-mode=hf is intentionally guarded in P116.1. "
            "Use teacher-mode=fake for CI-safe textbook builds, or run a future "
            "HF builder extension with explicit optional dependencies and model "
            "availability."
        )
    raise ValueError(f"unsupported teacher_mode: {config.teacher_mode!r}")


def build_fake_teacher_textbook(
    config: TeacherTextbookBuildConfig,
) -> TeacherTextbookValidationReport:
    _validate_config(config)
    examples = load_text_examples(config.dataset_path, max_examples=config.max_examples)
    if config.output_dir.exists():
        if not config.overwrite:
            raise ValueError(
                f"TeacherTextbook output already exists: {config.output_dir}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(config.output_dir)

    created_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    shard_count = (len(examples) + config.batch_size - 1) // config.batch_size
    metadata = TargetStoreMetadata(
        schema_version=TEACHER_TARGET_STORE_SCHEMA_VERSION,
        target_store_version=TEACHER_TARGET_STORE_VERSION,
        model_id=config.teacher_model_id,
        model_family="fake",
        tokenizer_id=DEFAULT_FAKE_TOKENIZER_ID,
        tokenizer_hash=None,
        vocab_size=config.vocab_size,
        target_type="synthetic",
        dtype=_canonical_dtype(config.logits_dtype),
        sequence_length=config.sequence_length,
        num_examples=len(examples),
        shard_count=shard_count,
        created_by="qrwkv_xla.artifacts.teacher_textbook_builder",
        created_at=created_at,
        source={"kind": _dataset_source(config.dataset_path)},
        provenance={"phase": "P116.1", "teacher_mode": "fake"},
    )
    store = TeacherTargetStore.create(config.output_dir, metadata, overwrite=True)
    for shard_id, start in enumerate(range(0, len(examples), config.batch_size)):
        batch = examples[start : start + config.batch_size]
        store.write_shard(shard_id, _fake_arrays(batch, config))

    _write_sidecars(config, examples, created_at)
    report = validate_teacher_textbook(config.output_dir)
    write_teacher_textbook_validation_report(
        report,
        config.output_dir / "validation_report.json",
    )
    if report.status != "pass":
        raise ValueError(
            "built TeacherTextbook failed validation: " + "; ".join(report.blockers)
        )
    return validate_teacher_textbook(config.output_dir)


def load_text_examples(
    path: Path | None,
    *,
    max_examples: int,
) -> tuple[TinyTextExample, ...]:
    if max_examples <= 0:
        raise ValueError("max_examples must be > 0")
    if path is None:
        examples = [
            TinyTextExample(example_id=f"builtin-{idx:04d}", text=text)
            for idx, text in enumerate(DEFAULT_TEXT_EXAMPLES)
        ]
        return tuple(examples[:max_examples])

    loaded: list[TinyTextExample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if len(loaded) >= max_examples:
                break
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            text = str(payload.get("text", ""))
            if not text.strip():
                raise ValueError(f"{path}:{line_number} text must be non-empty")
            example_id = str(payload.get("example_id") or f"row-{line_number:06d}")
            loaded.append(TinyTextExample(example_id=example_id, text=text))
    if not loaded:
        raise ValueError(f"dataset contains no usable text examples: {path}")
    return tuple(loaded)


def _validate_config(config: TeacherTextbookBuildConfig) -> None:
    if config.sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if config.max_examples <= 0:
        raise ValueError("max_examples must be > 0")
    if config.vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")
    _canonical_dtype(config.logits_dtype)


def _fake_arrays(
    examples: tuple[TinyTextExample, ...],
    config: TeacherTextbookBuildConfig,
) -> dict[str, np.ndarray]:
    input_ids = np.zeros((len(examples), config.sequence_length), dtype=np.int32)
    attention_mask = np.zeros_like(input_ids)
    logits = np.zeros(
        (len(examples), config.sequence_length, config.vocab_size),
        dtype=np.dtype(_canonical_dtype(config.logits_dtype)),
    )
    for row, example in enumerate(examples):
        ids = _fake_token_ids(example.text, config)
        input_ids[row, : len(ids)] = ids
        attention_mask[row, : len(ids)] = 1
        for pos in range(config.sequence_length):
            token_id = int(input_ids[row, pos])
            logits[row, pos, :] = _fake_logits(
                token_id=token_id,
                position=pos,
                row_seed=config.seed + row,
                vocab_size=config.vocab_size,
            )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "logits": logits,
    }


def _fake_token_ids(
    text: str,
    config: TeacherTextbookBuildConfig,
) -> np.ndarray:
    encoded = text.encode("utf-8")
    usable = encoded[: config.sequence_length]
    ids = [((byte + config.seed) % (config.vocab_size - 1)) + 1 for byte in usable]
    if not ids:
        raise ValueError("text must encode to at least one token")
    return np.asarray(ids, dtype=np.int32)


def _fake_logits(
    *,
    token_id: int,
    position: int,
    row_seed: int,
    vocab_size: int,
) -> np.ndarray:
    vocab = np.arange(vocab_size, dtype=np.float32)
    target = (token_id + position + row_seed) % vocab_size
    return -np.abs(vocab - float(target)) / max(vocab_size, 1)


def _write_sidecars(
    config: TeacherTextbookBuildConfig,
    examples: tuple[TinyTextExample, ...],
    created_at: str,
) -> None:
    shard_count = (len(examples) + config.batch_size - 1) // config.batch_size
    output_dir = config.output_dir
    write_json(
        output_dir / "vocab_contract.json",
        {
            "tokenizer_id": DEFAULT_FAKE_TOKENIZER_ID,
            "tokenizer_hash": None,
            "vocab_size": config.vocab_size,
            "model_id": config.teacher_model_id,
            "model_family": "fake",
        },
    )
    write_json(
        output_dir / "teacher_manifest.json",
        {
            "artifact_type": "teacher_textbook",
            "artifact_version": 0,
            "teacher_model_id": config.teacher_model_id,
            "teacher_backend_type": "fake",
            "teacher_revision_or_hash": None,
            "tokenizer_id": DEFAULT_FAKE_TOKENIZER_ID,
            "vocab_size": config.vocab_size,
            "vocab_contract_path": "vocab_contract.json",
            "target_type": "synthetic",
            "dtype": _canonical_dtype(config.logits_dtype),
            "sequence_length": config.sequence_length,
            "num_examples": len(examples),
            "shard_count": shard_count,
            "created_at": created_at,
            "local_files_only": config.local_files_only,
            "allow_downloads": config.allow_downloads,
            "claims_not_made": list(CLAIMS_NOT_MADE),
        },
    )
    write_json(
        output_dir / "emission_config.json",
        {
            "dataset_source": _dataset_source(config.dataset_path),
            "max_examples": config.max_examples,
            "batch_size": config.batch_size,
            "sequence_length": config.sequence_length,
            "logits_dtype": _canonical_dtype(config.logits_dtype),
            "include_hidden_states": config.include_hidden_states,
            "sampling_used": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "seed": config.seed,
            "teacher_mode": config.teacher_mode,
        },
    )


def _dataset_source(path: Path | None) -> str:
    return "builtin_examples" if path is None else str(path)


def _canonical_dtype(dtype: str) -> str:
    value = {
        "fp32": "float32",
        "bf16": "bfloat16",
        "fp16": "float16",
    }.get(dtype, dtype)
    try:
        np.dtype(value)
    except TypeError as exc:
        raise ValueError(f"unsupported logits dtype: {dtype!r}") from exc
    return value
