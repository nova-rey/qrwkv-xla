from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

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
HF_CLAIMS_NOT_MADE = (
    "no_model_quality_claim",
    "no_qwen_parity_claim",
    "no_training_claim",
    "no_remote_teacher_service_claim",
    "no_tokenizer_remapping_claim",
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
        return build_hf_teacher_textbook(config)
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


def build_hf_teacher_textbook(
    config: TeacherTextbookBuildConfig,
) -> TeacherTextbookValidationReport:
    _validate_config(config)
    if config.local_files_only and config.allow_downloads:
        raise ValueError("--local-files-only and --allow-downloads cannot both be set")
    examples = load_text_examples(config.dataset_path, max_examples=config.max_examples)
    if config.output_dir.exists():
        if not config.overwrite:
            raise ValueError(
                f"TeacherTextbook output already exists: {config.output_dir}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(config.output_dir)

    torch, auto_tokenizer, auto_model = _load_hf_dependencies(config)
    local_files_only = config.local_files_only or not config.allow_downloads
    try:
        tokenizer = auto_tokenizer.from_pretrained(
            config.teacher_model_id,
            local_files_only=local_files_only,
        )
        model = auto_model.from_pretrained(
            config.teacher_model_id,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise RuntimeError(_hf_error_message(config, local_files_only)) from exc

    _prepare_hf_tokenizer(tokenizer, config)
    if hasattr(model, "eval"):
        model.eval()

    created_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    vocab_size = _hf_vocab_size(tokenizer, config)
    shard_count = (len(examples) + config.batch_size - 1) // config.batch_size
    tokenizer_id = str(
        getattr(tokenizer, "name_or_path", None) or config.teacher_model_id
    )
    metadata = TargetStoreMetadata(
        schema_version=TEACHER_TARGET_STORE_SCHEMA_VERSION,
        target_store_version=TEACHER_TARGET_STORE_VERSION,
        model_id=config.teacher_model_id,
        model_family="hf-causal-lm",
        tokenizer_id=tokenizer_id,
        tokenizer_hash=None,
        vocab_size=vocab_size,
        target_type="full_logits",
        dtype=_canonical_dtype(config.logits_dtype),
        sequence_length=config.sequence_length,
        num_examples=len(examples),
        shard_count=shard_count,
        created_by="qrwkv_xla.artifacts.teacher_textbook_builder",
        created_at=created_at,
        source={"kind": _dataset_source(config.dataset_path)},
        provenance={"phase": "P116.2", "teacher_mode": "hf"},
    )
    store = TeacherTargetStore.create(config.output_dir, metadata, overwrite=True)
    inference_context = getattr(torch, "inference_mode", None) or torch.no_grad
    with inference_context():
        for shard_id, start in enumerate(range(0, len(examples), config.batch_size)):
            batch = examples[start : start + config.batch_size]
            arrays = _hf_arrays(
                batch,
                config,
                tokenizer=tokenizer,
                model=model,
                vocab_size=vocab_size,
            )
            store.write_shard(shard_id, arrays)

    _write_hf_sidecars(
        config,
        examples,
        created_at,
        tokenizer=tokenizer,
        tokenizer_id=tokenizer_id,
        vocab_size=vocab_size,
        local_files_only=local_files_only,
    )
    report = validate_teacher_textbook(config.output_dir)
    write_teacher_textbook_validation_report(
        report,
        config.output_dir / "validation_report.json",
    )
    if report.status != "pass":
        raise ValueError(
            "built HF TeacherTextbook failed validation: " + "; ".join(report.blockers)
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


def _write_hf_sidecars(
    config: TeacherTextbookBuildConfig,
    examples: tuple[TinyTextExample, ...],
    created_at: str,
    *,
    tokenizer: Any,
    tokenizer_id: str,
    vocab_size: int,
    local_files_only: bool,
) -> None:
    shard_count = (len(examples) + config.batch_size - 1) // config.batch_size
    special_tokens = {
        "pad_token_id": _optional_int(getattr(tokenizer, "pad_token_id", None)),
        "eos_token_id": _optional_int(getattr(tokenizer, "eos_token_id", None)),
        "bos_token_id": _optional_int(getattr(tokenizer, "bos_token_id", None)),
        "unk_token_id": _optional_int(getattr(tokenizer, "unk_token_id", None)),
    }
    write_json(
        config.output_dir / "vocab_contract.json",
        {
            "tokenizer_id": tokenizer_id,
            "tokenizer_hash": None,
            "tokenizer_family": "hf",
            "backend": "hf",
            "vocab_size": vocab_size,
            "model_id": config.teacher_model_id,
            "model_family": "hf-causal-lm",
            "special_tokens": special_tokens,
        },
    )
    write_json(
        config.output_dir / "teacher_manifest.json",
        {
            "artifact_type": "teacher_textbook",
            "artifact_version": 0,
            "teacher_model_id": config.teacher_model_id,
            "teacher_backend_type": "hf",
            "teacher_revision_or_hash": None,
            "tokenizer_id": tokenizer_id,
            "vocab_size": vocab_size,
            "vocab_contract_path": "vocab_contract.json",
            "target_type": "full_logits",
            "dtype": _canonical_dtype(config.logits_dtype),
            "sequence_length": config.sequence_length,
            "num_examples": len(examples),
            "shard_count": shard_count,
            "created_at": created_at,
            "local_files_only": local_files_only,
            "allow_downloads": config.allow_downloads and not local_files_only,
            "claims_not_made": list(HF_CLAIMS_NOT_MADE),
        },
    )
    write_json(
        config.output_dir / "emission_config.json",
        {
            "dataset_source": _dataset_source(config.dataset_path),
            "max_examples": config.max_examples,
            "batch_size": config.batch_size,
            "sequence_length": config.sequence_length,
            "logits_dtype": _canonical_dtype(config.logits_dtype),
            "include_hidden_states": False,
            "sampling_used": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "seed": config.seed,
            "teacher_mode": "hf",
            "teacher_model_id": config.teacher_model_id,
            "local_files_only": local_files_only,
            "allow_downloads": config.allow_downloads and not local_files_only,
        },
    )


def _hf_arrays(
    examples: tuple[TinyTextExample, ...],
    config: TeacherTextbookBuildConfig,
    *,
    tokenizer: Any,
    model: Any,
    vocab_size: int,
) -> dict[str, np.ndarray]:
    encoded = tokenizer(
        [example.text for example in examples],
        padding="max_length",
        truncation=True,
        max_length=config.sequence_length,
        return_tensors="pt",
    )
    output = model(**encoded)
    input_ids = _tensor_to_numpy(encoded["input_ids"]).astype(np.int32, copy=False)
    if "attention_mask" in encoded:
        attention_mask = _tensor_to_numpy(encoded["attention_mask"]).astype(
            np.int32,
            copy=False,
        )
    else:
        attention_mask = np.ones_like(input_ids, dtype=np.int32)
    logits = _tensor_to_numpy(output.logits).astype(
        np.dtype(_canonical_dtype(config.logits_dtype)),
        copy=False,
    )
    if logits.shape != (len(examples), config.sequence_length, vocab_size):
        raise ValueError(
            "teacher-mode=hf logits shape mismatch: "
            f"expected {(len(examples), config.sequence_length, vocab_size)}, "
            f"got {logits.shape}"
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "logits": logits,
    }


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _load_hf_dependencies(config: TeacherTextbookBuildConfig) -> tuple[Any, Any, Any]:
    try:
        torch = import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "teacher-mode=hf requires optional dependency torch. "
            f"teacher_model_id={config.teacher_model_id!r}; install teacher-hf "
            "dependencies, cache the model, or use --teacher-mode fake."
        ) from exc
    try:
        transformers = import_module("transformers")
    except ImportError as exc:
        raise RuntimeError(
            "teacher-mode=hf requires optional dependency transformers. "
            f"teacher_model_id={config.teacher_model_id!r}; install teacher-hf "
            "dependencies, cache the model, or use --teacher-mode fake."
        ) from exc
    return (
        torch,
        transformers.AutoTokenizer,
        transformers.AutoModelForCausalLM,
    )


def _prepare_hf_tokenizer(tokenizer: Any, config: TeacherTextbookBuildConfig) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            raise RuntimeError(
                "teacher-mode=hf tokenizer has no pad_token_id and no eos_token "
                f"fallback for teacher_model_id={config.teacher_model_id!r}"
            )
        tokenizer.pad_token = eos_token


def _hf_vocab_size(tokenizer: Any, config: TeacherTextbookBuildConfig) -> int:
    size = getattr(tokenizer, "vocab_size", None)
    if size is None:
        try:
            size = len(tokenizer)
        except TypeError as exc:
            raise RuntimeError(
                "teacher-mode=hf could not determine tokenizer vocab size for "
                f"teacher_model_id={config.teacher_model_id!r}"
            ) from exc
    value = int(size)
    if value <= 0:
        raise RuntimeError(
            "teacher-mode=hf tokenizer vocab size must be > 0 for "
            f"teacher_model_id={config.teacher_model_id!r}"
        )
    return value


def _hf_error_message(
    config: TeacherTextbookBuildConfig,
    local_files_only: bool,
) -> str:
    if local_files_only:
        fix = "cache model first or rerun with --allow-downloads"
    else:
        fix = "verify model id, network access, and optional teacher-hf dependencies"
    return (
        "teacher-mode=hf failed to load tokenizer/model; "
        f"teacher_model_id={config.teacher_model_id!r}; "
        f"local_files_only={local_files_only}; suggested fix: {fix}"
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


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
