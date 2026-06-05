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
DEFAULT_TARGET_TYPE = "dense_logits"
TOPK_TAIL_TARGET_TYPE = "topk_with_tail_v0"
CASCADED_TARGET_TYPE = "cascaded_soft_labels_v1"
DEFAULT_TOP_K = 256
DEFAULT_TOP_LOG_PROBS_DTYPE = "float16"
DEFAULT_BUCKET_EDGES = (1.0, 1e-3, 1e-6, 1e-9, 1e-12, 0.0)
DEFAULT_BUCKET_EDGE_TYPE = "probability"
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
TOPK_TAIL_CLAIMS_NOT_MADE = (
    "not_dense_logits",
    "not_full_distribution_exact",
    "not_trainer_consumable_until_p120",
    "not_model_quality_claim",
)
CASCADED_CLAIMS_NOT_MADE = (
    "not_dense_logits",
    "not_full_distribution_exact",
    "not_tail_membership_exact",
    "not_bucket_loss_consumable_until_p122",
    "not_cross_vocab",
    "not_model_quality_claim",
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
    target_type: str = DEFAULT_TARGET_TYPE
    top_k: int = DEFAULT_TOP_K
    top_log_probs_dtype: str = DEFAULT_TOP_LOG_PROBS_DTYPE
    bucket_edges: tuple[float, ...] = DEFAULT_BUCKET_EDGES
    bucket_edge_type: str = DEFAULT_BUCKET_EDGE_TYPE
    bucket_mass_dtype: str = "float32"
    bucket_mean_logp_dtype: str = "float32"


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
        target_type=config.target_type,
        dtype=_canonical_dtype(config.logits_dtype),
        sequence_length=config.sequence_length,
        num_examples=len(examples),
        shard_count=shard_count,
        created_by="qrwkv_xla.artifacts.teacher_textbook_builder",
        created_at=created_at,
        source={"kind": _dataset_source(config.dataset_path)},
        provenance={"phase": "P119", "teacher_mode": "fake"},
        target_params=_target_params(config),
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
    if _is_compressed_target(config.target_type) and config.top_k > vocab_size:
        raise ValueError("top_k must be <= tokenizer vocab_size")
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
        target_type=config.target_type,
        dtype=_canonical_dtype(config.logits_dtype),
        sequence_length=config.sequence_length,
        num_examples=len(examples),
        shard_count=shard_count,
        created_by="qrwkv_xla.artifacts.teacher_textbook_builder",
        created_at=created_at,
        source={"kind": _dataset_source(config.dataset_path)},
        provenance={"phase": "P119", "teacher_mode": "hf"},
        target_params=_target_params(config),
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
    if config.target_type not in {
        "dense_logits",
        TOPK_TAIL_TARGET_TYPE,
        CASCADED_TARGET_TYPE,
    }:
        raise ValueError(f"unsupported target_type: {config.target_type!r}")
    if config.top_k <= 0:
        raise ValueError("top_k must be > 0")
    if (
        _is_compressed_target(config.target_type)
        and config.teacher_mode == "fake"
        and config.top_k > config.vocab_size
    ):
        raise ValueError("top_k must be <= vocab_size")
    if config.target_type == CASCADED_TARGET_TYPE:
        _validate_bucket_edges(config.bucket_edges, config.bucket_edge_type)
    _canonical_dtype(config.logits_dtype)
    _canonical_dtype(config.top_log_probs_dtype)
    _canonical_dtype(config.bucket_mass_dtype)
    _canonical_dtype(config.bucket_mean_logp_dtype)


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
    dense = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "logits": logits,
    }
    if _is_compressed_target(config.target_type):
        return _compress_dense_arrays(
            dense, config=config, vocab_size=config.vocab_size
        )
    return dense


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
            "target_type": config.target_type,
            "dtype": _canonical_dtype(config.logits_dtype),
            "sequence_length": config.sequence_length,
            "num_examples": len(examples),
            "shard_count": shard_count,
            **_manifest_topk_fields(config),
            "created_at": created_at,
            "local_files_only": config.local_files_only,
            "allow_downloads": config.allow_downloads,
            "claims_not_made": _claims_not_made(config, teacher_mode="fake"),
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
            "target_type": config.target_type,
            "include_hidden_states": config.include_hidden_states,
            "sampling_used": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            **_emission_topk_fields(config),
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
            "target_type": config.target_type,
            "dtype": _canonical_dtype(config.logits_dtype),
            "sequence_length": config.sequence_length,
            "num_examples": len(examples),
            "shard_count": shard_count,
            **_manifest_topk_fields(config),
            "created_at": created_at,
            "local_files_only": local_files_only,
            "allow_downloads": config.allow_downloads and not local_files_only,
            "claims_not_made": _claims_not_made(config, teacher_mode="hf"),
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
            "target_type": config.target_type,
            "include_hidden_states": False,
            "sampling_used": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            **_emission_topk_fields(config),
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
    dense = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "logits": logits,
    }
    if _is_compressed_target(config.target_type):
        return _compress_dense_arrays(dense, config=config, vocab_size=vocab_size)
    return dense


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


def _target_params(config: TeacherTextbookBuildConfig) -> dict[str, str]:
    if not _is_compressed_target(config.target_type):
        return {}
    params = {
        "top_k": str(config.top_k),
        "top_log_probs_dtype": _canonical_dtype(config.top_log_probs_dtype),
        "top_token_ids_dtype": "int32",
        "top_mass_dtype": "float32",
        "tail_mass_dtype": "float32",
        "teacher_entropy_dtype": "float32",
    }
    if config.target_type == CASCADED_TARGET_TYPE:
        params.update(
            {
                "bucket_edges": _bucket_edges_string(config.bucket_edges),
                "bucket_edge_type": config.bucket_edge_type,
                "bucket_count": str(len(config.bucket_edges) - 1),
                "bucket_mass_dtype": _canonical_dtype(config.bucket_mass_dtype),
                "bucket_count_dtype": "int32",
                "bucket_mean_logp_dtype": _canonical_dtype(
                    config.bucket_mean_logp_dtype
                ),
                "empty_bucket_mean_logp_sentinel": "0.0",
            }
        )
    return params


def _manifest_topk_fields(config: TeacherTextbookBuildConfig) -> dict[str, object]:
    if not _is_compressed_target(config.target_type):
        return {}
    fields: dict[str, object] = {
        "top_k": config.top_k,
        "top_log_probs_dtype": _canonical_dtype(config.top_log_probs_dtype),
        "top_token_ids_dtype": "int32",
        "top_mass_dtype": "float32",
        "tail_mass_dtype": "float32",
        "teacher_entropy_dtype": "float32",
    }
    if config.target_type == CASCADED_TARGET_TYPE:
        fields.update(
            {
                "bucket_edges": list(config.bucket_edges),
                "bucket_edge_type": config.bucket_edge_type,
                "bucket_count": len(config.bucket_edges) - 1,
                "bucket_mass_dtype": _canonical_dtype(config.bucket_mass_dtype),
                "bucket_count_dtype": "int32",
                "bucket_mean_logp_dtype": _canonical_dtype(
                    config.bucket_mean_logp_dtype
                ),
            }
        )
    return fields


def _emission_topk_fields(config: TeacherTextbookBuildConfig) -> dict[str, object]:
    if not _is_compressed_target(config.target_type):
        return {}
    fields: dict[str, object] = {
        "top_k": config.top_k,
        "top_log_probs_dtype": _canonical_dtype(config.top_log_probs_dtype),
    }
    if config.target_type == CASCADED_TARGET_TYPE:
        fields.update(
            {
                "bucket_edges": list(config.bucket_edges),
                "bucket_edge_type": config.bucket_edge_type,
                "bucket_count": len(config.bucket_edges) - 1,
                "bucket_mass_dtype": _canonical_dtype(config.bucket_mass_dtype),
                "bucket_mean_logp_dtype": _canonical_dtype(
                    config.bucket_mean_logp_dtype
                ),
            }
        )
    return fields


def _claims_not_made(
    config: TeacherTextbookBuildConfig,
    *,
    teacher_mode: str,
) -> list[str]:
    base = HF_CLAIMS_NOT_MADE if teacher_mode == "hf" else CLAIMS_NOT_MADE
    claims = list(base)
    if config.target_type == TOPK_TAIL_TARGET_TYPE:
        claims.extend(TOPK_TAIL_CLAIMS_NOT_MADE)
    if config.target_type == CASCADED_TARGET_TYPE:
        claims.extend(CASCADED_CLAIMS_NOT_MADE)
    return claims


def _compress_dense_arrays(
    arrays: dict[str, np.ndarray],
    *,
    config: TeacherTextbookBuildConfig,
    vocab_size: int,
) -> dict[str, np.ndarray]:
    logits = np.asarray(arrays["logits"], dtype=np.float32)
    log_probs = _log_softmax(logits)
    top_token_ids = np.argsort(log_probs, axis=-1)[..., -config.top_k :][..., ::-1]
    top_log_probs = np.take_along_axis(log_probs, top_token_ids, axis=-1)
    top_probs = np.exp(top_log_probs).astype(np.float32)
    top_mass = np.sum(top_probs, axis=-1, dtype=np.float32)
    tail_mass = np.clip(1.0 - top_mass, 0.0, 1.0).astype(np.float32)
    probs = np.exp(log_probs).astype(np.float32)
    teacher_entropy = -np.sum(probs * log_probs, axis=-1, dtype=np.float32)
    compressed = {
        "input_ids": arrays["input_ids"],
        "attention_mask": arrays["attention_mask"],
        "top_token_ids": top_token_ids.astype(np.int32),
        "top_log_probs": top_log_probs.astype(
            np.dtype(_canonical_dtype(config.top_log_probs_dtype))
        ),
        "top_mass": top_mass.astype(np.float32),
        "tail_mass": tail_mass,
        "teacher_entropy": teacher_entropy.astype(np.float32),
    }
    if config.target_type == CASCADED_TARGET_TYPE:
        compressed.update(
            _bucket_tail_arrays(
                probs=probs,
                log_probs=log_probs,
                top_token_ids=top_token_ids,
                config=config,
            )
        )
    return compressed


def _bucket_tail_arrays(
    *,
    probs: np.ndarray,
    log_probs: np.ndarray,
    top_token_ids: np.ndarray,
    config: TeacherTextbookBuildConfig,
) -> dict[str, np.ndarray]:
    edges = np.asarray(config.bucket_edges, dtype=np.float32)
    bucket_count = len(edges) - 1
    top_mask = np.zeros(probs.shape, dtype=bool)
    np.put_along_axis(top_mask, top_token_ids, True, axis=-1)
    bucket_mass = np.zeros((*probs.shape[:2], bucket_count), dtype=np.float32)
    bucket_token_count = np.zeros((*probs.shape[:2], bucket_count), dtype=np.int32)
    bucket_mean_logp = np.zeros((*probs.shape[:2], bucket_count), dtype=np.float32)
    for bucket_id in range(bucket_count):
        upper = edges[bucket_id]
        lower = edges[bucket_id + 1]
        mask = (~top_mask) & (probs < upper) & (probs >= lower)
        mass = np.sum(np.where(mask, probs, 0.0), axis=-1, dtype=np.float32)
        count = np.sum(mask, axis=-1, dtype=np.int32)
        logp_sum = np.sum(np.where(mask, log_probs, 0.0), axis=-1, dtype=np.float32)
        mean_logp = np.divide(
            logp_sum,
            count,
            out=np.zeros_like(logp_sum, dtype=np.float32),
            where=count > 0,
        )
        bucket_mass[..., bucket_id] = mass
        bucket_token_count[..., bucket_id] = count
        bucket_mean_logp[..., bucket_id] = mean_logp
    return {
        "bucket_mass": bucket_mass.astype(
            np.dtype(_canonical_dtype(config.bucket_mass_dtype))
        ),
        "bucket_count": bucket_token_count.astype(np.int32),
        "bucket_mean_logp": bucket_mean_logp.astype(
            np.dtype(_canonical_dtype(config.bucket_mean_logp_dtype))
        ),
    }


def _is_compressed_target(target_type: str) -> bool:
    return target_type in {TOPK_TAIL_TARGET_TYPE, CASCADED_TARGET_TYPE}


def _validate_bucket_edges(edges: tuple[float, ...], edge_type: str) -> None:
    if edge_type != DEFAULT_BUCKET_EDGE_TYPE:
        raise ValueError("bucket_edge_type must be 'probability'")
    if len(edges) < 2:
        raise ValueError("bucket_edges must contain at least two edges")
    edge_array = np.asarray(edges, dtype=np.float64)
    if not np.all(np.isfinite(edge_array)):
        raise ValueError("bucket_edges must be finite")
    if not np.all(np.diff(edge_array) < 0):
        raise ValueError("bucket_edges must be strictly descending")
    if not np.isclose(edge_array[0], 1.0):
        raise ValueError("bucket_edges must start at 1.0")
    if not np.isclose(edge_array[-1], 0.0):
        raise ValueError("bucket_edges must end at 0.0")


def _bucket_edges_string(edges: tuple[float, ...]) -> str:
    return ",".join(f"{edge:.12g}" for edge in edges)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return shifted - logsumexp


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
