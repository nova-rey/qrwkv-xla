from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from qrwkv_xla.artifacts._json import (
    int_value,
    read_json_object,
    require_fields,
    write_json,
)
from qrwkv_xla.targets.store import TeacherTargetStore

TEACHER_TEXTBOOK_VERSION = 0

TEACHER_MANIFEST_FIELDS = (
    "artifact_type",
    "artifact_version",
    "teacher_model_id",
    "teacher_backend_type",
    "tokenizer_id",
    "vocab_size",
    "vocab_contract_path",
    "target_type",
    "dtype",
    "sequence_length",
    "num_examples",
    "shard_count",
    "created_at",
    "local_files_only",
    "allow_downloads",
    "claims_not_made",
)

EMISSION_CONFIG_FIELDS = (
    "dataset_source",
    "max_examples",
    "batch_size",
    "sequence_length",
    "logits_dtype",
    "include_hidden_states",
    "sampling_used",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "teacher_mode",
)


@dataclass(frozen=True)
class TeacherTextbookValidationReport:
    artifact_type: str = "teacher_textbook"
    artifact_version: int = TEACHER_TEXTBOOK_VERSION
    status: str = "fail"
    checks: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata_ok: bool = False
    vocab_contract_ok: bool = False
    manifest_ok: bool = False
    emission_config_ok: bool = False
    validation_report_ok: bool = False
    shards_ok: bool = False
    shape_ok: bool = False
    dtype_ok: bool = False
    count_ok: bool = False
    target_type: str | None = None
    top_k: int | None = None
    compressed_target_ok: bool | None = None
    mass_ok: bool | None = None
    sort_ok: bool | None = None
    duplicate_ok: bool | None = None
    claims_not_made: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_teacher_textbook(path: str | Path) -> TeacherTextbookValidationReport:
    root = Path(path)
    checks: list[str] = []
    blockers: list[str] = []
    warnings: list[str] = []
    metadata_ok = False
    vocab_contract_ok = False
    manifest_ok = False
    emission_config_ok = False
    validation_report_ok = False
    shards_ok = False
    shape_ok = False
    dtype_ok = False
    count_ok = False
    target_type: str | None = None
    top_k: int | None = None
    compressed_target_ok: bool | None = None
    mass_ok: bool | None = None
    sort_ok: bool | None = None
    duplicate_ok: bool | None = None
    claims_not_made: tuple[str, ...] = ()

    if not root.is_dir():
        blockers.append(f"teacher textbook path is not a directory: {root}")
        return _report(blockers=blockers)

    required_files = (
        "metadata.json",
        "vocab_contract.json",
        "teacher_manifest.json",
        "emission_config.json",
    )
    for name in required_files:
        if (root / name).is_file():
            checks.append(f"{name}: present")
        else:
            blockers.append(f"missing required file: {name}")
    if not (root / "validation_report.json").is_file():
        warnings.append(
            "validation_report.json missing; this validation can generate it"
        )
    else:
        validation_report_ok = True
        checks.append("validation_report.json: present")
    if not (root / "shards").is_dir():
        blockers.append("missing required directory: shards")

    metadata = None
    try:
        store = TeacherTargetStore.open(root)
        metadata = store.metadata
        target_type = metadata.target_type
        if metadata.target_type == "topk_with_tail_v0":
            top_k = int(metadata.target_params.get("top_k", "0"))
        store.validate()
        if metadata.target_type == "topk_with_tail_v0":
            compressed_target_ok = True
            mass_ok = True
            sort_ok = True
            duplicate_ok = True
        metadata_ok = True
        shards_ok = True
        shape_ok = True
        dtype_ok = True
        count_ok = True
        checks.append("TeacherTargetStore: valid")
    except ValueError as exc:
        if metadata is not None and metadata.target_type == "topk_with_tail_v0":
            compressed_target_ok = False
            mass_ok = False
            sort_ok = False
            duplicate_ok = False
        blockers.append(f"TeacherTargetStore validation failed: {exc}")

    vocab_contract: dict[str, Any] | None = None
    try:
        vocab_contract = read_json_object(root / "vocab_contract.json")
        vocab_contract_ok = True
        checks.append("vocab_contract.json: valid JSON object")
    except (OSError, ValueError) as exc:
        blockers.append(f"vocab_contract.json invalid: {exc}")

    try:
        manifest = read_json_object(root / "teacher_manifest.json")
        manifest_blockers = require_fields(
            manifest,
            TEACHER_MANIFEST_FIELDS,
            source="teacher_manifest.json",
        )
        blockers.extend(manifest_blockers)
        manifest_ok = not manifest_blockers
        claims = manifest.get("claims_not_made", ())
        if isinstance(claims, list):
            claims_not_made = tuple(str(item) for item in claims)
        _validate_manifest_matches_metadata(manifest, metadata, blockers)
        _validate_manifest_matches_vocab(manifest, vocab_contract, blockers)
        if manifest_ok:
            checks.append("teacher_manifest.json: required fields present")
    except (OSError, ValueError) as exc:
        blockers.append(f"teacher_manifest.json invalid: {exc}")

    try:
        emission_config = read_json_object(root / "emission_config.json")
        emission_blockers = require_fields(
            emission_config,
            EMISSION_CONFIG_FIELDS,
            source="emission_config.json",
        )
        blockers.extend(emission_blockers)
        emission_config_ok = not emission_blockers
        if emission_config_ok:
            checks.append("emission_config.json: required fields present")
    except (OSError, ValueError) as exc:
        blockers.append(f"emission_config.json invalid: {exc}")

    return _report(
        checks=checks,
        blockers=blockers,
        warnings=warnings,
        metadata_ok=metadata_ok,
        vocab_contract_ok=vocab_contract_ok,
        manifest_ok=manifest_ok,
        emission_config_ok=emission_config_ok,
        validation_report_ok=validation_report_ok,
        shards_ok=shards_ok,
        shape_ok=shape_ok,
        dtype_ok=dtype_ok,
        count_ok=count_ok,
        target_type=target_type,
        top_k=top_k,
        compressed_target_ok=compressed_target_ok,
        mass_ok=mass_ok,
        sort_ok=sort_ok,
        duplicate_ok=duplicate_ok,
        claims_not_made=claims_not_made,
    )


def write_teacher_textbook_validation_report(
    report: TeacherTextbookValidationReport,
    path: str | Path,
) -> None:
    write_json(Path(path), report.to_dict())


def _validate_manifest_matches_metadata(
    manifest: dict[str, Any],
    metadata: Any,
    blockers: list[str],
) -> None:
    if metadata is None:
        return
    expected = {
        "teacher_model_id": metadata.model_id,
        "tokenizer_id": metadata.tokenizer_id,
        "vocab_size": metadata.vocab_size,
        "target_type": metadata.target_type,
        "dtype": metadata.dtype,
        "sequence_length": metadata.sequence_length,
        "num_examples": metadata.num_examples,
        "shard_count": metadata.shard_count,
    }
    for key, value in expected.items():
        actual = (
            int_value(manifest, key) if isinstance(value, int) else manifest.get(key)
        )
        if actual != value:
            blockers.append(
                f"teacher_manifest.json {key} mismatch: "
                f"expected {value!r}, got {actual!r}"
            )


def _validate_manifest_matches_vocab(
    manifest: dict[str, Any],
    vocab_contract: dict[str, Any] | None,
    blockers: list[str],
) -> None:
    if vocab_contract is None:
        return
    for key in ("tokenizer_id", "vocab_size"):
        expected = (
            int_value(vocab_contract, key)
            if key == "vocab_size"
            else vocab_contract.get(key)
        )
        actual = int_value(manifest, key) if key == "vocab_size" else manifest.get(key)
        if expected is not None and actual != expected:
            blockers.append(
                f"teacher_manifest.json {key} does not match vocab_contract.json: "
                f"expected {expected!r}, got {actual!r}"
            )


def _report(
    *,
    checks: list[str] | None = None,
    blockers: list[str] | None = None,
    warnings: list[str] | None = None,
    metadata_ok: bool = False,
    vocab_contract_ok: bool = False,
    manifest_ok: bool = False,
    emission_config_ok: bool = False,
    validation_report_ok: bool = False,
    shards_ok: bool = False,
    shape_ok: bool = False,
    dtype_ok: bool = False,
    count_ok: bool = False,
    target_type: str | None = None,
    top_k: int | None = None,
    compressed_target_ok: bool | None = None,
    mass_ok: bool | None = None,
    sort_ok: bool | None = None,
    duplicate_ok: bool | None = None,
    claims_not_made: tuple[str, ...] = (),
) -> TeacherTextbookValidationReport:
    blocker_tuple = tuple(blockers or ())
    return TeacherTextbookValidationReport(
        status="fail" if blocker_tuple else "pass",
        checks=tuple(checks or ()),
        blockers=blocker_tuple,
        warnings=tuple(warnings or ()),
        metadata_ok=metadata_ok,
        vocab_contract_ok=vocab_contract_ok,
        manifest_ok=manifest_ok,
        emission_config_ok=emission_config_ok,
        validation_report_ok=validation_report_ok,
        shards_ok=shards_ok,
        shape_ok=shape_ok,
        dtype_ok=dtype_ok,
        count_ok=count_ok,
        target_type=target_type,
        top_k=top_k,
        compressed_target_ok=compressed_target_ok,
        mass_ok=mass_ok,
        sort_ok=sort_ok,
        duplicate_ok=duplicate_ok,
        claims_not_made=claims_not_made,
    )
