from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from qrwkv_xla.artifacts._json import read_json_object, require_fields, write_json
from qrwkv_xla.artifacts.teacher_textbook import validate_teacher_textbook

STUDENT_ARTIFACT_VERSION = 0

STUDENT_CONFIG_FIELDS = (
    "artifact_type",
    "artifact_version",
    "architecture_id",
    "student_family",
    "vocab_size",
    "vocab_contract_path",
    "runtime",
    "reference_runtime_default",
    "pallas_opt_in",
    "target_type",
    "checkpoint_format",
    "forward_input_shape",
    "forward_output_shape",
    "created_from_teacher_textbook",
    "claims_not_made",
)

RUNTIME_METADATA_FIELDS = (
    "runtime",
    "pallas_enabled",
    "reference_runtime_default",
)


@dataclass(frozen=True)
class StudentArtifactValidationReport:
    artifact_type: str = "student_artifact"
    artifact_version: int = STUDENT_ARTIFACT_VERSION
    status: str = "fail"
    checks: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    student_config_ok: bool = False
    vocab_contract_ok: bool = False
    checkpoint_ok: bool = False
    burn_report_ok: bool = False
    eval_report_ok: bool = False
    export_report_ok: bool = False
    runtime_metadata_ok: bool = False
    validation_report_ok: bool = False
    architecture_ok: bool = False
    vocab_matches_teacher_textbook: bool | None = None
    pallas_not_default: bool = False
    claims_not_made: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_student_artifact(
    path: str | Path,
    *,
    teacher_textbook_path: str | Path | None = None,
    expected_architecture_id: str = "current_qrwkv",
) -> StudentArtifactValidationReport:
    root = Path(path)
    checks: list[str] = []
    blockers: list[str] = []
    warnings: list[str] = []
    student_config_ok = False
    vocab_contract_ok = False
    checkpoint_ok = False
    burn_report_ok = False
    eval_report_ok = False
    export_report_ok = False
    runtime_metadata_ok = False
    validation_report_ok = False
    architecture_ok = False
    vocab_matches_teacher_textbook: bool | None = None
    pallas_not_default = False
    claims_not_made: tuple[str, ...] = ()

    if not root.is_dir():
        blockers.append(f"student artifact path is not a directory: {root}")
        return _report(blockers=blockers)

    for name in ("student_config.json", "vocab_contract.json", "runtime_metadata.json"):
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

    for name, attr in (
        ("burn_report.json", "burn_report_ok"),
        ("eval_report.json", "eval_report_ok"),
        ("export_report.json", "export_report_ok"),
    ):
        if (root / name).is_file():
            checks.append(f"{name}: present")
            if attr == "burn_report_ok":
                burn_report_ok = True
            elif attr == "eval_report_ok":
                eval_report_ok = True
            else:
                export_report_ok = True
        else:
            blockers.append(f"missing required file: {name}")

    checkpoint_ok = _checkpoint_exists(root)
    if checkpoint_ok:
        checks.append("checkpoint or params: present")
    else:
        blockers.append("missing checkpoint or params file")

    student_config: dict[str, Any] | None = None
    try:
        student_config = read_json_object(root / "student_config.json")
        config_blockers = require_fields(
            student_config,
            STUDENT_CONFIG_FIELDS,
            source="student_config.json",
        )
        blockers.extend(config_blockers)
        student_config_ok = not config_blockers
        architecture_ok = (
            student_config.get("architecture_id") == expected_architecture_id
        )
        if not architecture_ok:
            blockers.append(
                "student_config.json architecture_id mismatch: "
                f"expected {expected_architecture_id!r}, "
                f"got {student_config.get('architecture_id')!r}"
            )
        pallas_not_default = (
            student_config.get("runtime") == "reference"
            and student_config.get("reference_runtime_default") is True
            and student_config.get("pallas_opt_in") is True
        )
        if not pallas_not_default:
            blockers.append(
                "student_config.json does not preserve Pallas opt-in policy"
            )
        claims = student_config.get("claims_not_made", ())
        if isinstance(claims, list):
            claims_not_made = tuple(str(item) for item in claims)
        if student_config_ok:
            checks.append("student_config.json: required fields present")
    except (OSError, ValueError) as exc:
        blockers.append(f"student_config.json invalid: {exc}")

    vocab_contract: dict[str, Any] | None = None
    try:
        vocab_contract = read_json_object(root / "vocab_contract.json")
        vocab_contract_ok = True
        checks.append("vocab_contract.json: valid JSON object")
    except (OSError, ValueError) as exc:
        blockers.append(f"vocab_contract.json invalid: {exc}")

    try:
        runtime_metadata = read_json_object(root / "runtime_metadata.json")
        runtime_blockers = require_fields(
            runtime_metadata,
            RUNTIME_METADATA_FIELDS,
            source="runtime_metadata.json",
        )
        blockers.extend(runtime_blockers)
        runtime_metadata_ok = not runtime_blockers
        if runtime_metadata.get("pallas_enabled") is True:
            blockers.append(
                "runtime_metadata.json marks Pallas enabled for default artifact"
            )
        if runtime_metadata_ok:
            checks.append("runtime_metadata.json: required fields present")
    except (OSError, ValueError) as exc:
        blockers.append(f"runtime_metadata.json invalid: {exc}")

    if teacher_textbook_path is not None:
        teacher_report = validate_teacher_textbook(teacher_textbook_path)
        if teacher_report.status != "pass":
            blockers.append("teacher_textbook validation failed; cannot compare vocab")
            vocab_matches_teacher_textbook = False
        else:
            try:
                teacher_vocab = read_json_object(
                    Path(teacher_textbook_path) / "vocab_contract.json"
                )
                vocab_matches_teacher_textbook = _vocab_contracts_match(
                    vocab_contract,
                    teacher_vocab,
                )
                if not vocab_matches_teacher_textbook:
                    blockers.append(
                        "student vocab_contract.json does not match TeacherTextbook"
                    )
                else:
                    checks.append("student vocab contract matches TeacherTextbook")
            except (OSError, ValueError) as exc:
                blockers.append(f"could not compare teacher vocab contract: {exc}")
                vocab_matches_teacher_textbook = False

    return _report(
        checks=checks,
        blockers=blockers,
        warnings=warnings,
        student_config_ok=student_config_ok,
        vocab_contract_ok=vocab_contract_ok,
        checkpoint_ok=checkpoint_ok,
        burn_report_ok=burn_report_ok,
        eval_report_ok=eval_report_ok,
        export_report_ok=export_report_ok,
        runtime_metadata_ok=runtime_metadata_ok,
        validation_report_ok=validation_report_ok,
        architecture_ok=architecture_ok,
        vocab_matches_teacher_textbook=vocab_matches_teacher_textbook,
        pallas_not_default=pallas_not_default,
        claims_not_made=claims_not_made,
    )


def write_student_artifact_validation_report(
    report: StudentArtifactValidationReport,
    path: str | Path,
) -> None:
    write_json(Path(path), report.to_dict())


def _checkpoint_exists(root: Path) -> bool:
    return any(
        path.is_file()
        for path in (
            root / "params.npz",
            root / "model.safetensors",
            root / "checkpoint" / "params.npz",
            root / "checkpoint" / "checkpoint.json",
        )
    )


def _vocab_contracts_match(
    student_vocab: dict[str, Any] | None,
    teacher_vocab: dict[str, Any],
) -> bool:
    if student_vocab is None:
        return False
    keys = ("tokenizer_id", "vocab_size", "tokenizer_hash")
    return all(student_vocab.get(key) == teacher_vocab.get(key) for key in keys)


def _report(
    *,
    checks: list[str] | None = None,
    blockers: list[str] | None = None,
    warnings: list[str] | None = None,
    student_config_ok: bool = False,
    vocab_contract_ok: bool = False,
    checkpoint_ok: bool = False,
    burn_report_ok: bool = False,
    eval_report_ok: bool = False,
    export_report_ok: bool = False,
    runtime_metadata_ok: bool = False,
    validation_report_ok: bool = False,
    architecture_ok: bool = False,
    vocab_matches_teacher_textbook: bool | None = None,
    pallas_not_default: bool = False,
    claims_not_made: tuple[str, ...] = (),
) -> StudentArtifactValidationReport:
    blocker_tuple = tuple(blockers or ())
    return StudentArtifactValidationReport(
        status="fail" if blocker_tuple else "pass",
        checks=tuple(checks or ()),
        blockers=blocker_tuple,
        warnings=tuple(warnings or ()),
        student_config_ok=student_config_ok,
        vocab_contract_ok=vocab_contract_ok,
        checkpoint_ok=checkpoint_ok,
        burn_report_ok=burn_report_ok,
        eval_report_ok=eval_report_ok,
        export_report_ok=export_report_ok,
        runtime_metadata_ok=runtime_metadata_ok,
        validation_report_ok=validation_report_ok,
        architecture_ok=architecture_ok,
        vocab_matches_teacher_textbook=vocab_matches_teacher_textbook,
        pallas_not_default=pallas_not_default,
        claims_not_made=claims_not_made,
    )
