from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qrwkv_xla.artifacts._json import read_json_object
from qrwkv_xla.artifacts.fingerprint import (
    FingerprintManifest,
    validate_fingerprint_artifact,
)


@dataclass(frozen=True)
class FingerprintArtifactSummary:
    artifact_type: str
    artifact_version: str
    artifact_dir: str
    teacher_model_name: str
    tokenizer_name: str
    vocab_size: int
    max_seq_len: int
    tracked_stats: tuple[str, ...]
    num_modes: int
    num_corridor_records: int
    has_exemplars: bool
    exemplar_payload_type: str | None
    num_exemplar_records: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_fingerprint_artifact(path: str | Path) -> FingerprintArtifactSummary:
    artifact_dir = Path(path)
    validation = validate_fingerprint_artifact(artifact_dir)
    if not validation.ok:
        joined = "; ".join(validation.blockers)
        raise ValueError(f"fingerprint artifact validation failed: {joined}")

    manifest = FingerprintManifest.from_payload(
        read_json_object(artifact_dir / "manifest.json")
    )
    tracked = manifest.stats.get("tracked", ())
    return FingerprintArtifactSummary(
        artifact_type=manifest.artifact_type,
        artifact_version=manifest.artifact_version,
        artifact_dir=str(artifact_dir),
        teacher_model_name=str(manifest.teacher.get("model_name", "")),
        tokenizer_name=str(manifest.teacher.get("tokenizer_name", "")),
        vocab_size=_positive_int(manifest.teacher.get("vocab_size"), "vocab_size"),
        max_seq_len=_positive_int(manifest.sequence.get("max_seq_len"), "max_seq_len"),
        tracked_stats=tuple(str(stat) for stat in tracked),
        num_modes=int(validation.metadata.get("modes", 0)),
        num_corridor_records=int(validation.metadata.get("records", 0)),
        has_exemplars=bool(validation.metadata.get("exemplar_reservoir_enabled")),
        exemplar_payload_type=validation.metadata.get("exemplar_payload_type"),
        num_exemplar_records=int(validation.metadata.get("exemplar_records", 0)),
    )


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"manifest {name} must be a positive integer")
    return value
