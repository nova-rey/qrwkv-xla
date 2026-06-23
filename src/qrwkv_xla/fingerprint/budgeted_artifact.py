from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qrwkv_xla.artifacts import validate_fingerprint_artifact
from qrwkv_xla.artifacts._json import read_json_object, write_json
from qrwkv_xla.fingerprint.held_out_evaluation import write_fingerprint_provenance
from qrwkv_xla.fingerprint.provenance import file_sha256, stable_hash

SELECTION_POLICY = "canonical_prefix_v1"
SUBSET_SCHEMA_VERSION = "qrwkv_xla.budget_subset.v1"
SUBSET_ROLES = {"corridor_subset", "exemplar_subset", "combined_two_cycle_subset"}


@dataclass(frozen=True)
class BudgetedArtifactConfig:
    source_artifact: Path
    source_texts: Path
    output_dir: Path
    subset_role: str
    declared_byte_budget: int
    selection_seed: int = 0
    corridor_byte_fraction: float = 0.5
    overwrite: bool = False


@dataclass(frozen=True)
class BudgetedArtifactResult:
    output_dir: Path
    manifest_path: Path
    accounting_path: Path
    selection_receipt_path: Path
    cache_reused: bool


def materialize_budgeted_artifact(
    config: BudgetedArtifactConfig,
) -> BudgetedArtifactResult:
    _validate_config(config)
    cache_key = budget_subset_cache_key(config)
    manifest_path = config.output_dir / "budget_subset_manifest.json"
    if config.output_dir.exists() and not config.overwrite:
        if _valid_cached_subset(config.output_dir, cache_key=cache_key):
            return _result(config.output_dir, cache_reused=True)
        raise ValueError("budget subset cache invalid or incompatible")
    if config.output_dir.exists():
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True)

    source_manifest = read_json_object(config.source_artifact / "manifest.json")
    source_provenance = read_json_object(
        config.source_artifact / "fingerprint_provenance.json"
    )
    target_rows = _read_manifest_rows(
        config.source_artifact, source_manifest, "targets"
    )
    exemplar_rows = _read_manifest_rows(
        config.source_artifact, source_manifest, "exemplars"
    )
    corridor_budget, exemplar_budget = _payload_allocations(config)
    selected_targets = _canonical_prefix(target_rows, corridor_budget)
    selected_exemplars = _canonical_prefix(exemplar_rows, exemplar_budget)

    if config.subset_role == "corridor_subset":
        selected_exemplars = []
    elif config.subset_role == "exemplar_subset":
        selected_targets = _target_scaffolding(target_rows, selected_exemplars)
    else:
        allowed = {_payload_key(row) for row in selected_targets}
        selected_exemplars = [
            row for row in selected_exemplars if _payload_key(row) in allowed
        ]
    if not selected_targets:
        raise ValueError("byte ceiling cannot materialize one required target record")
    if config.subset_role != "corridor_subset" and not selected_exemplars:
        raise ValueError("byte ceiling cannot materialize one required exemplar record")

    charged_target_bytes = (
        _rows_size(selected_targets) if config.subset_role != "exemplar_subset" else 0
    )
    charged_exemplar_bytes = _rows_size(selected_exemplars)
    charged_bytes = charged_target_bytes + charged_exemplar_bytes
    if charged_bytes > config.declared_byte_budget:
        raise ValueError("selected payload exceeds declared byte budget")

    target_path = config.output_dir / "targets" / "targets-00000.jsonl"
    exemplar_path = config.output_dir / "exemplars" / "exemplars-00000.jsonl"
    target_path.parent.mkdir()
    _write_rows(target_path, selected_targets)
    if selected_exemplars:
        exemplar_path.parent.mkdir()
        _write_rows(exemplar_path, selected_exemplars)
    shutil.copy2(
        config.source_artifact / source_manifest["modes_file"], config.output_dir
    )

    subset_manifest = dict(source_manifest)
    subset_manifest["created_by"] = "p156_1_budget_subset_materializer"
    subset_manifest["sequence"] = {
        **source_manifest["sequence"],
        "target_positions": len(selected_targets),
    }
    subset_manifest["target_shards"] = [
        {"path": "targets/targets-00000.jsonl", "num_records": len(selected_targets)}
    ]
    if selected_exemplars:
        subset_manifest["exemplar_reservoir"] = {
            **source_manifest.get("exemplar_reservoir", {}),
            "enabled": True,
            "num_records": len(selected_exemplars),
            "shards": [
                {
                    "path": "exemplars/exemplars-00000.jsonl",
                    "num_records": len(selected_exemplars),
                }
            ],
        }
    else:
        subset_manifest.pop("exemplar_reservoir", None)
    write_json(config.output_dir / "manifest.json", subset_manifest)
    source_binding = {
        "source_artifact": str(config.source_artifact),
        "source_artifact_sha256": file_sha256(config.source_artifact / "manifest.json"),
        "source_capture_config_sha256": source_provenance["capture_config_sha256"],
        "source_teacher_identity_sha256": source_provenance["teacher_identity_sha256"],
    }
    write_json(config.output_dir / "budget_subset_source.json", source_binding)
    write_fingerprint_provenance(
        config.output_dir,
        source_file=config.source_texts,
        artifact_role="training",
    )
    validation = validate_fingerprint_artifact(config.output_dir)
    if not validation.ok:
        raise ValueError(
            "materialized subset is invalid: " + "; ".join(validation.blockers)
        )

    payload_bytes = _rows_size(selected_targets) + _rows_size(selected_exemplars)
    shared_metadata_bytes = sum(
        path.stat().st_size
        for path in (
            config.output_dir / "manifest.json",
            config.output_dir / source_manifest["modes_file"],
            config.output_dir / "fingerprint_provenance.json",
            config.output_dir / "budget_subset_source.json",
        )
    )
    physical_bytes = _directory_size(config.output_dir)
    ordered_ids = [f"target:{_record_key(row)}" for row in selected_targets] + [
        f"exemplar:{_record_key(row)}" for row in selected_exemplars
    ]
    accounting = {
        "byte_accounting_policy": "arm_charged_logical_payload_bytes_v1",
        "declared_byte_budget": config.declared_byte_budget,
        "physical_subset_bytes": physical_bytes,
        "logical_payload_bytes_selected": payload_bytes,
        "logical_payload_bytes_consumed": None,
        "required_uncharged_scaffolding_bytes": payload_bytes - charged_bytes,
        "shared_metadata_bytes": shared_metadata_bytes,
        "arm_charged_bytes": charged_bytes,
        "corridor_charged_bytes": charged_target_bytes,
        "exemplar_charged_bytes": charged_exemplar_bytes,
        "unused_budget_bytes": config.declared_byte_budget - charged_bytes,
        "budget_ceiling_respected": charged_bytes <= config.declared_byte_budget,
        "physical_file_counted_once": True,
    }
    accounting_path = config.output_dir / "artifact_byte_accounting.json"
    write_json(accounting_path, accounting)
    selection = {
        "selection_policy": SELECTION_POLICY,
        "selection_seed": config.selection_seed,
        "ordered_record_ids": ordered_ids,
        "ordered_record_ids_sha256": stable_hash(ordered_ids),
        "selected_target_count": len(selected_targets),
        "selected_exemplar_count": len(selected_exemplars),
    }
    selection_path = config.output_dir / "record_selection_receipt.json"
    write_json(selection_path, selection)
    shard_hashes = {
        str(path.relative_to(config.output_dir)): file_sha256(path)
        for path in sorted(config.output_dir.rglob("*.jsonl"))
    }
    receipt = {
        "schema_version": SUBSET_SCHEMA_VERSION,
        "cache_key": cache_key,
        **source_binding,
        "subset_role": config.subset_role,
        "declared_byte_budget": config.declared_byte_budget,
        "corridor_byte_fraction": config.corridor_byte_fraction,
        "corridor_byte_budget": corridor_budget,
        "exemplar_byte_budget": exemplar_budget,
        "selection_policy": SELECTION_POLICY,
        "selection_seed": config.selection_seed,
        "ordered_record_ids_sha256": selection["ordered_record_ids_sha256"],
        "selected_record_count": len(ordered_ids),
        "logical_payload_bytes_selected": payload_bytes,
        "arm_charged_bytes": charged_bytes,
        "physical_subset_bytes": physical_bytes,
        "budget_ceiling_respected": accounting["budget_ceiling_respected"],
        "allocation_policy_sha256": stable_hash(
            {
                "corridor_byte_fraction": config.corridor_byte_fraction,
                "corridor_byte_budget": corridor_budget,
                "exemplar_byte_budget": exemplar_budget,
            }
        ),
        "artifact_manifest_sha256": file_sha256(config.output_dir / "manifest.json"),
        "artifact_byte_accounting_sha256": file_sha256(accounting_path),
        "record_selection_receipt_sha256": file_sha256(selection_path),
        "shard_hashes": shard_hashes,
    }
    receipt["subset_manifest_sha256"] = stable_hash(receipt)
    write_json(manifest_path, receipt)
    return _result(config.output_dir, cache_reused=False)


def budget_subset_cache_key(config: BudgetedArtifactConfig) -> str:
    return stable_hash(
        {
            "schema_version": SUBSET_SCHEMA_VERSION,
            "source_artifact_sha256": file_sha256(
                config.source_artifact / "manifest.json"
            ),
            "subset_role": config.subset_role,
            "declared_byte_budget": config.declared_byte_budget,
            "corridor_byte_fraction": config.corridor_byte_fraction,
            "selection_policy": SELECTION_POLICY,
            "selection_seed": config.selection_seed,
        }
    )


def validate_budgeted_artifact(path: Path) -> dict[str, Any]:
    receipt = read_json_object(path / "budget_subset_manifest.json")
    expected_hash = receipt.pop("subset_manifest_sha256", None)
    blockers = []
    if stable_hash(receipt) != expected_hash:
        blockers.append("subset manifest hash mismatch")
    for relative, expected in receipt.get("shard_hashes", {}).items():
        shard = path / relative
        if not shard.is_file() or file_sha256(shard) != expected:
            blockers.append(f"subset shard hash mismatch: {relative}")
    validation = validate_fingerprint_artifact(path)
    blockers.extend(validation.blockers)
    accounting = read_json_object(path / "artifact_byte_accounting.json")
    if not accounting.get("budget_ceiling_respected"):
        blockers.append("budget ceiling not respected")
    return {"valid": not blockers, "blockers": blockers}


def _valid_cached_subset(path: Path, *, cache_key: str) -> bool:
    try:
        receipt = read_json_object(path / "budget_subset_manifest.json")
        return (
            receipt.get("cache_key") == cache_key
            and validate_budgeted_artifact(path)["valid"]
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _payload_allocations(config):
    if config.subset_role == "corridor_subset":
        return config.declared_byte_budget, 0
    if config.subset_role == "exemplar_subset":
        return 0, config.declared_byte_budget
    corridor = int(config.declared_byte_budget * config.corridor_byte_fraction)
    return corridor, config.declared_byte_budget - corridor


def _canonical_prefix(rows, ceiling):
    selected = []
    consumed = 0
    for row in sorted(rows, key=_canonical_record_key):
        size = len(_encode_row(row))
        if consumed + size > ceiling:
            continue
        selected.append(row)
        consumed += size
    return selected


def _target_scaffolding(targets, exemplars):
    keys = {_payload_key(row) for row in exemplars}
    return [
        row
        for row in sorted(targets, key=_canonical_record_key)
        if _payload_key(row) in keys
    ]


def _read_manifest_rows(root, manifest, kind):
    if kind == "targets":
        shards = manifest["target_shards"]
    else:
        shards = manifest.get("exemplar_reservoir", {}).get("shards", [])
    rows = []
    for shard in shards:
        for line in (root / shard["path"]).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _canonical_record_key(row):
    return (str(row["example_id"]), int(row["position"]), stable_hash(row))


def _record_key(row):
    return f"{row['example_id']}:{row['position']}"


def _payload_key(row):
    return (tuple(row["input_ids"]), int(row["position"]))


def _encode_row(row):
    return (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _rows_size(rows):
    return sum(len(_encode_row(row)) for row in rows)


def _write_rows(path, rows):
    path.write_bytes(b"".join(_encode_row(row) for row in rows))


def _directory_size(path):
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _result(path, *, cache_reused):
    return BudgetedArtifactResult(
        output_dir=path,
        manifest_path=path / "budget_subset_manifest.json",
        accounting_path=path / "artifact_byte_accounting.json",
        selection_receipt_path=path / "record_selection_receipt.json",
        cache_reused=cache_reused,
    )


def _validate_config(config):
    if config.subset_role not in SUBSET_ROLES:
        raise ValueError("unsupported subset_role")
    if config.declared_byte_budget < 1:
        raise ValueError("declared_byte_budget must be positive")
    if not 0.0 < config.corridor_byte_fraction < 1.0:
        raise ValueError("corridor_byte_fraction must be between zero and one")
