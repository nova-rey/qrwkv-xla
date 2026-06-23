from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from qrwkv_xla.artifacts import (
    load_fingerprint_exemplars,
    load_fingerprint_targets,
    validate_fingerprint_artifact,
)
from qrwkv_xla.artifacts._json import write_json
from qrwkv_xla.fingerprint import write_fingerprint_provenance
from qrwkv_xla.fingerprint.budgeted_artifact import (
    BudgetedArtifactConfig,
    budget_subset_cache_key,
    materialize_budgeted_artifact,
    validate_budgeted_artifact,
)

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "behavioral_fingerprint"
    / "v0_1_with_exemplars_tiny"
)


@pytest.mark.parametrize(
    ("role", "budget"),
    [("corridor_subset", 600), ("exemplar_subset", 600)],
)
def test_budgeted_subset_is_valid_and_loadable(
    tmp_path: Path, role: str, budget: int
) -> None:
    source, texts = _source_artifact(tmp_path)
    result = materialize_budgeted_artifact(
        BudgetedArtifactConfig(source, texts, tmp_path / role, role, budget)
    )
    assert validate_fingerprint_artifact(result.output_dir).ok
    assert tuple(load_fingerprint_targets(result.output_dir).iter_records())
    if role == "exemplar_subset":
        assert tuple(load_fingerprint_exemplars(result.output_dir).iter_records())
    accounting = _json(result.accounting_path)
    assert accounting["arm_charged_bytes"] <= budget
    assert accounting["budget_ceiling_respected"] is True
    assert accounting["shared_metadata_bytes"] > 0
    assert accounting["physical_subset_bytes"] > accounting["arm_charged_bytes"]


def test_selection_is_deterministic_and_valid_cache_is_reused(tmp_path: Path) -> None:
    source, texts = _source_artifact(tmp_path)
    config = BudgetedArtifactConfig(
        source, texts, tmp_path / "subset", "combined_two_cycle_subset", 1200
    )
    first = materialize_budgeted_artifact(config)
    first_receipt = _json(first.manifest_path)
    second = materialize_budgeted_artifact(config)
    assert second.cache_reused is True
    assert _json(second.manifest_path) == first_receipt


def test_different_ceilings_change_cache_and_subset_hashes(tmp_path: Path) -> None:
    source, texts = _source_artifact(tmp_path)
    small = BudgetedArtifactConfig(
        source, texts, tmp_path / "small", "corridor_subset", 600
    )
    large = replace(small, output_dir=tmp_path / "large", declared_byte_budget=1200)
    small_result = materialize_budgeted_artifact(small)
    large_result = materialize_budgeted_artifact(large)
    assert budget_subset_cache_key(small) != budget_subset_cache_key(large)
    assert (
        _json(small_result.manifest_path)["subset_manifest_sha256"]
        != _json(large_result.manifest_path)["subset_manifest_sha256"]
    )


def test_two_cycle_allocation_and_unused_remainder_are_reported(
    tmp_path: Path,
) -> None:
    source, texts = _source_artifact(tmp_path)
    result = materialize_budgeted_artifact(
        BudgetedArtifactConfig(
            source,
            texts,
            tmp_path / "combined",
            "combined_two_cycle_subset",
            1201,
            corridor_byte_fraction=0.4,
        )
    )
    receipt = _json(result.manifest_path)
    accounting = _json(result.accounting_path)
    assert receipt["corridor_byte_budget"] + receipt["exemplar_byte_budget"] == 1201
    assert accounting["corridor_charged_bytes"] <= receipt["corridor_byte_budget"]
    assert accounting["exemplar_charged_bytes"] <= receipt["exemplar_byte_budget"]
    assert accounting["unused_budget_bytes"] == 1201 - accounting["arm_charged_bytes"]


def test_corrupted_subset_shard_invalidates_cache(tmp_path: Path) -> None:
    source, texts = _source_artifact(tmp_path)
    config = BudgetedArtifactConfig(
        source, texts, tmp_path / "subset", "corridor_subset", 600
    )
    result = materialize_budgeted_artifact(config)
    shard = result.output_dir / "targets" / "targets-00000.jsonl"
    shard.write_text(shard.read_text() + "{}\n")
    assert validate_budgeted_artifact(result.output_dir)["valid"] is False
    with pytest.raises(ValueError, match="cache invalid"):
        materialize_budgeted_artifact(config)


def test_changed_allocation_policy_changes_cache_key(tmp_path: Path) -> None:
    source, texts = _source_artifact(tmp_path)
    config = BudgetedArtifactConfig(
        source, texts, tmp_path / "subset", "combined_two_cycle_subset", 1200
    )
    assert budget_subset_cache_key(config) != budget_subset_cache_key(
        replace(config, corridor_byte_fraction=0.6)
    )


def _source_artifact(tmp_path: Path) -> tuple[Path, Path]:
    artifact = tmp_path / "source-artifact"
    shutil.copytree(FIXTURE, artifact)
    targets = [
        json.loads(line)
        for line in (artifact / "targets" / "targets-00000.jsonl")
        .read_text()
        .splitlines()
        if line
    ]
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps({"example_id": row["example_id"], "text": f"text {index}"})
            + "\n"
            for index, row in enumerate(targets)
        )
    )
    manifest = _json(artifact / "manifest.json")
    manifest["created_by"] = "p156-1-test"
    write_json(artifact / "manifest.json", manifest)
    write_fingerprint_provenance(artifact, source_file=source, artifact_role="training")
    return artifact, source


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
