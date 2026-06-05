from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from qrwkv_xla.artifacts import (
    TeacherTextbookBuildConfig,
    build_teacher_textbook,
    validate_teacher_textbook,
)
from qrwkv_xla.artifacts._json import read_json_object
from qrwkv_xla.eval import run_mini_eval_harness
from qrwkv_xla.students import TINY_DEBUG_ARCHITECTURE_ID
from qrwkv_xla.targets import TeacherTargetStore, load_teacher_target_batch


def test_fake_builder_writes_valid_cascaded_textbook_without_logits(
    tmp_path: Path,
) -> None:
    output = _build_cascaded(tmp_path)

    metadata = read_json_object(output / "metadata.json")
    manifest = read_json_object(output / "teacher_manifest.json")
    emission = read_json_object(output / "emission_config.json")
    report = read_json_object(output / "validation_report.json")

    assert metadata["target_type"] == "cascaded_soft_labels_v1"
    assert metadata["target_params"]["top_k"] == "8"
    assert metadata["target_params"]["bucket_edges"] == "1,0.001,1e-06,1e-09,1e-12,0"
    assert metadata["target_params"]["bucket_count"] == "5"
    assert manifest["target_type"] == "cascaded_soft_labels_v1"
    assert manifest["top_k"] == 8
    assert manifest["bucket_count"] == 5
    assert emission["bucket_edge_type"] == "probability"
    assert report["status"] == "pass"
    assert report["bucket_target_ok"] is True
    assert report["bucket_mass_ok"] is True
    assert report["bucket_count_ok"] is True

    with np.load(output / "shards" / "shard-00000.npz", allow_pickle=False) as shard:
        assert "logits" not in shard.files
        assert shard["top_token_ids"].shape == (2, 8, 8)
        assert shard["bucket_mass"].shape == (2, 8, 5)
        assert shard["bucket_count"].shape == (2, 8, 5)
        assert shard["bucket_mean_logp"].shape == (2, 8, 5)
        np.testing.assert_array_equal(np.sum(shard["bucket_count"], axis=-1), 56)
        np.testing.assert_allclose(
            np.sum(shard["bucket_mass"], axis=-1),
            shard["tail_mass"],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            shard["top_mass"] + np.sum(shard["bucket_mass"], axis=-1),
            1.0,
            atol=1e-5,
        )
        empty = shard["bucket_count"] == 0
        assert np.all(shard["bucket_mean_logp"][empty] == 0.0)
        assert np.all(shard["teacher_entropy"] >= 0.0)


def test_cascaded_loader_exposes_bucket_fields_without_logits(tmp_path: Path) -> None:
    store = TeacherTargetStore.open(_build_cascaded(tmp_path))

    batch = load_teacher_target_batch(store, shard_id=0)

    assert batch.target_type == "cascaded_soft_labels_v1"
    assert batch.teacher_logits is None
    assert batch.bucket_mass is not None
    assert batch.bucket_count is not None
    assert batch.bucket_mean_logp is not None
    assert batch.bucket_mass.shape == (2, 8, 5)


def test_cascaded_eval_consumes_topk_head_only_until_p122(tmp_path: Path) -> None:
    store = TeacherTargetStore.open(_build_cascaded(tmp_path))

    result = run_mini_eval_harness(
        store=store,
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
    )

    assert result.status == "pass"
    assert result.teacher_target_type == "cascaded_soft_labels_v1"
    assert result.distillation_loss_type == "topk_head_only_until_p122"
    assert result.bucket_loss_weight == 0.0
    assert result.head_loss is not None
    assert result.tail_loss == 0.0
    assert result.top_k == 8


def test_missing_bucket_mass_fails_validation(tmp_path: Path) -> None:
    output = _build_cascaded(tmp_path)

    _rewrite_shard(output, remove=("bucket_mass",))

    report = validate_teacher_textbook(output)
    assert report.status == "fail"
    assert report.bucket_target_ok is False
    assert _has_blocker(report.blockers, "missing required arrays")


def test_missing_bucket_count_fails_validation(tmp_path: Path) -> None:
    output = _build_cascaded(tmp_path)

    _rewrite_shard(output, remove=("bucket_count",))

    report = validate_teacher_textbook(output)
    assert report.status == "fail"
    assert _has_blocker(report.blockers, "missing required arrays")


def test_wrong_bucket_shape_fails_validation(tmp_path: Path) -> None:
    output = _build_cascaded(tmp_path)

    def mutate(arrays: dict[str, np.ndarray]) -> None:
        arrays["bucket_mass"] = arrays["bucket_mass"][:, :, :4]

    _rewrite_shard(output, mutate=mutate)

    report = validate_teacher_textbook(output)
    assert report.status == "fail"
    assert _has_blocker(report.blockers, "bucket_mass shape")


def test_bucket_edges_wrong_length_fails_validation(tmp_path: Path) -> None:
    output = _build_cascaded(tmp_path)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["target_params"]["bucket_edges"] = "1,0.001,0"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    report = validate_teacher_textbook(output)
    assert report.status == "fail"
    assert _has_blocker(report.blockers, "bucket_edges length")


def test_bucket_edges_unsorted_fails_validation(tmp_path: Path) -> None:
    output = _build_cascaded(tmp_path)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["target_params"]["bucket_edges"] = "1,1e-6,1e-3,1e-9,1e-12,0"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    report = validate_teacher_textbook(output)
    assert report.status == "fail"
    assert _has_blocker(report.blockers, "strictly descending")


def test_bucket_count_sum_mismatch_fails_validation(tmp_path: Path) -> None:
    output = _build_cascaded(tmp_path)

    def mutate(arrays: dict[str, np.ndarray]) -> None:
        arrays["bucket_count"][0, 0, 0] += 1

    _rewrite_shard(output, mutate=mutate)

    report = validate_teacher_textbook(output)
    assert report.status == "fail"
    assert _has_blocker(report.blockers, "sum(bucket_count)")


def test_bucket_mass_sum_mismatch_fails_validation(tmp_path: Path) -> None:
    output = _build_cascaded(tmp_path)

    def mutate(arrays: dict[str, np.ndarray]) -> None:
        arrays["bucket_mass"][0, 0, 0] += 0.5

    _rewrite_shard(output, mutate=mutate)

    report = validate_teacher_textbook(output)
    assert report.status == "fail"
    assert _has_blocker(report.blockers, "sum(bucket_mass)")


def test_negative_bucket_mass_fails_validation(tmp_path: Path) -> None:
    output = _build_cascaded(tmp_path)

    def mutate(arrays: dict[str, np.ndarray]) -> None:
        arrays["bucket_mass"][0, 0, 0] = -0.1

    _rewrite_shard(output, mutate=mutate)

    report = validate_teacher_textbook(output)
    assert report.status == "fail"
    assert _has_blocker(report.blockers, "bucket_mass must be non-negative")


def test_negative_bucket_count_fails_validation(tmp_path: Path) -> None:
    output = _build_cascaded(tmp_path)

    def mutate(arrays: dict[str, np.ndarray]) -> None:
        arrays["bucket_count"][0, 0, 0] = -1

    _rewrite_shard(output, mutate=mutate)

    report = validate_teacher_textbook(output)
    assert report.status == "fail"
    assert _has_blocker(report.blockers, "bucket_count must be non-negative")


def test_invalid_empty_bucket_sentinel_fails_validation(tmp_path: Path) -> None:
    output = _build_cascaded(tmp_path)

    def mutate(arrays: dict[str, np.ndarray]) -> None:
        empty = arrays["bucket_count"] == 0
        first = np.argwhere(empty)[0]
        arrays["bucket_mean_logp"][tuple(first)] = -1.0

    _rewrite_shard(output, mutate=mutate)

    report = validate_teacher_textbook(output)
    assert report.status == "fail"
    assert _has_blocker(report.blockers, "bucket_mean_logp must be 0.0")


def test_cascaded_topk_duplicate_still_fails_validation(tmp_path: Path) -> None:
    output = _build_cascaded(tmp_path)

    def mutate(arrays: dict[str, np.ndarray]) -> None:
        arrays["top_token_ids"][0, 0, 1] = arrays["top_token_ids"][0, 0, 0]

    _rewrite_shard(output, mutate=mutate)

    report = validate_teacher_textbook(output)
    assert report.status == "fail"
    assert _has_blocker(report.blockers, "duplicates")


def test_dense_and_topk_tail_paths_still_pass(tmp_path: Path) -> None:
    dense = tmp_path / "dense"
    topk = tmp_path / "topk"

    assert build_teacher_textbook(_config(dense)).status == "pass"
    assert build_teacher_textbook(_config(topk, "topk_with_tail_v0")).status == "pass"
    assert validate_teacher_textbook(dense).status == "pass"
    assert validate_teacher_textbook(topk).status == "pass"


def test_default_bucket_edges_contract() -> None:
    config = TeacherTextbookBuildConfig(output_dir=Path("unused"))

    assert config.top_k == 256
    assert config.bucket_edges == (1.0, 1e-3, 1e-6, 1e-9, 1e-12, 0.0)


def _build_cascaded(tmp_path: Path) -> Path:
    output = tmp_path / "cascaded"
    report = build_teacher_textbook(_config(output, "cascaded_soft_labels_v1"))
    assert report.status == "pass"
    return output


def _config(
    output: Path, target_type: str = "dense_logits"
) -> TeacherTextbookBuildConfig:
    return TeacherTextbookBuildConfig(
        output_dir=output,
        teacher_mode="fake",
        sequence_length=8,
        batch_size=2,
        max_examples=4,
        logits_dtype="float32",
        local_files_only=True,
        allow_downloads=False,
        seed=123,
        overwrite=False,
        vocab_size=64,
        target_type=target_type,
        top_k=8,
        top_log_probs_dtype="float32",
    )


def _rewrite_shard(
    output: Path,
    *,
    remove: tuple[str, ...] = (),
    mutate: Callable[[dict[str, np.ndarray]], None] | None = None,
) -> None:
    path = output / "shards" / "shard-00000.npz"
    with np.load(path, allow_pickle=False) as shard:
        arrays = {key: shard[key].copy() for key in shard.files if key not in remove}
    if mutate is not None:
        mutate(arrays)
    np.savez(path, **arrays)


def _has_blocker(blockers: tuple[str, ...], text: str) -> bool:
    return any(text in blocker for blocker in blockers)
