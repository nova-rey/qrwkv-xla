from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from qrwkv_xla.artifacts import (
    TeacherTextbookBuildConfig,
    build_teacher_textbook,
    validate_teacher_textbook,
)
from qrwkv_xla.artifacts._json import read_json_object
from qrwkv_xla.contracts import vocab_contract_from_metadata
from qrwkv_xla.eval import run_mini_eval_harness
from qrwkv_xla.students import TINY_DEBUG_ARCHITECTURE_ID
from qrwkv_xla.targets import (
    TeacherTargetStore,
    load_teacher_target_batch,
)


def test_fake_builder_writes_topk_tail_target_without_dense_logits(
    tmp_path: Path,
) -> None:
    output = _build_topk(tmp_path)

    metadata = read_json_object(output / "metadata.json")
    manifest = read_json_object(output / "teacher_manifest.json")
    emission = read_json_object(output / "emission_config.json")
    report = read_json_object(output / "validation_report.json")

    assert metadata["target_type"] == "topk_with_tail_v0"
    assert metadata["target_params"]["top_k"] == "8"
    assert metadata["target_params"]["top_log_probs_dtype"] == "float32"
    assert manifest["target_type"] == "topk_with_tail_v0"
    assert manifest["top_k"] == 8
    assert emission["target_type"] == "topk_with_tail_v0"
    assert emission["top_k"] == 8
    assert report["status"] == "pass"
    assert report["compressed_target_ok"] is True
    assert report["mass_ok"] is True
    assert report["sort_ok"] is True
    assert report["duplicate_ok"] is True

    with np.load(output / "shards" / "shard-00000.npz", allow_pickle=False) as shard:
        assert "logits" not in shard.files
        assert shard["input_ids"].shape == (2, 8)
        assert shard["attention_mask"].shape == (2, 8)
        assert shard["top_token_ids"].shape == (2, 8, 8)
        assert shard["top_log_probs"].shape == (2, 8, 8)
        assert shard["top_log_probs"].dtype == np.float32
        assert shard["top_mass"].shape == (2, 8)
        assert shard["tail_mass"].shape == (2, 8)
        assert shard["teacher_entropy"].shape == (2, 8)
        assert np.all(np.diff(shard["top_log_probs"], axis=-1) <= 0)
        assert np.allclose(shard["top_mass"] + shard["tail_mass"], 1.0, atol=1e-5)
        assert np.all(shard["teacher_entropy"] >= 0.0)


def test_dense_logits_path_remains_default_and_valid(tmp_path: Path) -> None:
    output = tmp_path / "dense"

    report = build_teacher_textbook(_config(output))

    assert report.status == "pass"
    metadata = read_json_object(output / "metadata.json")
    manifest = read_json_object(output / "teacher_manifest.json")
    assert metadata["target_type"] == "dense_logits"
    assert manifest["target_type"] == "dense_logits"
    with np.load(output / "shards" / "shard-00000.npz", allow_pickle=False) as shard:
        assert "logits" in shard.files
        assert "top_token_ids" not in shard.files
        assert shard["logits"].shape == (2, 8, 17)


def test_loader_reads_topk_tail_textbook_without_logits_array(tmp_path: Path) -> None:
    output = _build_topk(tmp_path)
    store = TeacherTargetStore.open(output)

    batch = load_teacher_target_batch(store, shard_id=0)

    assert batch.target_type == "topk_with_tail_v0"
    assert batch.teacher_logits is None
    assert batch.top_token_ids is not None
    assert batch.top_log_probs is not None
    assert batch.top_token_ids.shape == (2, 8, 8)


def test_loader_reads_dense_textbook_with_logits_array(tmp_path: Path) -> None:
    output = tmp_path / "dense"
    build_teacher_textbook(_config(output))
    store = TeacherTargetStore.open(output)

    batch = load_teacher_target_batch(store, shard_id=0)

    assert batch.target_type == "dense_logits"
    assert batch.teacher_logits is not None
    assert batch.top_token_ids is None
    assert batch.teacher_logits.shape == (2, 8, 17)


def test_mini_eval_consumes_topk_tail_textbook_and_reports_sparse_metrics(
    tmp_path: Path,
) -> None:
    output = _build_topk(tmp_path)
    store = TeacherTargetStore.open(output)

    result = run_mini_eval_harness(
        store=store,
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
    )

    assert result.status == "pass"
    assert result.teacher_target_type == "topk_with_tail_v0"
    assert result.distillation_loss_type == "topk_tail_head_kl"
    assert result.mean_distillation_loss is not None
    assert result.mean_mse_loss is None
    assert result.head_loss is not None
    assert result.tail_loss == 0.0
    assert result.tail_loss_weight == 0.0
    assert result.top_k == 8
    assert result.mean_top_mass is not None
    assert result.mean_tail_mass is not None
    assert result.mean_teacher_entropy is not None


def test_mini_eval_dense_report_remains_valid(tmp_path: Path) -> None:
    output = tmp_path / "dense"
    build_teacher_textbook(_config(output))
    store = TeacherTargetStore.open(output)

    result = run_mini_eval_harness(
        store=store,
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
    )

    assert result.status == "pass"
    assert result.teacher_target_type == "dense_logits"
    assert result.distillation_loss_type == "dense_logits_kl"
    assert result.mean_mse_loss is not None
    assert result.mean_distillation_loss is not None
    assert result.top_k is None
    assert result.mean_top_mass is None


def test_mini_eval_student_vocab_mismatch_fails_before_sparse_consumption(
    tmp_path: Path,
) -> None:
    output = _build_topk(tmp_path)
    store = TeacherTargetStore.open(output)
    contract = vocab_contract_from_metadata(store.metadata)

    result = run_mini_eval_harness(
        store=store,
        architecture_id=TINY_DEBUG_ARCHITECTURE_ID,
        student_vocab_contract=contract.__class__(
            tokenizer_id=contract.tokenizer_id,
            tokenizer_hash=contract.tokenizer_hash,
            vocab_size=contract.vocab_size + 1,
            model_id=contract.model_id,
            model_family=contract.model_family,
        ),
    )

    assert result.status == "incompatible"
    assert "vocab_size mismatch" in result.compatibility_reason


def test_topk_validator_rejects_missing_required_array(tmp_path: Path) -> None:
    output = _build_topk(tmp_path)

    _rewrite_shard(output, remove=("top_token_ids",))

    report = validate_teacher_textbook(output)
    assert report.status == "fail"
    assert report.compressed_target_ok is False
    assert _has_blocker(report.blockers, "missing required arrays")


def test_topk_validator_rejects_wrong_k_shape(tmp_path: Path) -> None:
    output = _build_topk(tmp_path)

    def mutate(arrays: dict[str, np.ndarray]) -> None:
        arrays["top_log_probs"] = arrays["top_log_probs"][:, :, :7]

    _rewrite_shard(output, mutate=mutate)

    report = validate_teacher_textbook(output)
    assert report.status == "fail"
    assert _has_blocker(report.blockers, "top_log_probs shape must be")


def test_topk_validator_rejects_out_of_range_token_id(tmp_path: Path) -> None:
    output = _build_topk(tmp_path)

    def mutate(arrays: dict[str, np.ndarray]) -> None:
        arrays["top_token_ids"][0, 0, 0] = 999

    _rewrite_shard(output, mutate=mutate)

    report = validate_teacher_textbook(output)
    assert report.status == "fail"
    assert _has_blocker(report.blockers, "within [0, vocab_size)")


def test_topk_validator_rejects_duplicate_token_ids(tmp_path: Path) -> None:
    output = _build_topk(tmp_path)

    def mutate(arrays: dict[str, np.ndarray]) -> None:
        arrays["top_token_ids"][0, 0, 1] = arrays["top_token_ids"][0, 0, 0]

    _rewrite_shard(output, mutate=mutate)

    report = validate_teacher_textbook(output)
    assert report.status == "fail"
    assert _has_blocker(report.blockers, "must not contain duplicates")


def test_topk_validator_rejects_unsorted_log_probs(tmp_path: Path) -> None:
    output = _build_topk(tmp_path)

    def mutate(arrays: dict[str, np.ndarray]) -> None:
        arrays["top_log_probs"][0, 0, 1] = arrays["top_log_probs"][0, 0, 0] + 1.0

    _rewrite_shard(output, mutate=mutate)

    report = validate_teacher_textbook(output)
    assert report.status == "fail"
    assert _has_blocker(report.blockers, "sorted descending")


def test_topk_validator_rejects_nonfinite_log_probs(tmp_path: Path) -> None:
    output = _build_topk(tmp_path)

    def mutate(arrays: dict[str, np.ndarray]) -> None:
        arrays["top_log_probs"][0, 0, 0] = np.inf

    _rewrite_shard(output, mutate=mutate)

    report = validate_teacher_textbook(output)
    assert report.status == "fail"
    assert _has_blocker(report.blockers, "top_log_probs must be finite")


def test_topk_validator_rejects_mass_mismatch(tmp_path: Path) -> None:
    output = _build_topk(tmp_path)

    def mutate(arrays: dict[str, np.ndarray]) -> None:
        arrays["top_mass"][0, 0] = 0.0
        arrays["tail_mass"][0, 0] = 0.0

    _rewrite_shard(output, mutate=mutate)

    report = validate_teacher_textbook(output)
    assert report.status == "fail"
    assert _has_blocker(report.blockers, "top_mass + tail_mass")


def test_topk_validator_rejects_negative_entropy(tmp_path: Path) -> None:
    output = _build_topk(tmp_path)

    def mutate(arrays: dict[str, np.ndarray]) -> None:
        arrays["teacher_entropy"][0, 0] = -1.0

    _rewrite_shard(output, mutate=mutate)

    report = validate_teacher_textbook(output)
    assert report.status == "fail"
    assert _has_blocker(report.blockers, "teacher_entropy must be non-negative")


def _build_topk(tmp_path: Path) -> Path:
    output = tmp_path / "topk"
    report = build_teacher_textbook(
        _config(
            output,
            target_type="topk_with_tail_v0",
            top_k=8,
            top_log_probs_dtype="float32",
        )
    )
    assert report.status == "pass"
    return output


def _config(
    output: Path,
    *,
    target_type: str = "dense_logits",
    top_k: int = 256,
    top_log_probs_dtype: str = "float16",
) -> TeacherTextbookBuildConfig:
    return TeacherTextbookBuildConfig(
        output_dir=output,
        dataset_path=None,
        teacher_mode="fake",
        sequence_length=8,
        batch_size=2,
        max_examples=4,
        logits_dtype="float32",
        local_files_only=True,
        allow_downloads=False,
        seed=123,
        overwrite=False,
        vocab_size=17,
        target_type=target_type,
        top_k=top_k,
        top_log_probs_dtype=top_log_probs_dtype,
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


def test_default_top_k_contract_is_256() -> None:
    config = TeacherTextbookBuildConfig(output_dir=Path("unused"))

    assert config.target_type == "dense_logits"
    assert config.top_k == 256


def test_cli_writes_topk_tail_artifact(tmp_path: Path) -> None:
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "build_teacher_textbook.py"
    output = tmp_path / "cli_topk"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--teacher-mode",
            "fake",
            "--output",
            str(output),
            "--sequence-length",
            "8",
            "--batch-size",
            "2",
            "--max-examples",
            "2",
            "--vocab-size",
            "17",
            "--target-type",
            "topk_with_tail_v0",
            "--top-k",
            "8",
            "--top-log-probs-dtype",
            "float32",
        ],
        check=False,
        env={"PYTHONPATH": str(root / "src")},
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "status=pass" in result.stdout
    assert validate_teacher_textbook(output).status == "pass"
    with np.load(output / "shards" / "shard-00000.npz", allow_pickle=False) as shard:
        assert "logits" not in shard.files
