from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from qrwkv_xla.data import (
    TinyTextExample,
    batch_tiny_text_examples,
    run_tiny_dataset_pipeline_smoke,
)
from qrwkv_xla.targets import (
    TeacherTargetStore,
    iter_offline_target_batches,
    iter_target_store_shard_ids,
    load_offline_target_batch,
    mse_logits_loss,
    run_multishard_target_store_smoke,
)


def test_tiny_text_example_validates_non_empty_fields() -> None:
    TinyTextExample(example_id="a", text="hello")

    with pytest.raises(ValueError, match="example_id"):
        TinyTextExample(example_id=" ", text="hello")
    with pytest.raises(ValueError, match="text"):
        TinyTextExample(example_id="a", text=" ")


def test_batch_tiny_text_examples_preserves_order_and_final_partial() -> None:
    examples = _examples()[:3]

    batches = batch_tiny_text_examples(examples, batch_size=2)

    assert [[example.example_id for example in batch] for batch in batches] == [
        ["ex-0", "ex-1"],
        ["ex-2"],
    ]


def test_batch_tiny_text_examples_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        batch_tiny_text_examples(_examples(), batch_size=0)


def test_pipeline_writes_one_shard_per_tiny_batch(tmp_path: Path) -> None:
    result = _run_pipeline(tmp_path)

    assert result.example_count == 4
    assert result.batch_count == 2
    assert result.shard_count == 2


def test_pipeline_metadata_records_shard_and_example_counts(tmp_path: Path) -> None:
    result = _run_pipeline(tmp_path)
    store = TeacherTargetStore.open(result.target_store_path)

    assert store.metadata.shard_count == 2
    assert store.metadata.num_examples == 4
    assert store.metadata.provenance["phase"] == "P107"


def test_pipeline_preserves_canonical_target_store_layout(tmp_path: Path) -> None:
    result = _run_pipeline(tmp_path)
    store_path = Path(result.target_store_path)

    assert (store_path / "metadata.json").is_file()
    assert (store_path / "shards" / "shard-00000.npz").is_file()
    assert (store_path / "shards" / "shard-00001.npz").is_file()


def test_pipeline_store_validates_through_p106_smoke(tmp_path: Path) -> None:
    result = _run_pipeline(tmp_path)
    store = TeacherTargetStore.open(result.target_store_path)

    store.validate()
    smoke = run_multishard_target_store_smoke(store)

    assert result.target_store_validated is True
    assert smoke.status == "pass"
    assert smoke.examples_seen == 4


def test_pipeline_iterates_deterministic_shard_ids(tmp_path: Path) -> None:
    result = _run_pipeline(tmp_path)
    store = TeacherTargetStore.open(result.target_store_path)

    assert iter_target_store_shard_ids(store) == (0, 1)


def test_pipeline_loads_each_offline_target_batch(tmp_path: Path) -> None:
    result = _run_pipeline(tmp_path)
    store = TeacherTargetStore.open(result.target_store_path)

    first = load_offline_target_batch(store, shard_id=0)
    second = load_offline_target_batch(store, shard_id=1)

    assert first.input_ids.shape == (2, 5)
    assert second.teacher_logits.shape == (2, 5, 11)


def test_pipeline_computes_finite_per_shard_and_aggregate_loss(
    tmp_path: Path,
) -> None:
    result = _run_pipeline(tmp_path)
    store = TeacherTargetStore.open(result.target_store_path)

    losses = [
        mse_logits_loss(batch.teacher_logits, batch.teacher_logits)
        for batch in iter_offline_target_batches(store)
    ]

    assert len(losses) == 2
    assert all(bool(jnp.isfinite(loss)) for loss in losses)
    assert result.loss_finite is True
    assert result.aggregate_loss == pytest.approx(0.0)


def test_pipeline_report_includes_counts(tmp_path: Path) -> None:
    report = _run_pipeline(tmp_path).to_report()

    assert report["status"] == "pass"
    assert report["example_count"] == 4
    assert report["batch_count"] == 2
    assert report["shard_count"] == 2
    assert report["examples_seen"] == 4


def test_pipeline_requires_no_real_hf_internet_or_accelerator(
    tmp_path: Path,
) -> None:
    result = _run_pipeline(tmp_path)

    assert result.status == "pass"
    assert "large_dataset_pipeline_ready" in result.claims_not_made
    assert "streaming_ready" in result.claims_not_made


def test_pipeline_does_not_claim_training_or_optimizer_ready(
    tmp_path: Path,
) -> None:
    result = _run_pipeline(tmp_path)

    assert "training_ready" in result.claims_not_made
    assert result.scope == "tiny_dataset_pipeline_smoke"


def test_pipeline_does_not_add_qwen_or_tokenizer_remapping(
    tmp_path: Path,
) -> None:
    result = _run_pipeline(tmp_path)

    assert "qwen_specific_support" in result.claims_not_made
    assert "tokenizer_remapping_supported" in result.claims_not_made
    assert "qwen" not in result.teacher_model_id.lower()


def _run_pipeline(tmp_path: Path):
    return run_tiny_dataset_pipeline_smoke(
        examples=_examples(),
        output_dir=tmp_path,
        batch_size=2,
        sequence_length=5,
        vocab_size=11,
    )


def _examples() -> tuple[TinyTextExample, ...]:
    return (
        TinyTextExample(example_id="ex-0", text="hello world"),
        TinyTextExample(example_id="ex-1", text="tiny dataset example"),
        TinyTextExample(example_id="ex-2", text="radjax target smoke"),
        TinyTextExample(example_id="ex-3", text="teacher freezer drawer"),
    )
