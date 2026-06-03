from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from qrwkv_xla.data.tiny_dataset import TinyTextExample, batch_tiny_text_examples
from qrwkv_xla.targets import (
    TeacherTargetStore,
    run_multishard_target_store_smoke,
)
from qrwkv_xla.teachers import HFTeacherBackend

TINY_DATASET_PIPELINE_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "training_ready",
    "large_dataset_pipeline_ready",
    "streaming_ready",
    "production_data_pipeline_ready",
    "qwen_specific_support",
    "tokenizer_remapping_supported",
)


@dataclass(frozen=True)
class TinyDatasetPipelineResult:
    status: str
    example_count: int
    batch_count: int
    shard_count: int
    examples_seen: int
    aggregate_loss: float
    loss_finite: bool
    target_store_validated: bool
    target_store_path: str
    teacher_model_id: str
    tokenizer_id: str
    vocab_size: int
    sequence_length: int
    claims_not_made: tuple[str, ...] = TINY_DATASET_PIPELINE_CLAIMS_NOT_MADE
    phase: str = "P107"
    scope: str = "tiny_dataset_pipeline_smoke"

    def to_report(self) -> dict[str, Any]:
        return asdict(self)


def run_tiny_dataset_pipeline_smoke(
    *,
    examples: Sequence[TinyTextExample],
    output_dir: str | Path,
    batch_size: int = 2,
    sequence_length: int = 8,
    vocab_size: int = 16,
    model_id: str = "fake-hf-tiny-dataset-teacher",
    tokenizer_id: str = "fake-hf-tiny-dataset-tokenizer",
) -> TinyDatasetPipelineResult:
    if not examples:
        raise ValueError("examples must contain at least one TinyTextExample")
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be > 0, got {sequence_length}")
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be > 0, got {vocab_size}")

    batches = batch_tiny_text_examples(examples, batch_size=batch_size)
    tokenizer = _TinyDatasetFakeTokenizer(
        tokenizer_id=tokenizer_id,
        vocab_size=vocab_size,
    )
    model = _TinyDatasetFakeCausalLM(vocab_size=vocab_size)
    teacher = _fake_hf_teacher(
        model_id=model_id,
        tokenizer=tokenizer,
        model=model,
        prompts=[example.text for example in examples],
    )
    metadata = replace(
        teacher.build_metadata(
            num_examples=len(examples),
            sequence_length=sequence_length,
        ),
        shard_count=len(batches),
        provenance={
            "phase": "P107",
            "backend": teacher.name,
            "scope": "tiny_dataset_pipeline_smoke",
        },
    )

    store_path = Path(output_dir) / "target_store"
    store = TeacherTargetStore.create(store_path, metadata, overwrite=True)
    for shard_id, batch in enumerate(batches):
        shard_teacher = _fake_hf_teacher(
            model_id=model_id,
            tokenizer=tokenizer,
            model=model,
            prompts=[example.text for example in batch],
        )
        store.write_shard(
            shard_id,
            shard_teacher.emit_targets(
                num_examples=len(batch),
                sequence_length=sequence_length,
            ),
        )

    reopened = TeacherTargetStore.open(store.root)
    reopened.validate()
    smoke = run_multishard_target_store_smoke(reopened)
    return TinyDatasetPipelineResult(
        status=smoke.status,
        example_count=len(examples),
        batch_count=len(batches),
        shard_count=smoke.shard_count,
        examples_seen=smoke.examples_seen,
        aggregate_loss=smoke.aggregate_loss,
        loss_finite=smoke.loss_finite,
        target_store_validated=True,
        target_store_path=str(reopened.root),
        teacher_model_id=model_id,
        tokenizer_id=reopened.metadata.tokenizer_id,
        vocab_size=reopened.metadata.vocab_size,
        sequence_length=reopened.metadata.sequence_length,
    )


def _fake_hf_teacher(
    *,
    model_id: str,
    tokenizer: _TinyDatasetFakeTokenizer,
    model: _TinyDatasetFakeCausalLM,
    prompts: Sequence[str],
) -> HFTeacherBackend:
    return HFTeacherBackend(
        model_id=model_id,
        local_files_only=True,
        prompts=prompts,
        tokenizer=tokenizer,
        model=model,
        name="tiny_dataset_fake_hf",
        model_family="fake-hf",
    )


@dataclass
class _TinyDatasetFakeTokenizer:
    tokenizer_id: str
    vocab_size: int
    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int = 0
    unk_token_id: int = 2
    eos_token: str = "<eos>"
    pad_token: str = "<pad>"

    @property
    def name_or_path(self) -> str:
        return self.tokenizer_id

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        prompts: Sequence[str],
        *,
        padding: str,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, np.ndarray]:
        if padding != "max_length":
            raise ValueError("tiny dataset tokenizer only supports max_length padding")
        if not truncation:
            raise ValueError("tiny dataset tokenizer requires truncation=True")
        if return_tensors != "pt":
            raise ValueError("tiny dataset tokenizer expects return_tensors='pt'")
        input_ids = np.stack(
            [
                _encode_text(
                    prompt,
                    max_length=max_length,
                    vocab_size=self.vocab_size,
                )
                for prompt in prompts
            ],
            axis=0,
        ).astype(np.int32)
        return {
            "input_ids": input_ids,
            "attention_mask": np.ones_like(input_ids, dtype=np.int32),
        }


@dataclass
class _TinyDatasetFakeCausalLM:
    vocab_size: int

    def eval(self) -> None:
        return None

    def __call__(
        self,
        *,
        input_ids: Any,
        attention_mask: Any | None = None,
    ) -> SimpleNamespace:
        del attention_mask
        token_ids = np.asarray(input_ids, dtype=np.float32)
        vocab_positions = np.arange(self.vocab_size, dtype=np.float32)
        logits = (
            token_ids[:, :, None] * 0.125 + vocab_positions[None, None, :] * 0.03125
        )
        return SimpleNamespace(logits=logits.astype(np.float32))


def _encode_text(
    text: str,
    *,
    max_length: int,
    vocab_size: int,
) -> np.ndarray:
    seed = sum((index + 1) * ord(char) for index, char in enumerate(text))
    ids = [
        (
            3 + ((seed + position + ord(text[position % len(text)])) % (vocab_size - 3))
            if vocab_size > 3
            else (seed + position) % vocab_size
        )
        for position in range(max_length)
    ]
    return np.asarray(ids, dtype=np.int32)
