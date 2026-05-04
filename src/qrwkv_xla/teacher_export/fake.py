from __future__ import annotations

import shutil
from dataclasses import asdict

import numpy as np

from qrwkv_xla.lm.tokenized_corpus import LoadedTokenizedCorpus, load_tokenized_corpus
from qrwkv_xla.targets import (
    TargetFlags,
    TeacherTargetManifest,
    inspect_target_bundle,
    write_target_bundle,
)
from qrwkv_xla.teacher_export.base import ExportRequest, ExportResult
from qrwkv_xla.teacher_export.config import validate_teacher_export_config
from qrwkv_xla.teacher_export.prompts import resolve_prompts


class FakeTeacherExporter:
    name = "fake"

    def export(self, request: ExportRequest) -> ExportResult:
        config = request.config
        validate_teacher_export_config(config)
        tokenized = None
        prompts = None
        if config.targets.tokenized_corpus is not None:
            tokenized = load_tokenized_corpus(
                config.targets.tokenized_corpus,
                expected_sequence_length=config.targets.sequence_length,
            )
            prompt_source = _tokenized_prompt_source(tokenized)
            tokenizer_id = (
                config.teacher.tokenizer_id
                or tokenized.manifest.tokenizer.tokenizer_id
                or tokenized.manifest.tokenizer.backend
            )
            vocab_size = tokenized.manifest.tokenizer.vocab_size
        else:
            prompts = resolve_prompts(config)
            prompt_source = prompts.metadata
            tokenizer_id = config.teacher.tokenizer_id
            vocab_size = config.targets.vocab_size
        attention_metadata = {}
        if config.targets.include_attention_targets:
            attention_metadata = {
                "attention_targets": {
                    "kind": "attention_output_vectors",
                    "shape": "[batch,num_layers,sequence_length,hidden_size]",
                    "semantic": "teacher_attention_output_vectors",
                }
            }

        manifest = TeacherTargetManifest(
            schema_version="0.1",
            teacher_family=config.teacher.family,
            teacher_model_id=config.teacher.resolved_model_id,
            teacher_policy_label=config.teacher.policy_label,
            fallback_policy_label=config.teacher.fallback_label,
            tokenizer_id=tokenizer_id,
            sequence_length=config.targets.sequence_length,
            hidden_size=config.targets.hidden_size,
            num_layers=config.targets.num_layers,
            targets=TargetFlags(
                input_ids=True,
                attention_mask=True,
                loss_mask=True,
                hidden_states=True,
                logits=config.targets.include_logits,
                attention_targets=config.targets.include_attention_targets,
            ),
            dtype=config.targets.dtype,
            created_by="FakeTeacherExporter",
            notes=["deterministic fake exporter bundle"],
            prompt_source=prompt_source,
            extra={"vocab_size": vocab_size, **attention_metadata},
        )

        rng = np.random.default_rng(config.runtime.seed)
        if tokenized is not None:
            shards = list(self._make_tokenized_shards(config, rng, tokenized))
        else:
            shards = [
                self._make_shard(config, rng) for _ in range(config.runtime.num_shards)
            ]

        shutil.rmtree(request.output_dir, ignore_errors=True)
        write_target_bundle(request.output_dir, manifest, shards)
        summary = inspect_target_bundle(request.output_dir)

        return ExportResult(
            output_dir=request.output_dir,
            manifest=manifest,
            shard_count=int(summary["shard_count"]),
            total_examples=int(summary["total_examples"]),
        )

    def _make_shard(
        self,
        config,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        batch_size = config.runtime.batch_size
        sequence_length = config.targets.sequence_length
        arrays: dict[str, np.ndarray] = {
            "input_ids": rng.integers(
                0,
                config.targets.vocab_size,
                size=(batch_size, sequence_length),
                dtype=np.int32,
            ),
            "attention_mask": np.ones((batch_size, sequence_length), dtype=np.int32),
            "loss_mask": np.ones((batch_size, sequence_length), dtype=np.int32),
            "hidden_states": rng.normal(
                loc=0.0,
                scale=1.0,
                size=(
                    batch_size,
                    config.targets.num_layers,
                    sequence_length,
                    config.targets.hidden_size,
                ),
            ).astype(np.float32),
        }
        if config.targets.include_logits:
            arrays["logits"] = rng.normal(
                loc=0.0,
                scale=1.0,
                size=(batch_size, sequence_length, config.targets.vocab_size),
            ).astype(np.float32)
        if config.targets.include_attention_targets:
            arrays["attention_targets"] = rng.normal(
                loc=0.0,
                scale=1.0,
                size=(
                    batch_size,
                    config.targets.num_layers,
                    sequence_length,
                    config.targets.hidden_size,
                ),
            ).astype(np.float32)
        return arrays

    def _make_tokenized_shards(
        self,
        config,
        rng: np.random.Generator,
        tokenized: LoadedTokenizedCorpus,
    ):
        batch_size = config.runtime.batch_size
        for start in range(0, tokenized.input_ids.shape[0], batch_size):
            stop = min(start + batch_size, tokenized.input_ids.shape[0])
            row_count = stop - start
            arrays: dict[str, np.ndarray] = {
                "input_ids": np.ascontiguousarray(
                    tokenized.input_ids[start:stop], dtype=np.int32
                ),
                "attention_mask": np.ascontiguousarray(
                    tokenized.attention_mask[start:stop], dtype=np.int32
                ),
                "loss_mask": np.ascontiguousarray(
                    tokenized.loss_mask[start:stop], dtype=np.int32
                ),
                "hidden_states": rng.normal(
                    loc=0.0,
                    scale=1.0,
                    size=(
                        row_count,
                        config.targets.num_layers,
                        config.targets.sequence_length,
                        config.targets.hidden_size,
                    ),
                ).astype(np.float32),
            }
            if config.targets.include_logits:
                arrays["logits"] = rng.normal(
                    loc=0.0,
                    scale=1.0,
                    size=(
                        row_count,
                        config.targets.sequence_length,
                        tokenized.manifest.tokenizer.vocab_size,
                    ),
                ).astype(np.float32)
            if config.targets.include_attention_targets:
                arrays["attention_targets"] = rng.normal(
                    loc=0.0,
                    scale=1.0,
                    size=(
                        row_count,
                        config.targets.num_layers,
                        config.targets.sequence_length,
                        config.targets.hidden_size,
                    ),
                ).astype(np.float32)
            yield arrays


def _tokenized_prompt_source(tokenized: LoadedTokenizedCorpus) -> dict[str, object]:
    return {
        "type": "tokenized_corpus",
        "prompt_count": tokenized.manifest.source.selected_count,
        "path": str(tokenized.root),
        "source": asdict(tokenized.manifest.source),
        "tokenizer": asdict(tokenized.manifest.tokenizer),
        "packing": asdict(tokenized.manifest.packing),
        "totals": asdict(tokenized.manifest.totals),
        "shards": [asdict(shard) for shard in tokenized.manifest.shards],
    }
