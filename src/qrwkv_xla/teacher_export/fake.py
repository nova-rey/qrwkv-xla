from __future__ import annotations

import shutil

import numpy as np

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
        prompts = resolve_prompts(config)
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
            tokenizer_id=config.teacher.tokenizer_id,
            sequence_length=config.targets.sequence_length,
            hidden_size=config.targets.hidden_size,
            num_layers=config.targets.num_layers,
            targets=TargetFlags(
                input_ids=True,
                attention_mask=True,
                hidden_states=True,
                logits=config.targets.include_logits,
                attention_targets=config.targets.include_attention_targets,
            ),
            dtype=config.targets.dtype,
            created_by="FakeTeacherExporter",
            notes=["deterministic fake exporter bundle"],
            prompt_source=prompts.metadata,
            extra={"vocab_size": config.targets.vocab_size, **attention_metadata},
        )

        rng = np.random.default_rng(config.runtime.seed)
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
