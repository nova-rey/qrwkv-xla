# Artifact Formats

Teacher target artifacts are intended to start with a simple manifest-plus-shards
layout:

```text
artifacts/
  teacher_targets/
    <run_id>/
      manifest.json
      shards/
        shard_000000.npz
        shard_000001.npz
```

The implemented manifest contract for this phase is represented by a
`TeacherTargetManifest` dataclass plus explicit validation helpers.

Example manifest:

```json
{
  "schema_version": "0.1",
  "teacher_family": "qwen",
  "teacher_model_id": null,
  "teacher_policy_label": "Qwen3.latest",
  "fallback_policy_label": "Qwen3.0",
  "tokenizer_id": null,
  "sequence_length": 64,
  "hidden_size": 128,
  "num_layers": 2,
  "targets": {
    "input_ids": true,
    "attention_mask": true,
    "hidden_states": true,
    "logits": false,
    "attention_targets": false
  },
  "dtype": "fp32",
  "created_by": "teacher_exporter",
  "notes": []
}
```

## Notes

- `schema_version`, `teacher_family`, and `teacher_policy_label` must be
  non-empty.
- `sequence_length`, `hidden_size`, and `num_layers` must all be positive.
- `dtype` must currently be one of `fp32`, `bf16`, or `fp16`.
- `targets.input_ids` is required to remain `true` for now.
- Additional unknown top-level manifest fields are preserved under `extra` in
  the dataclass layer.

Real target shard generation is not implemented yet. This phase only establishes
an initial readable, testable contract for later exporter work.
