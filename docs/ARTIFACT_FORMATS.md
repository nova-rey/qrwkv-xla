# Artifact Formats

Teacher target artifacts are intended to start with a simple manifest-plus-shards layout:

```text
artifacts/
  teacher_targets/
    <run_id>/
      manifest.json
      shards/
        shard_000000.npz
        shard_000001.npz
```

Each manifest should eventually contain fields like:

```json
{
  "schema_version": "0.1",
  "teacher_family": "qwen",
  "teacher_model_id": "resolved-model-id-here",
  "teacher_policy_label": "Qwen3.latest",
  "fallback_policy_label": "Qwen3.0",
  "tokenizer_id": "resolved-tokenizer-id-here",
  "sequence_length": 0,
  "hidden_size": 0,
  "num_layers": 0,
  "targets": {
    "input_ids": true,
    "attention_mask": true,
    "hidden_states": true,
    "logits": false,
    "attention_targets": false
  },
  "dtype": "bf16-or-fp32",
  "created_by": "teacher_exporter",
  "notes": []
}
```

This is a starting contract, not final law from the mountain. The artifact surface should remain stable enough for staged distillation while still allowing schema evolution.
