# TeacherTargetStore

## Purpose

`TeacherTargetStore` is the P93 offline artifact boundary for teacher targets.
It decouples teacher execution from student runtime by giving future teacher
backends a versioned local format to write and future trainers/evaluators a
stable format to read.

P93 only scaffolds storage. It does not add a live TeacherBackend, load Hugging
Face or Qwen models, refactor training, or consume targets in a training loop.

## Artifact Layout

```text
target_store_dir/
  metadata.json
  shards/
    shard-00000.npz
```

`metadata.json` records the store contract. Shards are NumPy `.npz` files.

## Metadata Fields

The P93 metadata model is `TargetStoreMetadata` in
`src/qrwkv_xla/targets/schema.py`:

- `schema_version`
- `target_store_version`
- `model_id`
- `model_family`
- `tokenizer_id`
- `tokenizer_hash`
- `vocab_size`
- `target_type`
- `dtype`
- `sequence_length`
- `num_examples`
- `shard_count`
- `created_by`
- `created_at`
- `source`
- `provenance`

## Supported P93 Target Types

P93 round-trip validation supports tiny logits-style targets:

- `synthetic`
- `full_logits`

The known target-type vocabulary also reserves names for future phases:

- `full_logprobs`
- `top_k_logprobs`
- `hidden_states`
- `attention_derived`

## Shard Arrays

For `synthetic` and `full_logits`, each shard must contain:

- `input_ids`: integer `[N, T]`
- `attention_mask`: integer or bool `[N, T]`
- `logits`: floating `[N, T, V]`

`T` must match `metadata.sequence_length`, `V` must match
`metadata.vocab_size`, and `logits.dtype` must match `metadata.dtype`.

## Validation Guarantees

P93 validation catches:

- missing `metadata.json`
- unsupported schema/store versions
- unsupported target type
- invalid vocab size, sequence length, example count, or shard count
- missing shard files
- shard count mismatch
- total example count mismatch
- missing required arrays
- shape mismatches
- non-integer token/mask arrays
- non-floating logits
- logits dtype mismatch

## What P93 Does Not Do

- no live TeacherBackend
- no Hugging Face or Qwen loading
- no external model APIs
- no trainer or loss refactor
- no offline target consumption training
- no StudentBackend or StudentRuntime behavior changes
- no recurrence math, WKV equation, fixture tensor, or tolerance changes
- no Pallas promotion

## Future Phases

- P94: tiny TeacherBackend emission smoke
- P95: offline target consumption smoke
- P96: tiny overfit rehearsal
- P97: small Qwen-family smoke through the modular path
