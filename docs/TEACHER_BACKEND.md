# TeacherBackend

## Purpose

`TeacherBackend` is the teacher-side emission boundary for target artifacts.
It is the source of arrays that can be written into `TeacherTargetStore`.

P94 only proves this boundary with a deterministic synthetic backend. It does
not add a live Hugging Face or Qwen teacher, call external APIs, require
internet, or run training.

## P94 Backend

`SyntheticTeacherBackend` lives in `src/qrwkv_xla/teachers/synthetic.py`.

It emits deterministic tiny arrays:

- `input_ids`: `int32 [N, T]`
- `attention_mask`: `int32 [N, T]`
- `logits`: `float32 [N, T, V]`

The default synthetic metadata uses:

- `model_id`: `synthetic-teacher-v0`
- `model_family`: `synthetic`
- `tokenizer_id`: `synthetic-tokenizer-v0`
- `target_type`: `synthetic`
- `dtype`: `float32`
- `vocab_size`: `8`

## Emission Helper

`emit_teacher_target_store()` lives in `src/qrwkv_xla/teachers/emission.py`.
It asks a backend for metadata and arrays, writes the P93
`TeacherTargetStore` layout, validates it, and returns the reopened store.

```text
SyntheticTeacherBackend
  -> emit_teacher_target_store
  -> TeacherTargetStore
  -> metadata.json + shards/shard-00000.npz
```

## What P94 Proves

- a teacher-side backend boundary can produce target metadata
- a deterministic tiny backend can produce target arrays
- emitted targets can be written to the P93 store layout
- the resulting artifact validates and round-trips on CPU

## What P94 Does Not Do

- no live HF/Qwen teacher
- no model downloads
- no external APIs
- no trainer/loss refactor
- no target consumption training
- no StudentBackend or StudentRuntime behavior change
- no recurrence math, WKV equation, fixture tensor, or tolerance change
- no Pallas promotion

## Future Phases

- P95: offline target consumption smoke
- P96: tiny overfit rehearsal
- P97: small Qwen-family smoke through the modular path
