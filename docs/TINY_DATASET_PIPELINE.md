# Tiny Dataset Pipeline

P107 adds a tiny, deterministic dataset pipeline smoke for local/offline
teacher-target plumbing.

## Scope

The pipeline starts from in-memory `TinyTextExample` records, batches them with
`batch_tiny_text_examples()`, emits fake HF-style teacher targets through
`HFTeacherBackend`, writes one `TeacherTargetStore` shard per batch, validates
the multi-shard store through the P106 helpers, and consumes each shard through
the P95 offline target batch path.

Primary APIs:

- `TinyTextExample`: `src/qrwkv_xla/data/tiny_dataset.py`
- `batch_tiny_text_examples()`: `src/qrwkv_xla/data/tiny_dataset.py`
- `run_tiny_dataset_pipeline_smoke()`:
  `src/qrwkv_xla/data/tiny_dataset_pipeline.py`

## Behavior

The baseline smoke uses four tiny text examples, deterministic batching, and
fake in-process tokenizer/model objects. The fake objects exercise the generic
HFTeacherBackend-style boundary without importing transformers, downloading
models, accessing the internet, or requiring GPU/TPU.

For `batch_size=2`, four examples become two target shards:

- batch 0 -> `shards/shard-00000.npz`
- batch 1 -> `shards/shard-00001.npz`

The canonical target-store layout remains:

```text
metadata.json
shards/shard-XXXXX.npz
```

## Consumption Path

P107 reuses the existing target artifact path:

- `TeacherTargetStore.validate()` validates metadata, shard count, shapes, and
  dtypes.
- `iter_target_store_shard_ids()` and `iter_offline_target_batches()` provide
  deterministic P106 multi-shard iteration.
- `load_offline_target_batch()` loads each shard through the P95 offline
  consumption boundary.
- `mse_logits_loss()` computes finite per-shard and aggregate loss values.

## Claims Not Made

P107 does not add or claim:

- large dataset pipeline readiness
- streaming dataset readiness
- production data pipeline readiness
- training or optimizer readiness
- Qwen-specific support
- tokenizer remapping support
- runtime, WKV math, Pallas, fixture, or tolerance changes

Reference remains the default runtime and Pallas remains opt-in.
