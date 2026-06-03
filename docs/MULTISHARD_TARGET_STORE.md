# Multi-Shard TargetStore

P106 adds a multi-shard storage and iteration smoke for `TeacherTargetStore`.

## Purpose

P106 proves target artifacts can span multiple shard files and still validate,
iterate, load per shard, and compute finite tiny loss statistics.

## Canonical Layout

The layout remains unchanged:

```text
metadata.json
shards/shard-00000.npz
shards/shard-00001.npz
...
```

P106 does not add a new storage backend or change shard naming.

## Validation Rules

`TeacherTargetStore.validate()` checks metadata, shard count, shard presence,
required arrays, array shapes, vocabulary size, sequence length, example count,
and logits dtype. P106 locks in failures for missing shards, extra shards,
missing expected shard ids, and shape mismatches.

## Iteration / Consumption Flow

`src/qrwkv_xla/targets/multishard.py` exposes:

- `iter_target_store_shard_ids()`
- `iter_offline_target_batches()`
- `run_multishard_target_store_smoke()`
- `MultiShardTargetStoreSmokeResult`

The smoke iterates shard ids in deterministic order, loads each shard through
`load_offline_target_batch()`, computes finite per-shard `mse_logits_loss()`,
and reports aggregate example/loss stats.

## What P106 Proves

- multi-shard target storage/iteration smoke
- deterministic shard listing and iteration
- per-shard offline target batch loading
- finite per-shard and aggregate loss stats
- canonical layout preservation

## What P106 Does Not Prove

P106 does not add a full dataset pipeline, training, optimizer loops,
checkpointing, production storage format, Qwen-specific code, tokenizer
remapping, recurrence math changes, StudentBackend changes, StudentRuntime
changes, or Pallas promotion.

## Future Phases

P107 can add a tiny dataset pipeline smoke that creates sharded teacher-target
artifacts from tiny text examples without adding large-scale training.
