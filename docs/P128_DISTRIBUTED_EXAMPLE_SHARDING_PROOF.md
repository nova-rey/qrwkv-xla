# P128 Distributed Example Sharding Proof

P128 adds deterministic example sharding for the guarded real-burn path. It
focuses only on data/example assignment across JAX processes.

## Context

P127 verified that the P126 cascaded real-burn path can execute on TPU with 4
JAX processes and 16 TPU devices. P127 passed 25/25 real steps against
`cascaded_soft_labels_v1`, but it still reported unsharded example consumption:
each process appeared to consume the same 100-example textbook.

## Implementation

P128 adds `qrwkv_xla.burn.example_sharding` with deterministic strategies:

- `contiguous_by_process`
- `round_robin_by_process`

The real burn harness now resolves `jax.process_index()` and
`jax.process_count()`, computes a local example shard, slices dense and
cascaded TeacherTextbook arrays to local indices, and reports local/global
example accounting separately.

Default behavior:

- Single process: `single_process`, no distributed example-sharding claim.
- Multi-process: `contiguous_by_process`, with deterministic coverage and
  overlap verification.

Each real process writes:

```text
example_shard_process_<jax_process_index>.json
```

Process 0 also writes:

```text
example_sharding_global_report.json
```

The global report reconstructs every deterministic process shard and verifies
complete coverage with no duplicate example IDs.

## Claims

P128 may claim deterministic example sharding when
`distributed_example_sharding_verified=true`: each JAX process receives a
non-overlapping deterministic local shard and global coverage is verified by
reconstruction.

P128 does not prove optimizer synchronization, synchronized global batch
semantics, model quality, production training readiness, large-scale
performance, Qwen support, tokenizer remapping, WKV/Pallas changes, or full
distributed training correctness.

P129 is reserved for gradient/optimizer synchronization proof.
