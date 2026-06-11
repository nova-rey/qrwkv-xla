# P129 Gradient / Optimizer Synchronization Proof

P129 audits gradient and optimizer synchronization for the guarded real-burn
path.

## Context

P127 proved the cascaded real-burn path can execute on TPU. P128 proved
deterministic example sharding across JAX processes. P128 did not prove that
the processes train one synchronized model.

## Result

The current burn train step is a minimal local JAX update:

- Each process computes loss and gradients on its local shard.
- The train step is not inside a `pmap`/named-axis collective context.
- Gradients are not averaged across processes.
- The optimizer is stateless SGD with no momentum/update accumulator state.
- Checkpoint writing remains per-process existing behavior.

P129 therefore implements synchronization diagnostics and readiness gating, not
true distributed optimizer synchronization.

Added diagnostics:

- `--distributed-sync {auto,none,gradient_pmean}`
- Parameter fingerprints before and after training.
- Stateless optimizer-state fingerprints.
- Checkpoint byte fingerprint.
- Per-process `sync_report_process_<idx>.json`.
- Process-0 `sync_global_report.json` audit metadata.
- Batch/loss semantics fields: local batch, global batch, and `loss_reduction`.
- Readiness gating that keeps `distributed_training_ready=false` unless every
  synchronization predicate is actually proven.

## Claim Boundary

P129 can honestly report that example sharding remains verified and that the
current update path is local-only.

P129 does not prove gradient synchronization, optimizer synchronization,
matching post-step parameter fingerprints across processes, global loss
semantics, single-writer checkpoint safety, model quality, production training
readiness, Qwen support, tokenizer remapping, or WKV/Pallas changes.

A future phase must move the train step into a valid JAX distributed collective
context, average gradients globally before update, and compare per-process
fingerprints before `distributed_training_ready` can become true.
