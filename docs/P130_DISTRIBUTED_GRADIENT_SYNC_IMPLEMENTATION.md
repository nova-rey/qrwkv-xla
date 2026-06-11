# P130 Distributed Gradient Sync Implementation

P130 implements a real synchronized gradient/update path for the minimal
first-serious-burn trainer.

## Context

P127 proved cascaded TPU execution. P128 proved deterministic example sharding.
P129 added sync diagnostics and correctly reported the then-current update path
as local-only.

## Current Burn Train State

The first-serious-burn trainer is still intentionally small:

- Trainable parameters: `vocab_bias` and `token_scale`.
- Optimizer: stateless SGD with fixed learning rate `0.05`.
- Student logits: dense full-vocab materialization.
- Targets: dense logits or compressed/cascaded targets through the shared
  target-dispatch loss path.

This is a burn-harness train state, not a full production optimizer stack.

## Synchronization Path

For verified multi-process sharded runs, `distributed_sync=auto` resolves to
`gradient_pmean`. The implementation uses JAX multihost collectives to gather
each process gradient leaf, average across processes, and apply the same
averaged update locally:

```text
local loss/grad -> process_allgather -> mean gradient -> stateless SGD update
```

The same collective path globally averages the scalar loss reported in the
loss trace.

Reported method:

```text
jax.experimental.multihost_utils.process_allgather_mean
```

This is pmean-equivalent for the current tiny burn harness. It is not a Pallas,
WKV, model-quality, or production checkpointing change.

## Verification

P130 keeps P129 fingerprints and adds numeric train-state checksum verification.
After the synchronized update, each process computes a parameter checksum and
uses a multihost all-gather min/max check. Parameter sync is verified only when
the global min and max match.

Optimizer state is explicitly stateless (`stateless_sgd`) and is considered
synchronized when gradient sync is enabled and verified.

Checkpoint fingerprint matching remains reported separately. P130 may claim
training-step synchronization while checkpoint single-writer/export hygiene is
deferred to P131.

## Claims

P130 may set `distributed_training_ready=true` only when:

- distributed example sharding is verified,
- collective communication is verified,
- gradients are globally averaged before update,
- parameter checksum min/max agree across processes,
- stateless optimizer state is explicitly synchronized,
- loss is globally reduced, and
- `jax.process_count() > 1`.

P130 still does not claim model quality, production readiness, large-scale
performance, Qwen support, tokenizer remapping, Pallas default readiness, or
single-writer checkpoint hygiene.
