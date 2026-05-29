# Real-Teacher Overfit Rehearsal

P103 starts the Real-Teacher Rehearsal and Burn Readiness arc with a tiny,
CPU-friendly update rehearsal driven by a generic `HFTeacherBackend`-style
artifact.

## Purpose

P103 proves generic HFTeacherBackend-style artifacts can drive a tiny
compatibility-gated update rehearsal. It is the real-teacher-style counterpart
to the P96 synthetic-target tiny overfit rehearsal.

## Flow

```text
HFTeacherBackend-style TeacherTargetStore
  -> VocabContract from metadata
  -> StudentBackend registry selection
  -> P99 compatibility gate
  -> P95 offline target batch
  -> student logits
  -> tiny trainable logit head
  -> finite loss movement
```

Baseline tests use fake HF tokenizer/model objects, so `transformers`,
internet, downloaded models, Qwen, GPU, and TPU are not required.

## Compatibility Requirement

The rehearsal validates the store against the selected student contract before
loading a batch or updating. Incompatible contracts return an incompatible
result with no initial loss, final loss, or update.

## Update Strategy

P103 uses `tiny_trainable_logit_head`: a small local adapter over frozen student
logits. The loss is computed before and after real SGD updates through the P95
`mse_logits_loss()` helper.

P103 does not prove full QRWKV parameter training because it does not update
QRWKV parameters.

## What P103 Proves

- generic HFTeacherBackend-style artifacts can drive a tiny update rehearsal
- the artifact vocab contract can select a compatible student through the
  registry path
- the P99 compatibility gate is mandatory before update
- initial and final losses are finite
- a deterministic tiny case moves loss downward
- reference remains default and Pallas remains opt-in

## What P103 Does Not Prove

P103 does not train at scale, prove model quality, add Qwen-specific support,
support tokenizer remapping, add cross-vocab adapters, refactor the trainer,
change recurrence math, change StudentRuntime semantics, change CurrentQRWKV
behavior, or promote Pallas.

## Future Phases

The next phase can add a tiny HF causal-LM teacher specimen smoke. Broader
teacher specimens, multi-shard targets, dataset plumbing, checkpoint/resume,
TPU hygiene, mini eval, and burn readiness remain future work.
