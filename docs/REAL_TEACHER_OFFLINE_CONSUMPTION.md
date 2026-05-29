# Real-Teacher Offline Consumption

P100 adds the first compatibility-gated real-teacher offline consumption smoke.
It consumes generic `HFTeacherBackend`-style artifacts only after the P99
teacher/student compatibility validator says direct logits are eligible.

## Purpose

P100 proves that a generic HF-style teacher artifact can enter the student-side
loss pipe without training, optimizer updates, tokenizer remapping, or
Qwen-specific code.

## Flow

```text
HFTeacherBackend-style artifact
  -> TeacherTargetStore metadata
  -> VocabContract
  -> selected student config
  -> P99 compatibility gate
  -> P95 offline target batch
  -> CurrentQRWKVStudentBackend logits
  -> finite MSE logits loss
```

The implementation lives in
`src/qrwkv_xla/targets/real_teacher_consumption.py`.

## Compatibility Requirement

The smoke calls `validate_store_for_student_config()` before loading the batch
or computing loss. If the artifact and selected student contract do not match,
the result reports `incompatible` and no logits/loss computation is performed.

P100 does not support tokenizer remapping. It does not map vocab A to vocab B,
pad/slice logits, or add cross-vocab adapters.

## What P100 Proves

P100 proves that a generic HFTeacherBackend-style `full_logits` artifact can:

- reconstruct a vocab contract from target metadata
- instantiate a compatible current student config from that contract
- pass the P99 compatibility gate
- load through the P95 offline target path
- produce student logits with the same vocab dimension
- compute a finite MSE logits loss

Baseline tests use fake HF tokenizer/model objects, so `transformers`, internet,
Qwen, GPU/TPU, and downloaded models are not required.

## What P100 Does Not Prove

P100 does not train or update parameters. It does not add optimizer loops,
Qwen-specific support, tokenizer remapping, production distillation, model
quality claims, runtime changes, recurrence math changes, or StudentBackend
behavior changes.

## Future Phases

P101 should make student architecture selectable by architecture id while
preserving the separation between vocab contract, architecture, and runtime.
