# Radjax Architecture Extraction Notes

P91 starts the post-Pallas architecture extraction path. The project is still
QRWKV-XLA, but the next phases are shaping it toward a teacher-agnostic and
student-runtime-agnostic Radjax-style platform.

## Current Extracted Boundaries

TeacherBackend:

- `TeacherBackend` protocol: `src/qrwkv_xla/teachers/backend.py`
- `SyntheticTeacherBackend`: `src/qrwkv_xla/teachers/synthetic.py`

This is the source of teacher targets. P94 only includes deterministic
synthetic emission; live HF/Qwen teacher backends are future work.

TeacherTargetStore:

- `TargetStoreMetadata`: `src/qrwkv_xla/targets/schema.py`
- `TeacherTargetStore`: `src/qrwkv_xla/targets/store.py`

This is the versioned offline target artifact contract. P93 stores metadata in
`metadata.json` and target arrays in local `.npz` shards. It decouples teacher
execution from student runtime.

OfflineTargetConsumption:

- `OfflineTargetBatch`: `src/qrwkv_xla/targets/consumption.py`
- `load_offline_target_batch()`
- `mse_logits_loss()`

This is the first student-side consumption boundary for stored teacher
targets. P95 loads validated offline shards and computes a finite logits MSE
against the current student logits surface without requiring a live teacher.

TinyOverfitRehearsal:

- `TinyOverfitResult`: `src/qrwkv_xla/training/tiny_overfit.py`
- `run_tiny_overfit_rehearsal()`

This is the first tiny stored-target update loop. P96 uses offline synthetic
targets, a tiny trainable logit head, P95 logits MSE, and local SGD to prove
loss can move in a deterministic CPU-friendly rehearsal. It does not prove
full QRWKV training readiness.

StudentBackend:

- `StudentBackend` protocol: `src/qrwkv_xla/students/backend.py`
- `CurrentQRWKVStudentBackend`: `src/qrwkv_xla/students/current_backend.py`

This is the architecture-facing wrapper around the validated QRWKV student
core.

StudentRuntime:

- `StudentRuntime` protocol: `src/qrwkv_xla/students/student_runtime.py`
- `ReferenceJaxStudentRuntime`
- `PallasStudentRuntime`

This is the execution path boundary. The current runtime choices are
`reference_jax` and `pallas`.

These wrappers delegate to existing QRWKV student and WKV behavior. They do not
change recurrence math, WKV runtime policy, trainer behavior, teacher export,
target storage, checkpoint formats, Pallas semantics, or fixture tensors.

## Runtime Policy

- `reference` remains the default WKV runtime.
- `pallas` remains opt-in.

## Later Boundaries

Likely future extraction layers:

- small Qwen-family smoke
- broader target support
- real training/eval
- Trainer boundary
- Evaluator and parity gates

P96 implements only a tiny controlled rehearsal. It does not add live teacher
backends, broad trainer extraction, Qwen loading, GPU/TPU requirements, or
model-quality claims.
